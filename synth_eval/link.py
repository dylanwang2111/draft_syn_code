
"""synth_eval.link — post-hoc foreign-key linking for single-table synthesizers.

HMA fits parent and child tables jointly, so referential integrity holds by
construction. A single-table synthesizer (GaussianCopula, CTGAN, TVAE,
CopulaGAN) has no idea the other tables exist: it fits each one
independently, so a child table's foreign-key column just comes out as
whatever values that column's own model reproduced, almost never a real
parent key.

This module fixes that up after the fact: it reassigns each child table's
foreign-key column to values drawn from the *synthetic* parent's primary key,
with the number of children per parent resampled from the real per-parent
count distribution (so the shape, not just the linkage, resembles real data).
It doesn't retrain anything and doesn't touch any other column.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _normalize_key_values(values) -> np.ndarray:
    """Canonical string form for matching key values across a possible real/
    synthetic dtype mismatch -- e.g. a *_TP_CD column read from CSV as
    float64 (348820.0) whose synthesizer output comes back as a plain int
    (348820): naively casting both to str gives "348820.0" vs "348820",
    which never match. Numeric-looking values are routed through float
    first so both sides land on the same string; anything that isn't
    numeric (a real string code) is compared as plain text, unchanged.
    """
    s = pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    out = s.astype(str).to_numpy(dtype=object)
    is_num = num.notna().to_numpy()
    if is_num.any():
        out[is_num] = num[num.notna()].astype(float).astype(str).to_numpy()
    return out


def _real_parent_counts(real_parent: pd.DataFrame, real_child: pd.DataFrame,
                         pk: str, fk: str) -> pd.Series:
    """Real children-per-parent counts, KEYED BY THE PARENT KEY VALUE itself
    (not row position) -- including parents with zero children (they matter
    too: dropping them would overstate how many parents get a child).

    Keyed, not a plain array, so link_table can match a synthetic parent key
    that IS a real value (the normal case for a properly-typed categorical
    key column) to ITS OWN real popularity directly, instead of a randomly
    bootstrapped count from an unrelated key -- see link_table's docstring
    for why that distinction matters.
    """
    counts = real_child[fk].value_counts()
    keys = real_parent[pk]
    vals = keys.map(counts).fillna(0.0).to_numpy(dtype=float)
    s = pd.Series(vals, index=keys.to_numpy())
    return s[~s.index.duplicated(keep="first")]   # defensive: pk should already be unique


def link_table(child_df: pd.DataFrame, fk: str, parent_keys: Sequence,
                real_counts, seed: int = 0) -> pd.DataFrame:
    """Reassign ``child_df[fk]`` to values drawn from ``parent_keys``.

    Every resulting value is a real synthetic parent key (100% referential
    integrity by construction). The number of rows assigned to each parent
    is matched to ``real_counts`` (see :func:`_real_parent_counts`) and
    rescaled so the total matches ``len(child_df)`` exactly — the row count
    a single-table model already generated for this table (which reflects
    the run's `scale`) is left untouched, only the FK values are.

    A synthetic parent key is matched to ITS OWN real count whenever it IS a
    real key value (matched via :func:`_normalize_key_values`, robust to a
    real/synthetic dtype mismatch like ``348820.0`` vs ``348820`` from
    upstream float-vs-int coercion) — not a randomly bootstrapped count from
    an unrelated key. Matching shape-only (the previous behaviour: bootstrap
    every count independently of which key it lands on) preserves the
    AGGREGATE distribution of counts but scrambles WHICH specific value ends
    up common vs rare, destroying that column's own marginal-frequency
    fidelity (Column Shapes) even though the aggregate CardinalityShapeSimilarity
    metric looks fine — verified on a 30-code fixture where the 3 truly most
    common real codes came back as three unrelated, randomly "popular" codes
    under the old bootstrap (TVComplement 0.13); matching by value fixes it.
    A parent key with no real match at all (out-of-vocabulary — not the
    normal case for a properly-typed categorical column) falls back to a
    bootstrap draw from the real pool, same as the old behaviour throughout.
    """
    n = len(child_df)
    parent_keys = np.asarray(list(parent_keys))
    out = child_df.copy()
    if n == 0 or len(parent_keys) == 0:
        return out
    rng = np.random.default_rng(seed)
    if not isinstance(real_counts, pd.Series):
        real_counts = pd.Series(np.asarray(real_counts, dtype=float))
    pool = real_counts.to_numpy(dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0 or pool.sum() <= 0:
        pool = np.array([1.0])   # degenerate fallback: one child each

    lookup = real_counts.set_axis(_normalize_key_values(real_counts.index))
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    weights = lookup.reindex(_normalize_key_values(parent_keys)).to_numpy(dtype=float)
    missing = ~np.isfinite(weights)
    if missing.any():
        weights[missing] = rng.choice(pool, size=int(missing.sum()))

    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(parent_keys))
        total = weights.sum()
    counts = np.floor(weights * (n / total)).astype(int)
    diff = n - counts.sum()
    if diff != 0:
        idx = rng.integers(0, len(counts), size=abs(diff))
        if diff > 0:
            np.add.at(counts, idx, 1)
        else:
            np.subtract.at(counts, idx, 1)
            counts = np.clip(counts, 0, None)

    fks = np.repeat(parent_keys, counts)
    if len(fks) < n:                                    # rounding can leave a few short
        fks = np.concatenate([fks, rng.choice(parent_keys, size=n - len(fks))])
    fks = fks[:n]
    rng.shuffle(fks)

    out[fk] = fks
    return out


def link_relationships(rels: List[dict], real_tables: Dict[str, pd.DataFrame],
                        suite: Dict[str, Dict[str, pd.DataFrame]],
                        synth_names: Sequence[str], seed: int = 0) -> Dict[str, List[str]]:
    """Fix up referential integrity, in place, for the named synthesizers.

    For every declared relationship and every synthesizer in ``synth_names``
    that produced both sides of it, relinks the child table's foreign key
    (see :func:`link_table`). Relationships don't need to be processed in any
    particular order: relinking a child's FK column never touches the parent
    table it points at, so a table that is a parent in one relationship and a
    child in another is handled correctly regardless of ``rels`` order.

    Returns ``{synthesizer: [relationship labels linked]}`` for logging.
    """
    linked: Dict[str, List[str]] = {s: [] for s in synth_names}
    for name in synth_names:
        tabs = suite.get(name)
        if not tabs:
            continue
        for r in rels:
            pt, pk = r["parent_table_name"], r["parent_primary_key"]
            ct, fk = r["child_table_name"], r["child_foreign_key"]
            if pt not in tabs or ct not in tabs or pt not in real_tables or ct not in real_tables:
                continue
            if pk not in tabs[pt].columns or fk not in tabs[ct].columns:
                continue
            if pk not in real_tables[pt].columns or fk not in real_tables[ct].columns:
                continue
            parent_keys = tabs[pt][pk].dropna().unique()
            if len(parent_keys) == 0:
                continue
            real_counts = _real_parent_counts(real_tables[pt], real_tables[ct], pk, fk)
            tabs[ct] = link_table(tabs[ct], fk, parent_keys, real_counts,
                                   seed=seed + (abs(hash((name, pt, ct))) % 10_000))
            linked[name].append(f"{ct}.{fk} → {pt}.{pk}")
    return linked
