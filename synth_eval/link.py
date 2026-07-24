"""synth_eval.link — post-hoc foreign-key linking for single-table synthesizers.

HMA fits parent and child tables jointly, so referential integrity holds by
construction. A single-table synthesizer (CTGAN/TVAE/CopulaGAN) has no idea
the other tables exist: it fits each one independently, so a child table's
foreign-key column just comes out as whatever values that column's own model
reproduced, almost never a real parent key.

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


def _real_parent_counts(real_parent: pd.DataFrame, real_child: pd.DataFrame,
                         pk: str, fk: str) -> np.ndarray:
    """Real children-per-parent counts, including parents with zero children
    (they matter too: dropping them would overstate how many parents get a
    child). One count per real parent row."""
    counts = real_child[fk].value_counts()
    return real_parent[pk].map(counts).fillna(0).to_numpy()


def link_table(child_df: pd.DataFrame, fk: str, parent_keys: Sequence,
                real_counts: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Reassign ``child_df[fk]`` to values drawn from ``parent_keys``.

    Every resulting value is a real synthetic parent key (100% referential
    integrity by construction). The number of rows assigned to each parent
    is resampled from ``real_counts`` (the real per-parent child-count
    distribution) and rescaled so the total matches ``len(child_df)`` exactly
    — the row count a single-table model already generated for this table
    (which reflects the run's `scale`) is left untouched, only the FK values
    are.
    """
    n = len(child_df)
    parent_keys = np.asarray(list(parent_keys))
    out = child_df.copy()
    if n == 0 or len(parent_keys) == 0:
        return out
    rng = np.random.default_rng(seed)
    real_counts = np.asarray(real_counts, dtype=float)
    real_counts = real_counts[np.isfinite(real_counts)]
    if len(real_counts) == 0 or real_counts.sum() <= 0:
        real_counts = np.array([1.0])   # degenerate fallback: one child each

    # draw a raw count per synthetic parent from the real shape, then rescale
    # so the total lands exactly on the child rows already generated
    draws = rng.choice(real_counts, size=len(parent_keys))
    total = draws.sum()
    if total <= 0:
        draws = np.ones(len(parent_keys))
        total = draws.sum()
    counts = np.floor(draws * (n / total)).astype(int)
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
