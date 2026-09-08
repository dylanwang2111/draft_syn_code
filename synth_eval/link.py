
"""synth_eval.link — post-hoc foreign-key linking for single-table synthesizers.

HMA fits parent and child tables jointly, so referential integrity holds by
construction. A single-table synthesizer (GaussianCopula, CTGAN, TVAE,
CopulaGAN, TabSyn) has no idea the other tables exist: it fits each one
independently, so a child table's foreign-key column just comes out as
whatever values that column's own model reproduced, almost never a real
parent key.

This module fixes that up after the fact, reassigning each child table's
foreign-key column to values drawn from the *synthetic* parent's primary
key -- not by discarding the model's own output and rebuilding the column
from scratch (that gets referential integrity and the real per-parent count
shape right, but at the cost of every row-level relationship the model's own
joint fit learned between this column and the rest of that row, e.g. which
occupation codes skew toward which age group), but by RELABELING it: the
model's own per-row groupings are kept (see :func:`link_table`'s "rank-swap"
step), and only nudged toward the real per-parent counts where the model's
own counts were off (the "rebalance" step), moving the fewest rows needed
rather than reshuffling everything. It doesn't retrain anything and doesn't
touch any other column.
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
    integrity by construction), and the number of rows per parent is matched
    to ``real_counts`` (see :func:`_real_parent_counts`), rescaled so the
    total matches ``len(child_df)`` exactly — the row count a single-table
    model already generated for this table (reflecting the run's `scale`)
    is left untouched, only the FK values are.

    Unlike a pure popularity-weighted reshuffle, this REUSES the model's own
    pre-relink values in ``child_df[fk]`` as row-level structure, instead of
    discarding them outright:

      1. **Rank-swap**: the model's own DISTINCT generated values are ranked
         by how often IT produced them, and relabeled, rank for rank, to the
         real parent keys ranked by real popularity (its most-common value
         becomes the real most-popular key, etc.). This is a pure relabel —
         which ROWS share a value never changes — so any relationship the
         model's own joint fit learned between this column and the REST of
         that row (age, region, whatever else is in the table) survives
         under the new, correctly-popular label instead of being destroyed.
      2. **Rebalance**: rank-swap alone only inherits the model's OWN counts
         per rank, which is exactly what these models tend to get wrong
         (flattening extreme real-world skew) -- so a second pass moves the
         FEWEST rows needed to bring each key's count to its real-weighted
         target, pulled from surplus keys into needy ones. Every row NOT
         selected for a move keeps the label rank-swap gave it.

    A key with no real match at all (out-of-vocabulary) and rows the model
    never produced anything usable for both fall back to a popularity-
    weighted random draw, the same mechanism the old pure-reshuffle
    approach used throughout -- so a model that gives no usable row-level
    signal degrades gracefully to that instead of the hybrid doing nothing.
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
    p = weights / weights.sum()
    # Largest-remainder (Hamilton) apportionment, not floor + a random
    # weighted top-up: floor(weight * n/total) alone systematically zeroes
    # out every key whose real weight is small (e.g. exactly 1 -- the common
    # case for a near-unique key, where most real parents have exactly one
    # child) any time the resampled total lands even slightly above n, which
    # is the ordinary case, not an edge case. A random top-up then only
    # restores a random SUBSET of those zeroed keys (weighted by their own
    # weight, so a weight-1 key competes on equal footing with every other
    # weight-1 key for a limited number of draws) -- confirmed on a real
    # near-unique key where this alone held coverage to ~53% of keys instead
    # of the ~84% real ratio. The leftover units belong to whichever keys'
    # fractional share was closest to rounding up, not a lottery.
    raw = weights * (n / total)
    target = np.floor(raw).astype(int)
    remainder = n - target.sum()   # always >= 0: sum(floor(raw)) <= sum(raw) == n
    if remainder > 0:
        frac = raw - target
        top = np.argsort(-frac, kind="stable")[:remainder]
        target[top] += 1
    elif remainder < 0:  # pragma: no cover - defensive; not reachable in practice
        frac = raw - target
        bottom = np.argsort(frac, kind="stable")[:abs(remainder)]
        target[bottom] = np.clip(target[bottom] - 1, 0, None)

    # 1. rank-swap: relabel the model's own values by frequency rank,
    # keeping every row's original position (a pure `.map`, never a reorder)
    order = np.argsort(-target, kind="stable")
    ranked_keys, ranked_target = parent_keys[order], target[order]
    model_vals = child_df[fk]
    model_ranked = model_vals.value_counts().index.to_numpy()  # dropna=True by default
    k = min(len(model_ranked), len(ranked_keys))
    label_map = {model_ranked[i]: ranked_keys[i] for i in range(k)}
    # a model that produced MORE distinct values than there are real parent
    # keys: the overflow collapses onto the least-popular mapped key rather
    # than being silently dropped
    for i in range(k, len(model_ranked)):
        label_map[model_ranked[i]] = ranked_keys[k - 1] if k else ranked_keys[0]
    fks = model_vals.map(label_map).to_numpy()
    # the only way .map() leaves a gap here is a null in the model's own
    # column (every non-null value is guaranteed a label_map entry, built
    # straight from model_vals' own distinct values above)
    unmapped = model_vals.isna().to_numpy()
    if unmapped.any():
        fks[unmapped] = rng.choice(parent_keys, size=int(unmapped.sum()), p=p)

    # 2. rebalance: move the fewest rows needed from surplus keys to needy
    # ones so the FINAL counts match `target`; every untouched row keeps
    # whichever label the rank-swap gave it
    fks = pd.Series(fks, index=child_df.index)
    current = fks.value_counts()
    target_s = pd.Series(target, index=parent_keys)
    deficit = target_s.subtract(current, fill_value=0)
    surplus = (-deficit[deficit < 0]).astype(int)
    needy = deficit[deficit > 0].astype(int)
    if len(surplus) and len(needy):
        movable = []
        for key, cnt in surplus.items():
            idxs = fks.index[fks.to_numpy() == key].to_numpy()
            take = min(int(cnt), len(idxs))
            if take:
                movable.append(rng.choice(idxs, size=take, replace=False))
        movable = np.concatenate(movable) if movable else np.array([], dtype=fks.index.dtype)
        needy_slots = np.repeat(needy.index.to_numpy(), needy.to_numpy())
        rng.shuffle(needy_slots)
        m = min(len(movable), len(needy_slots))
        if m:
            fks.loc[movable[:m]] = needy_slots[:m]

    out[fk] = fks.to_numpy()
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
