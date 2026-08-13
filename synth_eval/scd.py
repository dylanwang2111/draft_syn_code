"""synth_eval.scd — repair slowly-changing-dimension (SCD-2) timelines.

Cross-table referential integrity (the entity hub) does not fix a table's own
temporal coherence: a synthesizer generates each version row independently, so
one entity's effective/end windows can overlap, leave gaps, or have several
"current" rows.  This module rewrites, per entity, the windows so they tile a
timeline — sorted by effective date, each version's end = the next version's
start, the latest version left "current" — and optionally sets a current flag.

Requires the effective-date column to parse as real dates; on Excel-mangled
date columns it is a no-op (returns a note) rather than producing nonsense.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse(s: pd.Series) -> pd.Series:
    try:
        out = pd.to_datetime(s, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        out = pd.to_datetime(s, errors="coerce")
    # pd.to_datetime's default nanosecond-precision datetime64[ns] can't
    # represent a date past ~2262-04-11 -- a common "current/open" SCD
    # sentinel (e.g. 9999-12-31, the convention used elsewhere in this
    # pipeline) silently comes back NaT here even though it's a perfectly
    # valid date, indistinguishable from genuinely missing/malformed input.
    # Recovered per-value via pd.Timestamp (which pandas 2.x's wider
    # microsecond resolution CAN represent) for whichever values failed the
    # vectorized ns-bound parse above but aren't actually missing.
    missing = out.isna() & s.notna()
    if missing.any():
        recovered = {}
        for idx, val in s[missing].items():
            try:
                recovered[idx] = pd.Timestamp(val)
            except (ValueError, TypeError):
                pass   # genuinely unparseable -- stays NaT, correctly
        if recovered:
            out = out.astype("datetime64[us]")
            for idx, ts in recovered.items():
                out.loc[idx] = ts
    return out


def detect_ordered_date_pairs(
    real: pd.DataFrame, cols: Sequence[str],
    min_support: int = 20, min_ok_ratio: float = 0.99,
) -> List[Tuple[str, str]]:
    """Find ``(low, high)`` column pairs among ``cols`` where ``low <= high``
    holds for virtually every real row with both non-null -- e.g. an
    effective/end date pair on an SCD-versioned table.

    Measured directly from the data, never guessed from column names: this
    schema's own naming convention for "start"/"effective"/"end"/"expiry"
    won't generalize to the next uploaded schema (see the synth-lab-context
    skill's principle 1 -- measure, don't hardcode). Scales as O(k^2)
    comparisons where k = number of date-parseable columns among ``cols``,
    each comparison one vectorized boolean mean over the real column -- fine
    for the handful to a dozen date-like columns a typical table has; not
    meant for (and not needed on) hundreds of candidate columns.

    A pair only qualifies if the ordering holds in ONE direction almost
    always and the OTHER direction does not (``low <= high`` near-total,
    ``high <= low`` well short of it) -- two columns that are both true
    almost always are functionally duplicates or constant, not a meaningful
    low/high relationship, and are excluded rather than arbitrarily picking
    a direction.

    Returned pairs are NOT forced disjoint -- a column can appear in more
    than one (e.g. created <= effective <= end all pairwise qualify at
    once, so ``created`` appears in one pair and ``effective`` in two).
    Forcing disjointness would risk dropping the exact relationship a
    caller cares about whenever it loses a tiebreak to an unrelated pair
    that happens to share a column -- confirmed directly: on real CONTACT
    data, greedily keeping pairs disjoint let ``CREATED_DT``/
    ``IDP_EFFECTIVE_DATE`` claim ``IDP_EFFECTIVE_DATE`` first and silently
    drop ``IDP_EFFECTIVE_DATE``/``IDP_END_DATE`` -- the one pair actually
    named in the bug this exists to catch. Sorted by ordering-ratio then
    support, most confident first; a caller that needs disjoint GROUPS
    (not pairs) should union-find over these as edges instead.
    """
    parsed = {}
    for c in cols:
        if c not in real.columns:
            continue
        col = real[c]
        # a raw int/float column (a surrogate id, an audit tx id, ...) must
        # never reach pd.to_datetime here: it silently reinterprets a large
        # integer as a Unix-epoch timestamp instead of rejecting it as "not
        # a date" -- confirmed directly, a real CONT_ID column (10-digit
        # ints) "parsed" as 100% valid dates near 1970-01-01. Only a
        # string/object column can plausibly BE a date string in the first
        # place; an already-datetime64 column is fine as-is.
        if pd.api.types.is_numeric_dtype(col):
            continue
        raw_present = int(col.notna().sum())
        if raw_present == 0:
            continue
        s = _parse(col)
        # of the values that EXIST, do they mostly parse as real dates? --
        # not "what fraction of every row, including legitimately-null
        # ones" -- an end-date column is routinely mostly null BY DESIGN
        # (most entities are still on their first, still-open version) and
        # that must not disqualify it; confirmed directly: IDP_END_DATE
        # (55% null on real CONTACT data) was being dropped entirely by an
        # earlier version of this check that measured against every row.
        if s.notna().sum() / raw_present >= 0.9:
            parsed[c] = s
    names = list(parsed)
    candidates = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = parsed[a].notna() & parsed[b].notna()
            n = int(both.sum())
            if n < min_support:
                continue
            le = float((parsed[a][both] <= parsed[b][both]).mean())
            ge = float((parsed[b][both] <= parsed[a][both]).mean())
            if le >= min_ok_ratio and ge < min_ok_ratio:
                candidates.append((a, b, le, n))
            elif ge >= min_ok_ratio and le < min_ok_ratio:
                candidates.append((b, a, ge, n))
    candidates.sort(key=lambda t: (-t[2], -t[3]))
    return [(low, high) for low, high, _, _ in candidates]


def detect_scd_window_pair(
    real: pd.DataFrame, entity_key: str, cols: Sequence[str],
    min_entities: int = 5, min_open_frac: float = 0.05,
) -> Optional[Tuple[str, str]]:
    """Find the ``(effective, end)`` pair among ``cols`` that is THIS
    table's own SCD-2 version window for ``entity_key`` -- as opposed to
    some other pair that also happens to satisfy ``low <= high`` (e.g. two
    unrelated audit timestamps like created-at/updated-at).

    ``detect_ordered_date_pairs`` alone can't tell those apart: it only
    checks ordering. Two extra, real-data-measured signals narrow it down:

    1. ``entity_key`` must actually have versioned (multi-row) entities in
       ``real`` -- a table with at most one row per entity has no timeline
       to repair, so there's nothing to detect a window for.
    2. the "high"/end column must carry an "open" signal for a meaningful
       share of rows -- precisely because "not yet closed" is a common
       state for real entities, most of which are still on their latest
       version. Two conventions are recognized, since a source system can
       use either:
       (a) a dominant, repeated sentinel value (e.g. 9999-12-31) -- a
           genuine end-date column lands on it for many rows, whereas a
           per-row audit timestamp is ~always distinct per row and has no
           such spike. Confirmed directly on real PERSONNAME data:
           END_DT/IDP_END_DATE both cluster 93% of rows on one value;
           LAST_UPDATE_DT's most common value covers 0.1% of rows.
       (b) NULL meaning "still open" instead of a sentinel value -- an
           equally common convention this project's own seed data doesn't
           happen to use, but the next uploaded schema's might. Checked as
           its own signal (not folded into the value-dominance count)
           because a NULL-convention end column can otherwise look exactly
           like a sparse, mostly-missing, unrelated column once its NULLs
           are set aside -- the missingness itself, not a repeated value
           among what's left, is what marks it as "open" here.

    Never guesses from column names -- the next uploaded schema's own
    versioning columns won't share this one's naming convention. Returns
    the first ``detect_ordered_date_pairs`` candidate (already sorted,
    most confident first) that also clears either open-signal check, or
    ``None``.
    """
    if entity_key not in real.columns:
        return None
    sizes = real.groupby(entity_key).size()
    if int((sizes >= 2).sum()) < min_entities:
        return None
    for low, high in detect_ordered_date_pairs(real, cols):
        col = real[high]
        if col.empty:
            continue
        null_frac = float(col.isna().mean())
        s = col.dropna()
        top_frac = float(s.value_counts().iloc[0]) / len(s) if not s.empty else 0.0
        if top_frac >= min_open_frac or null_frac >= min_open_frac:
            return (low, high)
    return None


def find_mirror_pair(
    real: pd.DataFrame, low: str, high: str, cols: Sequence[str],
    min_match: float = 0.99,
) -> Optional[Tuple[str, str]]:
    """Find another ``(low2, high2)`` pair among ``cols`` that duplicates
    ``(low, high)`` value-for-value -- e.g. a data-warehouse load-audit
    pair (``IDP_EFFECTIVE_DATE``/``IDP_END_DATE``) that mirrors a business
    pair (``START_DT``/``END_DT``) exactly, a common ETL pattern. Confirmed
    directly: on real PERSONNAME data the two pairs agree on 100% of rows.

    Repairing ``(low, high)`` without also repairing its mirror would trade
    one inconsistency (two "open" rows) for another (the repaired pair and
    its untouched mirror now disagreeing on the same row). Matched by
    value, not name, so it generalizes to any schema's own duplicate
    columns. Returns ``None`` if no column pairs both sides closely enough.
    """
    def _best_match(col: str) -> Optional[str]:
        best, best_score = None, 0.0
        for c in cols:
            if c == col or c == low or c == high or c not in real.columns:
                continue
            both = real[col].notna() & real[c].notna()
            if int(both.sum()) < 20:
                continue
            score = float((real[col][both].astype(str) == real[c][both].astype(str)).mean())
            if score >= min_match and score > best_score:
                best, best_score = c, score
        return best

    low2, high2 = _best_match(low), _best_match(high)
    return (low2, high2) if low2 and high2 else None


def scd_duration_fidelity(
    real: pd.DataFrame, synth: pd.DataFrame, entity_key: str,
    effective_col: str, end_col: str, min_closed: int = 10,
) -> Optional[dict]:
    """How well does ``synth``'s inter-version duration (``end - effective``,
    closed rows only) match ``real``'s, for the SCD pair
    ``(effective_col, end_col)``?

    ``repair_scd_timeline``'s tiling (a version's end = the next version's
    effective date) is structurally correct -- it mirrors genuine real
    closed-row semantics exactly -- but it says nothing about whether the
    SPACING between one entity's own successive versions looks real. That
    spacing comes from wherever the effective dates themselves came from:
    independently synthesizer-generated per row, then in many pipelines
    regrouped into entities post-hoc (entity-hub relinking), neither of
    which has any visibility into "how far apart should this entity's
    versions be". Repair has no way to fix what it's never told.

    Confirmed directly: on real PERSONNAME data, simulating relinking
    (`CONT_ID` reassigned at random) and then running the existing,
    structurally-correct repair still leaves post-repair durations
    diverging from real ones at KS stat 0.215 (p=0.008) -- correctness and
    this kind of distributional fidelity are different questions, and a
    table can have the former without the latter.

    Only closed rows count -- the one open (still-current) row per entity
    has no real duration yet by definition, so it's excluded on both sides
    the same way ``repair_scd_timeline`` identifies it: whichever end value
    is the max (the open sentinel), or NaT under the NULL-open convention
    (already excluded by requiring both columns non-null).

    Returns ``{ks_stat, ks_pvalue, real_median_days, synth_median_days,
    n_real, n_synth}`` -- ``ks_stat`` near 0 means the two duration
    distributions look alike, near 1 means they don't -- or ``None`` if
    either side has fewer than ``min_closed`` closed rows to compare, too
    little to trust a distribution comparison from.
    """
    from scipy import stats

    def _closed_days(df: pd.DataFrame) -> Optional[pd.Series]:
        if entity_key not in df.columns or effective_col not in df.columns or end_col not in df.columns:
            return None
        eff = _parse(df[effective_col])
        end = _parse(df[end_col])
        valid = eff.notna() & end.notna()
        if int(valid.sum()) < 2:
            return None
        open_value = end[valid].max()
        closed = valid & (end < open_value)
        return (end[closed] - eff[closed]).dt.days.astype(float)

    real_days, synth_days = _closed_days(real), _closed_days(synth)
    if real_days is None or synth_days is None \
            or len(real_days) < min_closed or len(synth_days) < min_closed:
        return None

    ks = stats.ks_2samp(real_days, synth_days)
    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "real_median_days": float(real_days.median()),
        "synth_median_days": float(synth_days.median()),
        "n_real": int(len(real_days)),
        "n_synth": int(len(synth_days)),
    }


def repair_scd_timeline(
    df: pd.DataFrame,
    entity_key: str,
    effective_col: str,
    end_col: str,
    current_col: Optional[str] = None,
    current_value: str = "Y",
    noncurrent_value: str = "N",
    open_value: Optional[object] = None,
    date_format: str = "%Y-%m-%d",
) -> Tuple[pd.DataFrame, str]:
    """Return ``(repaired_df, note)``.

    For every ``entity_key`` group, rows are sorted by ``effective_col`` and:
      * ``end_col`` of version *i* is set to the effective date of version *i+1*
        (contiguous, non-overlapping windows),
      * the last version's ``end_col`` is set to ``open_value`` (default: the
        largest real end date seen, else 2999-12-31),
      * ``current_col`` (if given) is set to ``current_value`` on the last
        version and ``noncurrent_value`` elsewhere.
    Effective/end are written back as ``date_format`` strings.

    If fewer than half the effective values parse as dates, the frame is
    returned unchanged with an explanatory note.
    """
    for c in (entity_key, effective_col, end_col):
        if c not in df.columns:
            return df, f"column '{c}' not found — timeline repair skipped"

    eff = _parse(df[effective_col])
    valid = eff.dropna()
    # real dates must parse AND span more than one calendar day; Excel-mangled
    # time-like values ('00:00.0') all collapse to a single day and are rejected.
    if (eff.notna().mean() < 0.5 or valid.empty
            or valid.dt.normalize().nunique() <= 1
            or (valid.max() - valid.min()).days < 2):
        return df, ("effective dates are not real dates (Excel-mangled?) — "
                    "re-export as proper datetimes to enable timeline repair")

    if open_value is None:
        end_parsed = _parse(df[end_col])
        open_value = end_parsed.max() if end_parsed.notna().any() else pd.Timestamp("2999-12-31")
    open_value = pd.Timestamp(open_value)

    work = df.copy()
    work["__eff"] = eff
    work["__ord"] = np.arange(len(work))     # stable tiebreak

    pieces = []
    for _, g in work.groupby(entity_key, dropna=False, sort=False):
        g = g.sort_values(["__eff", "__ord"], kind="stable")
        effs = list(g["__eff"])
        ends = [effs[i + 1] if i + 1 < len(effs) else open_value for i in range(len(effs))]
        g[effective_col] = [d.strftime(date_format) if pd.notna(d) else "" for d in effs]
        g[end_col] = [d.strftime(date_format) if pd.notna(d) else "" for d in ends]
        if current_col and current_col in g.columns and len(g):
            g[current_col] = [noncurrent_value] * (len(g) - 1) + [current_value]
        pieces.append(g)

    out = pd.concat(pieces).drop(columns=["__eff", "__ord"])
    out = out[[c for c in df.columns]]        # original column order
    return out.reset_index(drop=True), ""
