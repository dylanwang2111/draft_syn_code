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

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _parse(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(s, errors="coerce")


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
