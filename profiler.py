"""
profiler.py
===========
Structure profiler + synthesis-strategy recommender for the synthetic-data lab.

Goal: make "put in any data" work.  Given one or more tables, this module
reports *only structure* (never raw values, so it is safe to run on
unshareable data) and recommends the highest synthesis tier the data actually
supports:

    Tier 0  independent          nothing linkable -> per-table synthesis
    Tier 1  relational           durable key + FK -> HMA on relationships
    Tier 2  sequential           entity key + time -> PARSynthesizer sequences
    Tier 3  temporal, no key     versioned but unlinkable -> statistical history

The profiler answers, from the data itself:
  * which columns are surrogate/audit ids (unique per row),
  * which are durable-key candidates (repeat across dated version windows),
  * which are plain attributes,
  * how the tables actually join (value-set overlap, cardinality),
  * whether date columns are usable or Excel-mangled,
  * and therefore which tier + strategy to default the UI to.

Only pandas / numpy are used, so it is fast and import-safe (no SDV needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Column-name hints (kept local so the profiler does not depend on synth_eval).
_ID_TOKENS = ("id", "guid", "uuid", "key", "sk", "pk")
_AUDIT_TOKENS = ("audit", "batch", "load", "etl", "job", "run", "seq",
                 "user", "createdby", "updatedby", "createby", "updateby")
_AUDIT_SUFFIX = ("_user", "_by")
_DATE_TOKENS = ("dt", "date", "eff", "end", "start", "expiry", "expire",
                "created", "updated", "timestamp", "ts", "valid")
_DATE_SUFFIX = ("_dt", "_date", "_ts", "_datetime", "_time")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _tokens(col: str) -> List[str]:
    return str(col).lower().replace("-", "_").replace(" ", "_").split("_")


def _name_is_datey(col: str) -> bool:
    c = str(col).lower()
    return c.endswith(_DATE_SUFFIX) or any(t in _DATE_TOKENS for t in _tokens(col))


def _name_is_idish(col: str) -> bool:
    toks = _tokens(col)
    return any(t in _ID_TOKENS for t in toks)


def _name_is_audit(col: str) -> bool:
    c = str(col).lower()
    return c.endswith(_AUDIT_SUFFIX) or any(t in _AUDIT_TOKENS for t in _tokens(col))


def _parse_dt(s: pd.Series) -> pd.Series:
    """Best-effort datetime parse tolerant of mixed formats."""
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(s, errors="coerce")


def _try_datetime(s: pd.Series) -> Tuple[Optional[pd.Series], float]:
    """Attempt to parse a column as *real calendar dates*.

    Returns (parsed_series_or_None, usable_fraction).  Excel-mangled time-like
    strings ('43:45.2', '00:00.0') technically coerce to a time on a single
    day, so we additionally require the parsed values to span more than one
    calendar day and more than a 2-day range — otherwise they are rejected as
    not-real-dates (usable fraction 0).
    """
    non_null = s.dropna()
    if non_null.empty:
        return None, 0.0
    if pd.api.types.is_datetime64_any_dtype(s):
        full = s
        frac = 1.0
    else:
        parsed = _parse_dt(non_null)
        frac = float(parsed.notna().mean())
        if frac < 0.8:
            return None, frac
        full = _parse_dt(s)

    valid = full.dropna()
    if valid.empty:
        return None, 0.0
    # Reject mangled time-only values: real date columns span multiple
    # calendar days, not one day with varying times.
    if valid.dt.normalize().nunique() <= 1:
        return None, 0.0
    if (valid.max() - valid.min()).days < 2:
        return None, 0.0
    return full, frac


# ---------------------------------------------------------------------------
# per-column profile
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_rows: int
    n_distinct: int
    null_frac: float
    uniqueness: float              # n_distinct / n_non_null  (1.0 == unique per row)
    role: str                      # surrogate | audit | durable_candidate | attribute | date | free_text
    datetime_parse_frac: float = 0.0
    note: str = ""

    @property
    def is_unique_per_row(self) -> bool:
        return self.uniqueness >= 0.99


def profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
    s = df[col]
    n = len(s)
    non_null = s.dropna()
    n_nn = len(non_null)
    n_distinct = int(non_null.nunique())
    uniq = float(n_distinct / n_nn) if n_nn else 0.0
    null_frac = float(1 - n_nn / n) if n else 1.0

    # datetime detection (only worth trying if the name or dtype suggests it)
    dt_frac = 0.0
    if _name_is_datey(col) or pd.api.types.is_datetime64_any_dtype(s):
        _, dt_frac = _try_datetime(s)

    # role assignment (order matters: usable date > audit/surrogate > date-named
    # but mangled > free text > durable candidate > plain attribute)
    if dt_frac >= 0.8:
        role = "date"
    elif _name_is_audit(col):
        # audit/system columns (per-row load ids AND low-card update-user
        # columns) are never join keys or entity keys
        role = "audit"
    elif uniq >= 0.99 and (_name_is_idish(col) or pd.api.types.is_integer_dtype(s) or n_nn == n_distinct):
        role = "surrogate"
    elif _name_is_datey(col):
        # date-named but did not parse to real calendar dates -> mangled/placeholder
        role = "date"
    elif uniq >= 0.9 and not pd.api.types.is_numeric_dtype(s):
        role = "free_text"
    elif 1 < n_distinct < n_nn:
        role = "durable_candidate"
    else:
        role = "attribute"

    note = ""
    if role == "date" and dt_frac < 0.8:
        note = "date-named but not parseable as real dates (Excel-mangled?) — unusable for temporal logic"
    if role == "durable_candidate":
        avg_rows = n_nn / n_distinct if n_distinct else 0
        note = f"repeats ~{avg_rows:.1f} rows/value across {n_distinct} values"

    return ColumnProfile(
        name=col, dtype=str(s.dtype), n_rows=n, n_distinct=n_distinct,
        null_frac=round(null_frac, 4), uniqueness=round(uniq, 4), role=role,
        datetime_parse_frac=round(dt_frac, 3), note=note,
    )


# ---------------------------------------------------------------------------
# per-table profile (durable-key + temporal analysis)
# ---------------------------------------------------------------------------

@dataclass
class DurableKeyCandidate:
    column: str
    n_entities: int                # distinct values
    avg_versions: float            # rows per value
    max_versions: int
    dated_window_frac: float = 0.0   # fraction of entities whose rows have distinct effective dates
    contiguity_frac: float = 0.0     # fraction of entities whose windows tile (end≈next start)
    score: float = 0.0             # 0-1 confidence it is a real versioning key


@dataclass
class TableProfile:
    name: str
    n_rows: int
    n_cols: int
    columns: List[ColumnProfile]
    date_columns: List[str] = field(default_factory=list)
    usable_date_columns: List[str] = field(default_factory=list)
    surrogate_columns: List[str] = field(default_factory=list)
    durable_candidates: List[DurableKeyCandidate] = field(default_factory=list)
    is_versioned: bool = False
    note: str = ""


def _analyze_durable_key(df: pd.DataFrame, col: str,
                         eff_col: Optional[str]) -> DurableKeyCandidate:
    grp = df.groupby(col, dropna=True)
    sizes = grp.size()
    n_entities = int(len(sizes))
    avg_v = float(sizes.mean()) if n_entities else 0.0
    max_v = int(sizes.max()) if n_entities else 0

    dated_frac = 0.0
    contig_frac = 0.0
    if eff_col is not None:
        parsed, frac = _try_datetime(df[eff_col])
        if parsed is not None and frac >= 0.8:
            tmp = pd.DataFrame({col: df[col], "_eff": parsed}).dropna(subset=[col])
            multi = tmp.groupby(col)["_eff"]
            # entities whose versions have >1 distinct effective date
            distinct_dates = multi.nunique()
            versioned = sizes[sizes > 1].index
            if len(versioned):
                dated_frac = float(
                    (distinct_dates.reindex(versioned).fillna(0) > 1).mean())

    # Score a *durable/entity key*: MANY entities, each with a FEW versions.
    #   - entity_fit: needs a meaningful number of distinct entities.  A
    #     low-cardinality type code (e.g. 3 values x 500 rows) scores ~0 here.
    #   - version_fit: ideal ~2-15 versions/entity; decays for the huge
    #     versions-per-value of a categorical, and is 0 for ~1 (surrogate-like).
    #   - dated_frac: entities whose rows carry multiple effective dates =
    #     genuine versioning evidence.
    n = max(1, len(df))
    entity_fit = min(1.0, n_entities / max(20.0, 0.05 * n))
    if avg_v < 1.5:
        version_fit = 0.0
    elif avg_v <= 15:
        version_fit = 1.0
    else:
        version_fit = max(0.0, 1.0 - (avg_v - 15) / 85.0)     # ~0 by 100 versions
    score = float(np.clip(0.45 * entity_fit + 0.25 * version_fit + 0.30 * dated_frac, 0, 1))

    return DurableKeyCandidate(
        column=col, n_entities=n_entities, avg_versions=round(avg_v, 2),
        max_versions=max_v, dated_window_frac=round(dated_frac, 3),
        contiguity_frac=round(contig_frac, 3), score=round(score, 3),
    )


def profile_table(df: pd.DataFrame, name: str) -> TableProfile:
    cols = [profile_column(df, c) for c in df.columns]
    date_cols = [c.name for c in cols if c.role == "date"]
    usable_dates = [c.name for c in cols if c.role == "date" and c.datetime_parse_frac >= 0.8]
    surrogates = [c.name for c in cols if c.role in ("surrogate", "audit")]

    # effective-date column guess: a usable date whose name mentions eff/start/valid
    eff_col = next((c for c in usable_dates
                    if any(t in _tokens(c) for t in ("eff", "start", "valid", "from"))),
                   usable_dates[0] if usable_dates else None)

    candidates = []
    for cp in cols:
        if cp.role == "durable_candidate":
            candidates.append(_analyze_durable_key(df, cp.name, eff_col))
    candidates.sort(key=lambda k: k.score, reverse=True)

    # table is "versioned" only if a candidate scores as a plausible entity key
    # (many entities, few dated versions each) -- not merely a repeating column.
    is_versioned = bool(candidates and candidates[0].score >= 0.5)

    note = ""
    if date_cols and not usable_dates:
        note = ("date columns present but none parse as real datetimes "
                "(Excel-mangled) — temporal modelling blocked until re-exported")

    return TableProfile(
        name=name, n_rows=len(df), n_cols=len(df.columns), columns=cols,
        date_columns=date_cols, usable_date_columns=usable_dates,
        surrogate_columns=surrogates, durable_candidates=candidates[:5],
        is_versioned=is_versioned, note=note,
    )


# ---------------------------------------------------------------------------
# cross-table links
# ---------------------------------------------------------------------------

@dataclass
class TableLink:
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    overlap_frac: float            # child values found among parent values
    parent_is_unique: bool         # parent col unique (true PK) -> clean 1:many
    cardinality: str               # "1:1" | "1:many" | "many:many"
    linkable_via_durable: bool     # both sides repeat -> shared durable key, not PK/FK


def _column_values(df: pd.DataFrame, col: str) -> set:
    return set(df[col].dropna().astype(str).unique())


def find_links(tables: Dict[str, pd.DataFrame],
               profiles: Dict[str, TableProfile],
               min_overlap: float = 0.3) -> List[TableLink]:
    """Detect join candidates by value-set overlap on same-named columns.

    Only columns that could plausibly be join keys are considered — surrogate
    ids and durable-key candidates.  Date, audit/user and free-text columns are
    excluded: they overlap trivially (every table shares 'LAST_UPDATE_USER' or
    an effective date) without being real relationships.
    """
    links: List[TableLink] = []
    names = list(tables)
    # per-table set of columns eligible to be a join key
    key_roles = {"surrogate", "durable_candidate"}
    eligible = {
        name: {c.name for c in profiles[name].columns if c.role in key_roles}
        for name in names
    }
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dfa, dfb = tables[a], tables[b]
            # shared columns that are key-eligible on at least one side
            shared = (set(dfa.columns) & set(dfb.columns)) & (eligible[a] | eligible[b])
            for col in shared:
                va, vb = _column_values(dfa, col), _column_values(dfb, col)
                if not va or not vb:
                    continue
                inter = len(va & vb)
                ov_ab = inter / len(vb)   # child=b covered by parent=a
                ov_ba = inter / len(va)
                best = max(ov_ab, ov_ba)
                if best < min_overlap:
                    continue
                a_uniq = dfa[col].is_unique
                b_uniq = dfb[col].is_unique
                if a_uniq and b_uniq:
                    card = "1:1"
                elif a_uniq or b_uniq:
                    card = "1:many"
                else:
                    card = "many:many"
                # parent = the unique side (true PK); else the higher-overlap side
                if b_uniq and not a_uniq:
                    parent, child, pcol, ccol, ov = b, a, col, col, ov_ba
                    p_uniq = b_uniq
                else:
                    parent, child, pcol, ccol, ov = a, b, col, col, ov_ab
                    p_uniq = a_uniq
                links.append(TableLink(
                    parent_table=parent, parent_column=pcol,
                    child_table=child, child_column=ccol,
                    overlap_frac=round(ov, 3), parent_is_unique=bool(p_uniq),
                    cardinality=card, linkable_via_durable=not (a_uniq or b_uniq),
                ))
    links.sort(key=lambda l: l.overlap_frac, reverse=True)
    return links


# ---------------------------------------------------------------------------
# strategy recommendation
# ---------------------------------------------------------------------------

@dataclass
class StrategyRecommendation:
    tier: int
    strategy: str                  # independent | relational | sequential | temporal_statistical
    reasons: List[str]
    relationships: List[dict] = field(default_factory=list)   # ready for HMA
    sequence_specs: Dict[str, dict] = field(default_factory=dict)  # per-table PAR spec
    warnings: List[str] = field(default_factory=list)


def recommend_strategy(tables: Dict[str, pd.DataFrame],
                       profiles: Dict[str, TableProfile],
                       links: List[TableLink]) -> StrategyRecommendation:
    reasons: List[str] = []
    warnings: List[str] = []

    any_versioned = any(p.is_versioned for p in profiles.values())
    any_usable_dates = any(p.usable_date_columns for p in profiles.values())
    # a clean PK/FK link => relational
    pk_links = [l for l in links if l.parent_is_unique and l.overlap_frac >= 0.5]
    durable_links = [l for l in links if l.linkable_via_durable and l.overlap_frac >= 0.5]

    # Tier 2: versioned + usable time + an entity key we can sequence on
    if any_versioned and any_usable_dates:
        seq_specs = {}
        for name, p in profiles.items():
            if p.is_versioned and p.usable_date_columns and p.durable_candidates:
                seq_specs[name] = {
                    "sequence_key": p.durable_candidates[0].column,
                    "sequence_index": p.usable_date_columns[0],
                }
        if seq_specs:
            reasons.append("tables are versioned with parseable effective dates and a "
                           "repeating entity key — model per-entity sequences (PAR)")
            if len(tables) > 1:
                warnings.append("SDV PAR is single-table; run per table and link via the "
                                "shared entity key or the temporal-statistical pass")
            return StrategyRecommendation(
                tier=2, strategy="sequential", reasons=reasons,
                sequence_specs=seq_specs, warnings=warnings)

    # Tier 1: a real PK/FK relationship exists
    if pk_links:
        rels = [{
            "parent_table_name": l.parent_table, "parent_primary_key": l.parent_column,
            "child_table_name": l.child_table, "child_foreign_key": l.child_column,
        } for l in pk_links]
        reasons.append(f"{len(pk_links)} clean primary-key/foreign-key link(s) found — "
                       "model relationally with HMA")
        return StrategyRecommendation(
            tier=1, strategy="relational", reasons=reasons,
            relationships=rels, warnings=warnings)

    # Tier 3: versioned but no durable key we can trust => statistical history
    if any_versioned:
        reasons.append("tables look versioned but no reliable durable key links the "
                       "rows — preserve history *distributions* (versions per entity, "
                       "window lengths, transitions) rather than individual identities")
        if durable_links:
            warnings.append("shared repeating columns exist (" +
                            ", ".join(f"{l.parent_table}.{l.parent_column}" for l in durable_links[:3]) +
                            ") — you could nominate a natural key to lift this to Tier 1/2, "
                            "but entity resolution carries a re-identification risk")
        if not any_usable_dates:
            warnings.append("no parseable date columns — temporal windows cannot be "
                            "reconstructed until dates are re-exported as real datetimes")
        return StrategyRecommendation(
            tier=3, strategy="temporal_statistical", reasons=reasons, warnings=warnings)

    # Tier 0: nothing linkable
    reasons.append("no reliable relationships or versioning detected — synthesize each "
                   "table independently (preserves per-column and intra-row structure)")
    if links:
        warnings.append("weak column overlaps exist but none are clean keys; declare a "
                        "relationship manually if you know one holds")
    return StrategyRecommendation(
        tier=0, strategy="independent", reasons=reasons, warnings=warnings)


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

def profile_dataset(tables: Dict[str, pd.DataFrame]) -> dict:
    """Full profile of a multi-table dataset + a synthesis-strategy recommendation.

    Returns a JSON-safe dict: {tables: {...}, links: [...], recommendation: {...}}.
    Emits only structural statistics — never raw cell values — so it is safe to
    run on unshareable data.
    """
    profiles = {name: profile_table(df, name) for name, df in tables.items()}
    links = find_links(tables, profiles)
    rec = recommend_strategy(tables, profiles, links)

    return {
        "tables": {
            name: {
                "n_rows": p.n_rows, "n_cols": p.n_cols,
                "is_versioned": p.is_versioned,
                "date_columns": p.date_columns,
                "usable_date_columns": p.usable_date_columns,
                "surrogate_columns": p.surrogate_columns,
                "durable_candidates": [vars(k) for k in p.durable_candidates],
                "columns": [vars(c) for c in p.columns],
                "note": p.note,
            }
            for name, p in profiles.items()
        },
        "links": [vars(l) for l in links],
        "recommendation": {
            "tier": rec.tier, "strategy": rec.strategy, "reasons": rec.reasons,
            "relationships": rec.relationships, "sequence_specs": rec.sequence_specs,
            "warnings": rec.warnings,
        },
    }
