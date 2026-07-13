"""synth_eval.columns — column classification and mixed-type encoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Column classification (numeric vs categorical, skip id / name)
# ---------------------------------------------------------------------------

# Substrings that mark a column as an identifier or free-text name.  Such
# columns are excluded from distributions, correlations, MIA/DCR features and
# ML modelling because they carry no distributional signal and hurt privacy
# distance metrics.  NOTE: "code" is intentionally NOT here — in this schema
# *_CODE / *_TP_CD columns are type codes, i.e. genuine categoricals.
ID_NAME_TOKENS = (
    "id",
    "guid",
    "uuid",
    "key",
    "name",
    "fname",
    "lname",
    "surname",
    "email",
    "phone",
    "address",
    "uri",
    "url",
)

# Schema naming conventions (CONTACT / PERSON / PERSONNAME warehouse tables):
#   *_TP_CD, *_TP_CODE, *_CD, *_CODE -> type codes            => categorical
#   *_IND                            -> Y/N indicator flags   => categorical
#   *_ID, *_TX_ID                    -> identifiers           => skip
#   *_DT, *_DATE                     -> dates (often Excel-mangled) => skip
#   *_NAME, *_DESC, *_USER           -> names / free text / audit  => skip
SUFFIX_CATEGORICAL = ("_tp_cd", "_tp_code", "_cd", "_code", "_ind")
SUFFIX_SKIP = ("_id", "_dt", "_date", "_name", "_desc", "_user")


def _looks_like_id_or_name(col: str) -> bool:
    """Heuristic fallback used when metadata does not tell us the sdtype."""
    c = str(col).lower()
    # token match on word boundaries-ish (handle snake / camel / plain)
    parts = c.replace("-", "_").replace(" ", "_").split("_")
    if any(p in ID_NAME_TOKENS for p in parts):
        return True
    return any(c.endswith(tok) or c.startswith(tok) for tok in ("id", "guid", "uuid"))


def _metadata_sdtypes(metadata, table_name: str) -> Dict[str, str]:
    """Extract {column: sdtype} for a table from any SDV metadata flavour."""
    if metadata is None:
        return {}
    try:
        # Modern single-table view or dict-shaped metadata
        d = metadata.to_dict() if hasattr(metadata, "to_dict") else metadata
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    # Multi-table metadata: {"tables": {name: {"columns": {...}}}}
    tables = d.get("tables")
    if isinstance(tables, dict):
        tbl = tables.get(table_name, {})
    else:
        tbl = d
    cols = tbl.get("columns", {}) if isinstance(tbl, dict) else {}
    out = {}
    for col, spec in cols.items():
        if isinstance(spec, dict) and "sdtype" in spec:
            out[col] = spec["sdtype"]
    return out


@dataclass
class ColumnRoles:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # id / name / high-card

    @property
    def modelable(self) -> List[str]:
        return self.numeric + self.categorical


def classify_columns(
    df: pd.DataFrame,
    metadata=None,
    table_name: str = "",
    max_categorical_card: int = 50,
) -> ColumnRoles:
    """Split a table's columns into numeric / categorical / skipped.

    Priority order for each column:
        1. Schema suffix conventions (``*_ID``/``*_DT``/``*_NAME`` -> skip,
           ``*_TP_CD``/``*_CODE``/``*_IND`` -> categorical, with a
           cardinality guard).
        2. SDV metadata sdtype ('id' -> skip, 'numerical' -> numeric,
           'categorical'/'boolean' -> categorical, 'datetime' -> skip).
        3. Name heuristic (looks like an id / name -> skip).
        4. pandas dtype + cardinality.
    """
    sdtypes = _metadata_sdtypes(metadata, table_name)
    roles = ColumnRoles()

    for col in df.columns:
        cl = str(col).lower()
        # 1. schema suffix conventions win over everything else
        if cl.endswith(SUFFIX_SKIP):
            roles.skipped.append(col)
            continue
        if cl.endswith(SUFFIX_CATEGORICAL):
            # a "code" with an id-like number of distinct values is id-like
            if df[col].nunique(dropna=True) > max_categorical_card:
                roles.skipped.append(col)
            else:
                roles.categorical.append(col)
            continue

        sdt = sdtypes.get(col)
        if sdt in {"id", "datetime", "unknown"}:
            roles.skipped.append(col)
            continue
        if sdt == "numerical":
            roles.numeric.append(col)
            continue
        if sdt == "boolean":
            roles.categorical.append(col)
            continue
        if sdt == "categorical":
            # SDV labels free-text name/id columns 'categorical'.  As one-hot
            # features they add no signal and break TSTR (holdout categories
            # unseen in training), so skip them if they look like a name/id or
            # are high-cardinality -- matching the suffix-skip path.
            if _looks_like_id_or_name(col) or df[col].nunique(dropna=True) > max_categorical_card:
                roles.skipped.append(col)
            else:
                roles.categorical.append(col)
            continue

        # No usable metadata -> fall back to heuristics.
        if _looks_like_id_or_name(col):
            roles.skipped.append(col)
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # A numeric column that is really a low-cardinality code is treated
            # as categorical only if it has very few distinct values.
            nunique = series.nunique(dropna=True)
            if nunique <= 10 and set(series.dropna().unique()).issubset(set(range(-1, 11))):
                roles.categorical.append(col)
            else:
                roles.numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            roles.skipped.append(col)
        else:
            nunique = series.nunique(dropna=True)
            if nunique > max_categorical_card:
                # Too many categories (likely free text / near-id) -> skip.
                roles.skipped.append(col)
            else:
                roles.categorical.append(col)
    return roles


# ---------------------------------------------------------------------------
# 1. Mixed-type encoding used by privacy distance metrics & ML models
# ---------------------------------------------------------------------------

def _coerce_frame(df: pd.DataFrame, num: Sequence[str], cat: Sequence[str]) -> pd.DataFrame:
    """Force declared numeric cols to numbers and categorical cols to strings.

    Real-world CSVs are messy: a column SDV calls 'numerical' may hold stray
    text, blanks or Excel artefacts.  We coerce numeric columns with
    ``pd.to_numeric(errors='coerce')`` (bad values -> NaN, handled downstream by
    the imputer) and cast categoricals to ``str`` so one-hot encoding is stable.
    """
    out = df.copy()
    for c in num:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in cat:
        if c in out.columns:
            s = out[c]
            out[c] = s.where(s.isna(), s.astype(str))
    return out


def _fit_mixed_encoder(real: pd.DataFrame, roles: ColumnRoles):
    """Build a fitted sklearn ColumnTransformer for numeric+categorical cols.

    Numeric  -> coerce to number, median impute + standard scale
    Category -> cast to str, most-frequent impute + one-hot (dense, ignore unknown)
    Returned encoder maps a dataframe to a dense float matrix, so it can be
    reused for real/synthetic/holdout consistently.

    Robust to messy data: numeric columns are coerced to numbers first, and any
    column with no non-null value after coercion is dropped (otherwise the
    imputer would silently drop it and hand StandardScaler a 0-width array,
    raising "Found array with 0 feature(s)").  ``keep_empty_features=True`` keeps
    the matrix width stable if a column is empty only in some split.  Raises
    ValueError if nothing usable remains, so callers can skip gracefully.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = [c for c in roles.numeric if c in real.columns]
    cat = [c for c in roles.categorical if c in real.columns]
    real = _coerce_frame(real, num, cat)

    # Drop columns that are entirely empty after coercion -- these carry no
    # signal and are what triggers the 0-feature StandardScaler error.
    num = [c for c in num if real[c].notna().any()]
    cat = [c for c in cat if real[c].notna().any()]
    if not num and not cat:
        raise ValueError("no usable numeric/categorical columns after cleaning "
                         "(all candidate feature columns were empty or non-coercible)")

    # OneHotEncoder arg name changed across sklearn versions.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # keep_empty_features added in sklearn 1.2; fall back if unavailable.
    try:
        num_imp = SimpleImputer(strategy="median", keep_empty_features=True)
        cat_imp = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    except TypeError:  # pragma: no cover - old sklearn
        num_imp = SimpleImputer(strategy="median")
        cat_imp = SimpleImputer(strategy="most_frequent")

    transformers = []
    if num:
        transformers.append(("num", Pipeline([("impute", num_imp), ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([("impute", cat_imp), ("ohe", ohe)]), cat))
    enc = ColumnTransformer(transformers, remainder="drop")
    enc.fit(real[num + cat])
    # remember the per-role columns so _encode can coerce identically
    enc._num_cols, enc._cat_cols = num, cat
    return enc, num + cat


def _encode(enc, df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    X = df.reindex(columns=cols).copy()
    X = _coerce_frame(X, getattr(enc, "_num_cols", []), getattr(enc, "_cat_cols", []))
    mat = enc.transform(X)
    return np.asarray(mat, dtype=float)


