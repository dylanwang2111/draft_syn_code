"""
synth_eval.py
=============
Generic helper functions for evaluating SDV synthetic data across three axes:

    * FIDELITY      -> handled by sdmetrics QualityReport in the notebook
    * VISUALIZATION -> distribution + correlation plots, HTML report
    * PRIVACY       -> Membership Inference Attack, DCR, exact-match, sdmetrics
    * UTILITY (ML)  -> TSTR (Train on Synthetic, Test on Real)

Everything is written to work on ANY dataframe: numeric vs categorical columns
are auto-detected, missing values are handled, and ID / name columns are
skipped where appropriate.  Only pandas, numpy, scikit-learn, matplotlib,
seaborn, sdv and sdmetrics are used.

The module is import-safe: heavy optional dependencies (sdv, sdmetrics) are
imported lazily inside the functions that need them so that, e.g., the plotting
helpers still work in an environment without sdv installed.
"""

from __future__ import annotations

import base64
import io
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless / notebook-safe backend
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    _HAS_SNS = True
except Exception:  # pragma: no cover - seaborn is optional at import time
    _HAS_SNS = False


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

def _fit_mixed_encoder(real: pd.DataFrame, roles: ColumnRoles):
    """Build a fitted sklearn ColumnTransformer for numeric+categorical cols.

    Numeric  -> median impute + standard scale
    Category -> most-frequent impute + one-hot (dense, ignore unknown)
    Returned encoder maps a dataframe to a dense float matrix, so it can be
    reused for real/synthetic/holdout consistently.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = [c for c in roles.numeric if c in real.columns]
    cat = [c for c in roles.categorical if c in real.columns]

    # OneHotEncoder arg name changed across sklearn versions.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", ohe),
                    ]
                ),
                cat,
            )
        )
    enc = ColumnTransformer(transformers, remainder="drop")
    enc.fit(real[num + cat].copy())
    return enc, num + cat


def _encode(enc, df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    X = df.reindex(columns=cols).copy()
    mat = enc.transform(X)
    return np.asarray(mat, dtype=float)


# ---------------------------------------------------------------------------
# 2. VISUALIZATION
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _save_fig(fig, path: str) -> str:
    fig.savefig(path, bbox_inches="tight", dpi=110)
    b64 = _fig_to_base64(fig)  # also closes the figure
    return b64


def plot_column_distribution(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    column: str,
    is_numeric: bool,
    out_png: str,
    metadata=None,
    table_name: str = "",
) -> str:
    """Real-vs-synthetic distribution for one column.

    Tries SDV's ``get_column_plot`` first (saved into the HTML as an
    interactive figure); always also produces a matplotlib PNG so there is a
    portable artefact regardless of plotly/kaleido availability.
    Returns a base64 PNG for embedding into the HTML report.
    """
    real_c = real[column].dropna()
    synth_c = synth[column].dropna() if column in synth.columns else pd.Series([], dtype=real[column].dtype)

    fig, ax = plt.subplots(figsize=(6, 4))
    if is_numeric:
        real_v = pd.to_numeric(real_c, errors="coerce").dropna()
        synth_v = pd.to_numeric(synth_c, errors="coerce").dropna()
        bins = min(40, max(10, int(np.sqrt(max(len(real_v), 1)))))
        ax.hist(real_v, bins=bins, alpha=0.5, density=True, label="real", color="#1f77b4")
        ax.hist(synth_v, bins=bins, alpha=0.5, density=True, label="synthetic", color="#ff7f0e")
        if _HAS_SNS and len(real_v) > 1:
            try:
                sns.kdeplot(real_v, ax=ax, color="#1f77b4", lw=1.5)
                sns.kdeplot(synth_v, ax=ax, color="#ff7f0e", lw=1.5)
            except Exception:
                pass
    else:
        # Categorical -> proportion bar chart, aligned categories.
        rp = real_c.astype(str).value_counts(normalize=True)
        sp = synth_c.astype(str).value_counts(normalize=True)
        cats = list(dict.fromkeys(list(rp.index[:20]) + list(sp.index[:20])))
        rp = rp.reindex(cats, fill_value=0.0)
        sp = sp.reindex(cats, fill_value=0.0)
        x = np.arange(len(cats))
        w = 0.4
        ax.bar(x - w / 2, rp.values, width=w, label="real", color="#1f77b4")
        ax.bar(x + w / 2, sp.values, width=w, label="synthetic", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels([c[:16] for c in cats], rotation=45, ha="right")
    ax.set_title(f"{table_name}.{column}")
    ax.set_ylabel("density" if is_numeric else "proportion")
    ax.legend()
    return _save_fig(fig, out_png)


def try_sdv_column_plot(real, synth, column, metadata, table_name, out_dir) -> Optional[str]:
    """Best-effort SDV get_column_plot -> interactive HTML snippet (or None)."""
    try:
        from sdv.evaluation.single_table import get_column_plot

        single_meta = _single_table_metadata(metadata, table_name)
        fig = get_column_plot(
            real_data=real, synthetic_data=synth, column_name=column, metadata=single_meta
        )
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        # Best-effort static PNG too (needs kaleido).
        try:
            fig.write_image(os.path.join(out_dir, f"sdv_col_{table_name}_{column}.png"))
        except Exception:
            pass
        return html
    except Exception:
        return None


def _single_table_metadata(metadata, table_name: str):
    """Return an SDV single-table metadata object for one table if possible."""
    if metadata is None:
        return None
    # Multi-table metadata exposes .get_table_metadata in recent SDV.
    for attr in ("get_table_metadata",):
        if hasattr(metadata, attr):
            try:
                return getattr(metadata, attr)(table_name)
            except Exception:
                pass
    return metadata


def plot_correlation_heatmaps(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_cols: Sequence[str],
    out_png: str,
    table_name: str = "",
) -> Optional[str]:
    """Side-by-side real vs synthetic Pearson correlation heatmaps."""
    numeric_cols = [c for c in numeric_cols if c in real.columns and c in synth.columns]
    if len(numeric_cols) < 2:
        return None
    r = real[numeric_cols].apply(pd.to_numeric, errors="coerce").corr()
    s = synth[numeric_cols].apply(pd.to_numeric, errors="coerce").corr()

    fig, axes = plt.subplots(1, 2, figsize=(2 + 1.1 * len(numeric_cols) * 2, 1 + 1.0 * len(numeric_cols)))
    for ax, mat, title in ((axes[0], r, "real"), (axes[1], s, "synthetic")):
        if _HAS_SNS:
            sns.heatmap(mat, ax=ax, vmin=-1, vmax=1, cmap="coolwarm", annot=len(numeric_cols) <= 8, fmt=".2f", cbar=True)
        else:
            im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_yticklabels(numeric_cols)
            fig.colorbar(im, ax=ax)
        ax.set_title(f"{table_name} corr ({title})")
    fig.tight_layout()
    return _save_fig(fig, out_png)


def visualize_table(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    table_name: str,
    reports_dir: str,
    metadata=None,
    max_cols: int = 25,
) -> Dict[str, object]:
    """Produce all distribution + correlation figures for one table.

    Returns a dict describing the artefacts (used to assemble the HTML report).
    """
    fig_dir = os.path.join(reports_dir, "figures", table_name)
    os.makedirs(fig_dir, exist_ok=True)

    col_entries = []
    cols_to_plot = (roles.numeric + roles.categorical)[:max_cols]
    for col in cols_to_plot:
        is_num = col in roles.numeric
        png = os.path.join(fig_dir, f"dist_{col}.png")
        b64 = plot_column_distribution(
            real, synth, col, is_num, png, metadata=metadata, table_name=table_name
        )
        sdv_html = try_sdv_column_plot(real, synth, col, metadata, table_name, fig_dir)
        col_entries.append({"column": col, "png": png, "b64": b64, "sdv_html": sdv_html})

    corr_png = os.path.join(fig_dir, "correlation.png")
    corr_b64 = plot_correlation_heatmaps(real, synth, roles.numeric, corr_png, table_name)

    return {
        "table": table_name,
        "columns": col_entries,
        "correlation_png": corr_png if corr_b64 else None,
        "correlation_b64": corr_b64,
    }


def build_html_report(
    table_artifacts: List[Dict],
    out_html: str,
    extra_summary: Optional[Dict] = None,
    comparison_figures: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Assemble a single self-contained HTML report from per-table artefacts.

    ``comparison_figures`` is an optional list of ``(title, base64_png)``
    cross-synthesizer figures rendered at the top of the report.
    """
    parts = [
        "<html><head><meta charset='utf-8'><title>Synthetic Data Report</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#222}"
        "h1{border-bottom:2px solid #444}h2{margin-top:36px;color:#1f4e79}"
        ".grid{display:flex;flex-wrap:wrap;gap:16px}.card{border:1px solid #ddd;"
        "border-radius:8px;padding:8px;background:#fafafa}img{max-width:520px;height:auto}"
        "pre{background:#f4f4f4;padding:12px;border-radius:6px;overflow:auto}</style></head><body>",
        "<h1>Synthetic Data Evaluation Report</h1>",
    ]
    if extra_summary:
        import json

        parts.append("<h2>Consolidated summary</h2>")
        parts.append(f"<pre>{json.dumps(extra_summary, indent=2, default=str)}</pre>")

    if comparison_figures:
        parts.append("<h2>Synthesizer comparison</h2>")
        parts.append("<div class='grid'>")
        for title, b64 in comparison_figures:
            if not b64:
                continue
            parts.append("<div class='card'>")
            parts.append(f"<div><b>{title}</b></div>")
            parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:900px'/>")
            parts.append("</div>")
        parts.append("</div>")

    for art in table_artifacts:
        parts.append(f"<h2>Table: {art['table']}</h2>")
        if art.get("correlation_b64"):
            parts.append("<h3>Correlation (real vs synthetic)</h3>")
            parts.append(f"<img src='data:image/png;base64,{art['correlation_b64']}'/>")
        parts.append("<h3>Column distributions (real vs synthetic)</h3>")
        parts.append("<div class='grid'>")
        for c in art["columns"]:
            parts.append("<div class='card'>")
            parts.append(f"<div><b>{c['column']}</b></div>")
            parts.append(f"<img src='data:image/png;base64,{c['b64']}'/>")
            if c.get("sdv_html"):
                parts.append("<details><summary>SDV interactive plot</summary>")
                parts.append(c["sdv_html"])
                parts.append("</details>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")

    html = "\n".join(parts)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_html


# ---------------------------------------------------------------------------
# 3. PRIVACY METRICS
# ---------------------------------------------------------------------------

def membership_inference_attack(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    k: int = 5,
    random_state: int = 0,
) -> Dict[str, float]:
    """Distance-based Membership Inference Attack.

    Idea: if the synthesizer memorised training rows, training members will sit
    *closer* to the nearest synthetic records than fresh holdout rows do.  We
    build features = distances to the k nearest synthetic records for every
    real record (members + holdout), then train a classifier to tell members
    from non-members.  AUC ~ 0.5 => attacker cannot distinguish => good privacy.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    cols = roles.modelable
    result = {"auc": float("nan"), "accuracy": float("nan"), "n_members": len(train_real),
              "n_holdout": len(holdout_real), "note": ""}
    if not cols or len(holdout_real) < 10 or len(train_real) < 10 or len(synth) < 5:
        result["note"] = "insufficient data / columns for MIA"
        return result

    enc, use_cols = _fit_mixed_encoder(train_real, roles)
    S = _encode(enc, synth, use_cols)
    kk = int(min(k, len(S)))
    nn = NearestNeighbors(n_neighbors=kk).fit(S)

    def feats(df):
        d, _ = nn.kneighbors(_encode(enc, df, use_cols))
        # features: distance to each of the k nearest synth records + summary
        return np.hstack([d, d.mean(axis=1, keepdims=True), d.min(axis=1, keepdims=True)])

    Xm, Xh = feats(train_real), feats(holdout_real)
    # Balance classes by subsampling the larger group.
    n = min(len(Xm), len(Xh))
    rng = np.random.default_rng(random_state)
    mi = rng.choice(len(Xm), n, replace=False)
    hi = rng.choice(len(Xh), n, replace=False)
    X = np.vstack([Xm[mi], Xh[hi]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    result["auc"] = float(roc_auc_score(yte, proba))
    result["accuracy"] = float(accuracy_score(yte, (proba >= 0.5).astype(int)))
    return result


def dcr_distributions(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    out_png: str,
    sample: int = 2000,
    random_state: int = 0,
) -> Dict[str, object]:
    """Distance to Closest Record: real->synth vs real->real baseline + plot."""
    from sklearn.neighbors import NearestNeighbors

    cols = roles.modelable
    info: Dict[str, object] = {"note": ""}
    if not cols or len(synth) < 2 or len(real) < 3:
        info["note"] = "insufficient data for DCR"
        return info

    rng = np.random.default_rng(random_state)
    real_s = real.sample(min(sample, len(real)), random_state=random_state) if len(real) > sample else real

    enc, use_cols = _fit_mixed_encoder(real, roles)
    R = np.nan_to_num(_encode(enc, real_s, use_cols))
    S = np.nan_to_num(_encode(enc, synth, use_cols))
    Rall = np.nan_to_num(_encode(enc, real, use_cols))

    # real -> nearest synthetic
    d_rs, _ = NearestNeighbors(n_neighbors=1).fit(S).kneighbors(R)
    d_rs = d_rs.ravel()
    # real -> nearest OTHER real (baseline): 2 neighbours, drop self (distance 0)
    nn_rr = NearestNeighbors(n_neighbors=2).fit(Rall)
    d_rr, _ = nn_rr.kneighbors(R)
    d_rr = d_rr[:, 1]  # nearest non-self

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.percentile(np.concatenate([d_rs, d_rr]), 99) + 1e-9, 40)
    ax.hist(d_rr, bins=bins, alpha=0.5, density=True, label="real->real (baseline)", color="#2ca02c")
    ax.hist(d_rs, bins=bins, alpha=0.5, density=True, label="real->synthetic (DCR)", color="#ff7f0e")
    ax.set_xlabel("distance to closest record")
    ax.set_ylabel("density")
    ax.set_title("Distance to Closest Record")
    ax.legend()
    b64 = _save_fig(fig, out_png)

    info.update(
        {
            "dcr_real_synth_median": float(np.median(d_rs)),
            "dcr_real_synth_p05": float(np.percentile(d_rs, 5)),
            "dcr_real_real_median": float(np.median(d_rr)),
            "dcr_ratio_median": float(np.median(d_rs) / (np.median(d_rr) + 1e-12)),
            "png": out_png,
            "b64": b64,
            # raw distances kept for cross-synthesizer overlay plots
            # (stripped out of JSON summaries by privacy_report)
            "distances_real_synth": d_rs,
            "distances_real_real": d_rr,
        }
    )
    return info


def exact_match_rate(real: pd.DataFrame, synth: pd.DataFrame, roles: ColumnRoles) -> Dict[str, float]:
    """Fraction of synthetic rows that exactly match a real row (modelable cols)."""
    cols = [c for c in roles.modelable if c in real.columns and c in synth.columns]
    if not cols or len(synth) == 0:
        return {"exact_match_rate": 0.0, "n_exact_matches": 0, "note": "no comparable columns"}
    real_keys = set(map(tuple, real[cols].astype(str).fillna("<NA>").itertuples(index=False, name=None)))
    synth_rows = list(map(tuple, synth[cols].astype(str).fillna("<NA>").itertuples(index=False, name=None)))
    matches = sum(1 for row in synth_rows if row in real_keys)
    return {
        "exact_match_rate": float(matches / len(synth_rows)),
        "n_exact_matches": int(matches),
        "note": "",
    }


def sdmetrics_privacy(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    metadata=None,
    table_name: str = "",
) -> Dict[str, object]:
    """Run sdmetrics single-table privacy metrics that apply generically."""
    out: Dict[str, object] = {}
    single_meta = _single_table_metadata(metadata, table_name)
    meta_dict = None
    try:
        meta_dict = single_meta.to_dict() if hasattr(single_meta, "to_dict") else single_meta
    except Exception:
        meta_dict = None

    # NewRowSynthesis: fraction of synthetic rows that are genuinely new.
    try:
        from sdmetrics.single_table import NewRowSynthesis

        score = NewRowSynthesis.compute(
            real_data=real, synthetic_data=synth, metadata=meta_dict,
            numerical_match_tolerance=0.01,
        )
        out["NewRowSynthesis"] = float(score)
    except Exception as e:  # pragma: no cover
        out["NewRowSynthesis"] = None
        out["NewRowSynthesis_error"] = str(e)[:200]

    # CategoricalCAP: only meaningful with >=1 key field and 1 sensitive field.
    if len(roles.categorical) >= 2:
        try:
            from sdmetrics.single_table import CategoricalCAP

            sensitive = roles.categorical[-1]
            key = [c for c in roles.categorical if c != sensitive][:3]
            cap = CategoricalCAP.compute(
                real_data=real, synthetic_data=synth,
                key_fields=key, sensitive_fields=[sensitive],
            )
            out["CategoricalCAP"] = float(cap)
            out["CategoricalCAP_fields"] = {"key": key, "sensitive": sensitive}
        except Exception as e:  # pragma: no cover
            out["CategoricalCAP"] = None
            out["CategoricalCAP_error"] = str(e)[:200]
    return out


def privacy_report(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    table_name: str,
    reports_dir: str,
    metadata=None,
) -> Dict[str, object]:
    """Full privacy module for one table, plus pass/warn verdicts."""
    fig_dir = os.path.join(reports_dir, "figures", table_name)
    os.makedirs(fig_dir, exist_ok=True)

    # MIA uses train (members) vs holdout (non-members).
    mia = membership_inference_attack(train_real, holdout_real, synth, roles)
    # DCR / exact-match compare the *training* real data with synthetic.
    dcr = dcr_distributions(train_real, synth, roles, os.path.join(fig_dir, "dcr.png"))
    exact = exact_match_rate(train_real, synth, roles)
    sdm = sdmetrics_privacy(train_real, synth, roles, metadata, table_name)

    verdicts = {}
    # MIA AUC close to 0.5 is good.
    auc = mia.get("auc")
    if auc is None or np.isnan(auc):
        verdicts["mia"] = ("SKIP", mia.get("note", ""))
    elif abs(auc - 0.5) <= 0.10:
        verdicts["mia"] = ("PASS", f"AUC={auc:.3f} (~0.5 => members indistinguishable)")
    elif abs(auc - 0.5) <= 0.20:
        verdicts["mia"] = ("WARN", f"AUC={auc:.3f} (some membership signal)")
    else:
        verdicts["mia"] = ("FAIL", f"AUC={auc:.3f} (strong membership signal)")

    emr = exact.get("exact_match_rate", 0.0)
    verdicts["exact_match"] = (
        "PASS" if emr <= 0.001 else "WARN" if emr <= 0.01 else "FAIL",
        f"{emr:.4%} of synthetic rows exactly match a real row",
    )

    ratio = dcr.get("dcr_ratio_median")
    if ratio is None:
        verdicts["dcr"] = ("SKIP", dcr.get("note", ""))
    else:
        verdicts["dcr"] = (
            "PASS" if ratio >= 0.9 else "WARN" if ratio >= 0.5 else "FAIL",
            f"median DCR(real->synth)/DCR(real->real)={ratio:.2f} (>=1 is safe)",
        )

    nrs = sdm.get("NewRowSynthesis")
    if nrs is not None:
        verdicts["new_row_synthesis"] = (
            "PASS" if nrs >= 0.9 else "WARN" if nrs >= 0.7 else "FAIL",
            f"NewRowSynthesis={nrs:.3f} (fraction of novel synthetic rows)",
        )

    return {
        "table": table_name,
        "membership_inference": mia,
        "dcr": {
            k: v
            for k, v in dcr.items()
            if k != "b64" and not k.startswith("distances")
        },
        "dcr_arrays": {
            "real_synth": dcr.get("distances_real_synth"),
            "real_real": dcr.get("distances_real_real"),
        },
        "exact_match": exact,
        "sdmetrics": sdm,
        "verdicts": {k: {"status": s, "detail": d} for k, (s, d) in verdicts.items()},
    }


# ---------------------------------------------------------------------------
# 4. ML EFFICACY (TSTR: Train on Synthetic, Test on Real)
# ---------------------------------------------------------------------------

def auto_select_target(df: pd.DataFrame, roles: ColumnRoles) -> Optional[Tuple[str, str]]:
    """Pick a modelling target: (column, task) where task in {classification, regression}.

    Prefers a categorical column with 2-20 classes (classification); otherwise
    falls back to a numeric column with reasonable variance (regression).
    """
    for col in roles.categorical:
        nun = df[col].nunique(dropna=True)
        if 2 <= nun <= 20:
            return col, "classification"
    # regression fallback: numeric column with the most variance
    best, best_var = None, -1.0
    for col in roles.numeric:
        v = pd.to_numeric(df[col], errors="coerce")
        if v.notna().sum() < 20:
            continue
        var = float(v.var())
        if var > best_var:
            best, best_var = col, var
    if best is not None:
        return best, "regression"
    return None


def _build_model(task: str, kind: str, random_state: int = 0):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task == "classification":
        if kind == "rf":
            return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        return LogisticRegression(max_iter=1000)
    else:
        if kind == "rf":
            return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        return LinearRegression()


def _prep_xy(df: pd.DataFrame, target: str, feature_roles: ColumnRoles):
    feats = [c for c in feature_roles.modelable if c != target and c in df.columns]
    return df[feats].copy(), df[target].copy(), feats


def _score(task, y_true, y_pred, y_proba=None):
    from sklearn.metrics import (
        accuracy_score, f1_score, r2_score, roc_auc_score, mean_squared_error,
    )

    if task == "classification":
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted")),
        }
        try:
            if y_proba is not None:
                classes = np.unique(y_true)
                if len(classes) == 2:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            out["roc_auc"] = float("nan")
        return out
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {"rmse": rmse, "r2": float(r2_score(y_true, y_pred))}


def ml_efficacy_tstr(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    target: str,
    task: str,
    table_name: str = "",
    random_state: int = 0,
) -> pd.DataFrame:
    """TSTR: train on real vs synthetic, always test on the SAME real holdout.

    ``synth`` may be a single dataframe (labelled 'synthetic') or a dict of
    ``{synthesizer_name: dataframe}`` to benchmark several synthesizers in one
    tidy comparison table (one row per model x train-source).
    """
    from sklearn.pipeline import Pipeline

    feature_roles = ColumnRoles(
        numeric=[c for c in roles.numeric if c != target],
        categorical=[c for c in roles.categorical if c != target],
    )
    rows = []

    # Test set is always the real holdout.
    Xte, yte, feats = _prep_xy(holdout_real, target, feature_roles)
    if not feats:
        return pd.DataFrame([{"table": table_name, "note": "no usable feature columns"}])

    # Align target dtype for classification (stringify to avoid mixed types).
    def _yfix(y):
        return y.astype(str) if task == "classification" else pd.to_numeric(y, errors="coerce")

    yte = _yfix(yte)

    if isinstance(synth, dict):
        sources = {"real": train_real, **synth}
    else:
        sources = {"real": train_real, "synthetic": synth}
    for kind in ("rf", "baseline"):
        for src_name, src_df in sources.items():
            if target not in src_df.columns:
                continue
            Xtr, ytr, _ = _prep_xy(src_df, target, feature_roles)
            ytr = _yfix(ytr)
            # Drop rows with missing target.
            m = ytr.notna()
            Xtr, ytr = Xtr[m.values], ytr[m.values]
            if len(ytr) < 10 or (task == "classification" and ytr.nunique() < 2):
                continue
            enc, use_cols = _fit_mixed_encoder(train_real, feature_roles)
            model = _build_model(task, kind, random_state)
            pipe = Pipeline([("enc", enc), ("model", model)])
            try:
                pipe.fit(Xtr[use_cols], ytr)
                yp = pipe.predict(Xte[use_cols])
                proba = None
                if task == "classification" and hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(Xte[use_cols])
                    except Exception:
                        proba = None
                mask = yte.notna()
                sc = _score(task, yte[mask.values], np.asarray(yp)[mask.values],
                            None if proba is None else proba[mask.values])
            except Exception as e:  # pragma: no cover
                sc = {"error": str(e)[:200]}
            rows.append({
                "table": table_name, "target": target, "task": task,
                "model": "RandomForest" if kind == "rf" else "baseline",
                "train_on": src_name, **sc,
            })
    df = pd.DataFrame(rows)

    # Add real-vs-synthetic efficacy gap per model x synthetic source.
    gap_rows = []
    metric = "accuracy" if task == "classification" else "r2"
    if not df.empty and "model" in df.columns and metric in df.columns:
        synth_names = [k for k in sources if k != "real"]
        for model in df["model"].unique():
            sub = df[df["model"] == model]
            r = sub[sub["train_on"] == "real"][metric]
            for sname in synth_names:
                s = sub[sub["train_on"] == sname][metric]
                if len(r) and len(s) and pd.notna(r.iloc[0]) and pd.notna(s.iloc[0]):
                    gap_rows.append({
                        "table": table_name, "target": target, "task": task,
                        "model": model, "train_on": f"gap(real-{sname})",
                        metric: float(r.iloc[0] - s.iloc[0]),
                    })
    if gap_rows:
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)
    return df


def sdmetrics_ml_efficacy(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth,
    roles: ColumnRoles,
    target: str,
    task: str,
    table_name: str = "",
    max_train_rows: int = 20000,
) -> pd.DataFrame:
    """Condensed TSTR using **sdmetrics** ML-efficacy metrics.

    Metric selection (kept deliberately small):
        binary target      -> BinaryDecisionTreeClassifier + BinaryLogisticRegression (F1)
        multiclass target  -> MulticlassDecisionTreeClassifier (macro F1)
        numeric target     -> LinearRegression (R^2)

    Each metric's model is trained on (a) the real training split (TRTR
    reference) and (b) each synthesizer's data (TSTR), and always evaluated
    on the SAME real holdout via ``Metric.compute(test_data, train_data,
    target=...)``.  Only modelable columns (id/name/date columns dropped) are
    passed in.  Returns a tidy frame with ``gap(real-<synth>)`` rows appended.
    """
    from sdmetrics import single_table as st

    if isinstance(synth, dict):
        sources = {"real": train_real, **synth}
    else:
        sources = {"real": train_real, "synthetic": synth}

    nun = int(train_real[target].nunique(dropna=True))
    if task == "classification" and nun == 2:
        metric_classes = {
            "BinaryDecisionTreeClassifier (F1)": st.BinaryDecisionTreeClassifier,
            "BinaryLogisticRegression (F1)": st.BinaryLogisticRegression,
        }
    elif task == "classification":
        metric_classes = {
            "MulticlassDecisionTreeClassifier (macro F1)": st.MulticlassDecisionTreeClassifier,
        }
    else:
        metric_classes = {"LinearRegression (R2)": st.LinearRegression}

    cols = [c for c in roles.modelable if c in holdout_real.columns]
    if target not in cols:
        cols.append(target)
    test = holdout_real[cols].dropna(subset=[target]).copy()

    cat_cols = [c for c in cols if c in roles.categorical]

    def _align_categories(tr, te):
        """Make ``te`` safe for a model trained on ``tr``.

        sdmetrics' ML-efficacy metrics one-hot encode with handle_unknown=
        'error', so any category present in the real holdout but absent from the
        training data (common in TSTR when a synthesizer never generated a rare
        value) raises "Found unknown categories".  We map such test-only
        categories to the training column's most frequent value so the metric
        computes; returns the aligned test frame and how many columns changed.
        """
        te = te.copy()
        n_aligned = 0
        for c in cat_cols:
            if c == target or c not in tr.columns:
                continue
            known = set(tr[c].dropna().astype(str).unique())
            col_str = te[c].astype(str)
            mask = ~col_str.isin(known) & te[c].notna()
            if mask.any():
                mode = tr[c].dropna().astype(str).mode()
                fill = mode.iloc[0] if len(mode) else next(iter(known), "")
                te.loc[mask, c] = fill
                n_aligned += 1
        return te, n_aligned

    rows = []
    for mname, M in metric_classes.items():
        for src, df in sources.items():
            if target not in df.columns:
                continue
            tr = df[[c for c in cols if c in df.columns]].dropna(subset=[target]).copy()
            if len(tr) > max_train_rows:
                tr = tr.sample(max_train_rows, random_state=0)
            if len(tr) < 10 or len(test) < 5:
                continue
            # drop test rows whose TARGET class was never seen in training
            # (an unseen label cannot be predicted and would crash scoring)
            train_labels = set(tr[target].dropna().astype(str).unique())
            test_src = test[test[target].astype(str).isin(train_labels)].copy()
            test_src, n_aligned = _align_categories(tr, test_src)
            note = ""
            if len(test_src) < len(test):
                note = f"dropped {len(test)-len(test_src)} holdout rows with unseen target class"
            if n_aligned:
                note = (note + "; " if note else "") + \
                    f"aligned {n_aligned} feature col(s) with unseen categories to train mode"
            try:
                if len(test_src) < 5:
                    raise ValueError("too few holdout rows share the training classes")
                score = float(M.compute(test_data=test_src, train_data=tr, target=target))
            except Exception as e:  # pragma: no cover - defensive
                score = float("nan")
                note = (note + "; " if note else "") + str(e)[:140]
            rows.append({
                "table": table_name, "target": target, "task": task,
                "metric": mname, "train_on": src, "score": score, "note": note,
            })
    out = pd.DataFrame(rows)

    # gap(real - synth) per metric x synthetic source
    gaps = []
    if not out.empty:
        for mname in metric_classes:
            sub = out[out["metric"] == mname]
            r = sub[sub["train_on"] == "real"]["score"]
            for sname in [k for k in sources if k != "real"]:
                s = sub[sub["train_on"] == sname]["score"]
                if len(r) and len(s) and pd.notna(r.iloc[0]) and pd.notna(s.iloc[0]):
                    gaps.append({
                        "table": table_name, "target": target, "task": task,
                        "metric": mname, "train_on": f"gap(real-{sname})",
                        "score": float(r.iloc[0] - s.iloc[0]), "note": "",
                    })
    if gaps:
        out = pd.concat([out, pd.DataFrame(gaps)], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# 5. MULTI-SYNTHESIZER SUITE (HMA + CTGAN + TVAE + copulas)
# ---------------------------------------------------------------------------

#: Distinct colour per synthesizer, consistent across every comparison figure.
SYNTH_PALETTE = {
    "real": "#333333",
    "HMA": "#1f77b4",
    "GaussianCopula": "#2ca02c",
    "CTGAN": "#d62728",
    "TVAE": "#9467bd",
    "CopulaGAN": "#ff7f0e",
}


def _color_for(name: str, i: int = 0) -> str:
    return SYNTH_PALETTE.get(name, plt.cm.tab10.colors[i % 10])


def build_single_table_synthesizer(name: str, single_meta, epochs: int = 300, random_state: int = 0):
    """Factory for SDV single-table synthesizers by (case-insensitive) name."""
    from sdv.single_table import (
        CopulaGANSynthesizer,
        CTGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    key = name.lower().replace("_", "").replace("-", "")
    if key in {"gaussiancopula", "gc", "copula"}:
        return GaussianCopulaSynthesizer(single_meta)
    if key == "ctgan":
        return CTGANSynthesizer(single_meta, epochs=epochs, verbose=False)
    if key == "tvae":
        return TVAESynthesizer(single_meta, epochs=epochs)
    if key == "copulagan":
        return CopulaGANSynthesizer(single_meta, epochs=epochs, verbose=False)
    raise ValueError(f"Unknown synthesizer '{name}'")


def generate_synthetic_suite(
    train_tables: Dict[str, pd.DataFrame],
    metadata,
    synthesizers: Sequence[str] = ("HMA", "GaussianCopula", "CTGAN", "TVAE"),
    scale: float = 1.0,
    epochs: int = 300,
    verbose: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fit every requested synthesizer and sample synthetic data.

    'HMA' uses the multi-table HMASynthesizer over all tables at once; every
    other name is an SDV *single-table* synthesizer fitted per table (valid
    here because relationships were removed -> tables are independent).

    Returns ``{synthesizer_name: {table_name: synthetic_df}}``.  A synthesizer
    that fails (e.g. torch missing for CTGAN/TVAE) is skipped with a warning
    instead of aborting the whole run.
    """
    suite: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name in synthesizers:
        try:
            if name.upper() == "HMA":
                from sdv.multi_table import HMASynthesizer

                if verbose:
                    print(f"[{name}] fitting multi-table HMASynthesizer ...")
                syn = HMASynthesizer(metadata)
                syn.fit(train_tables)
                suite["HMA"] = syn.sample(scale=scale)
            else:
                tbls: Dict[str, pd.DataFrame] = {}
                for tname, df in train_tables.items():
                    if verbose:
                        print(f"[{name}] fitting {tname} ({len(df)} rows) ...")
                    single_meta = _single_table_metadata(metadata, tname)
                    syn = build_single_table_synthesizer(name, single_meta, epochs=epochs)
                    syn.fit(df)
                    tbls[tname] = syn.sample(num_rows=max(1, int(len(df) * scale)))
                suite[name] = tbls
            if verbose:
                shapes = {t: d.shape for t, d in suite[name].items()}
                print(f"[{name}] done: {shapes}")
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"Synthesizer '{name}' failed and was skipped: {e}")
    return suite


# ---------------------------------------------------------------------------
# 6. COMPARISON VISUALIZATIONS (fidelity / privacy / utility / leaderboard)
# ---------------------------------------------------------------------------

def _annotate_bars(ax, fmt="{:.2f}", fontsize=7):
    for p in ax.patches:
        h = p.get_height()
        if h == h and abs(h) > 1e-12:  # skip NaN / zero
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width() / 2, h),
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=fontsize,
                rotation=0,
            )


def plot_quality_comparison(
    quality_scores: Dict[str, Dict[str, Dict[str, float]]], out_png: str
) -> Optional[str]:
    """Grouped bars: fidelity QualityReport scores per synthesizer per table.

    ``quality_scores`` = {synth: {table: {overall, column_shapes,
    column_pair_trends}}}.  Returns base64 PNG.
    """
    synths = list(quality_scores)
    tables = sorted({t for s in quality_scores.values() for t in s})
    if not synths or not tables:
        return None
    metrics = [("overall", "Overall quality"), ("column_shapes", "Column Shapes"),
               ("column_pair_trends", "Column Pair Trends")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    x = np.arange(len(tables))
    w = 0.8 / max(len(synths), 1)
    for ax, (key, title) in zip(axes, metrics):
        for i, s in enumerate(synths):
            vals = [quality_scores[s].get(t, {}).get(key, np.nan) for t in tables]
            ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=s, color=_color_for(s, i))
        _annotate_bars(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(tables, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.axhline(0.9, color="green", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_ylabel("score (1 = perfect fidelity)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Fidelity: sdmetrics QualityReport by synthesizer", y=1.03)
    fig.tight_layout()
    return _save_fig(fig, out_png)


def plot_column_shapes_heatmap(
    shape_scores: Dict[str, pd.Series], out_png: str, table_name: str = ""
) -> Optional[str]:
    """Per-column KS/TV shape scores as a columns x synthesizer heatmap.

    ``shape_scores`` = {synth: Series(column -> score)} from
    QualityReport.get_details('Column Shapes').
    """
    if not shape_scores:
        return None
    mat = pd.DataFrame(shape_scores)
    if mat.empty:
        return None
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * mat.shape[1], 0.8 + 0.42 * mat.shape[0]))
    if _HAS_SNS:
        sns.heatmap(mat, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", annot=True, fmt=".2f",
                    cbar_kws={"label": "shape score"}, linewidths=0.5)
    else:
        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=30, ha="right")
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        fig.colorbar(im, ax=ax)
    ax.set_title(f"{table_name}: per-column shape score by synthesizer")
    fig.tight_layout()
    return _save_fig(fig, out_png)


def plot_pair_trends_heatmap(
    details: Sequence[dict], out_png: str, table_name: str = "", synth_name: str = ""
) -> Optional[str]:
    """Column Pair Trends as a column x column similarity matrix.

    ``details`` = records from ``QualityReport.get_details('Column Pair Trends')``
    with keys 'Column 1', 'Column 2', 'Score'.  Cell (i, j) is how well the
    relationship between columns i and j is preserved (1 = identical trend).
    Pairs sdmetrics could not score (constant / near-empty columns) are left
    blank.  Columns that have no scored pair at all are dropped so the plot
    focuses on the relationships that actually exist.
    """
    if not details:
        return None
    scores: Dict[tuple, float] = {}
    cols: List[str] = []
    for rec in details:
        c1, c2, sc = rec.get("Column 1"), rec.get("Column 2"), rec.get("Score")
        if c1 is None or c2 is None:
            continue
        for c in (c1, c2):
            if c not in cols:
                cols.append(c)
        if sc is not None and not (isinstance(sc, float) and np.isnan(sc)):
            scores[(c1, c2)] = float(sc)
            scores[(c2, c1)] = float(sc)
    # keep only columns that participate in at least one scored pair
    scored_cols = [c for c in cols if any((c, o) in scores for o in cols)]
    if len(scored_cols) < 2:
        return None
    mat = pd.DataFrame(np.nan, index=scored_cols, columns=scored_cols, dtype=float)
    for (a, b), v in scores.items():
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = v
    np.fill_diagonal(mat.values, 1.0)

    n = len(scored_cols)
    annot = n <= 18
    fig, ax = plt.subplots(figsize=(1.5 + 0.55 * n, 1.2 + 0.5 * n))
    if _HAS_SNS:
        sns.heatmap(mat, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", annot=annot, fmt=".2f",
                    cbar_kws={"label": "pair-trend similarity"}, linewidths=0.5,
                    square=True, mask=mat.isna())
    else:
        im = ax.imshow(np.ma.masked_invalid(mat.values), vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(n)); ax.set_xticklabels(scored_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(scored_cols, fontsize=7)
        fig.colorbar(im, ax=ax, label="pair-trend similarity")
    title = f"{table_name}: column pair trends"
    if synth_name:
        title += f" ({synth_name})"
    ax.set_title(title)
    fig.tight_layout()
    return _save_fig(fig, out_png)


def plot_privacy_comparison(
    privacy_all: Dict[str, Dict[str, Dict]], out_png: str
) -> Optional[str]:
    """Three-panel privacy dashboard across synthesizers.

    Panels: MIA AUC (ideal 0.5, green band = safe), DCR ratio (ideal >= 1),
    exact-match %.  ``privacy_all`` = {synth: {table: privacy_report dict}}.
    """
    synths = list(privacy_all)
    tables = sorted({t for s in privacy_all.values() for t in s})
    if not synths or not tables:
        return None
    x = np.arange(len(tables))
    w = 0.8 / max(len(synths), 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    def grab(s, t, path, default=np.nan):
        d = privacy_all.get(s, {}).get(t, {})
        for p in path:
            d = d.get(p, {}) if isinstance(d, dict) else {}
        return d if isinstance(d, (int, float)) else default

    # -- panel 1: MIA AUC
    ax = axes[0]
    ax.axhspan(0.4, 0.6, color="green", alpha=0.10, label="safe zone")
    ax.axhline(0.5, color="green", lw=1.2, ls="--")
    for i, s in enumerate(synths):
        vals = [grab(s, t, ["membership_inference", "auc"]) for t in tables]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=s, color=_color_for(s, i))
    _annotate_bars(ax)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(tables, rotation=15)
    ax.set_title("Membership Inference AUC (ideal = 0.5)")
    ax.legend(fontsize=7)

    # -- panel 2: DCR ratio
    ax = axes[1]
    ax.axhline(1.0, color="green", lw=1.2, ls="--", label="ideal >= 1")
    for i, s in enumerate(synths):
        vals = [grab(s, t, ["dcr", "dcr_ratio_median"]) for t in tables]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=s, color=_color_for(s, i))
    _annotate_bars(ax)
    ax.set_xticks(x); ax.set_xticklabels(tables, rotation=15)
    ax.set_title("DCR ratio: median d(real→synth) / d(real→real)")
    ax.legend(fontsize=7)

    # -- panel 3: exact match rate (%)
    ax = axes[2]
    for i, s in enumerate(synths):
        vals = [100 * grab(s, t, ["exact_match", "exact_match_rate"], 0.0) for t in tables]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=s, color=_color_for(s, i))
    _annotate_bars(ax, fmt="{:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(tables, rotation=15)
    ax.set_title("Exact-match rate (% synthetic rows copying a real row)")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)

    fig.suptitle("Privacy metrics by synthesizer", y=1.03)
    fig.tight_layout()
    return _save_fig(fig, out_png)


def plot_dcr_overlay(
    dcr_arrays: Dict[str, np.ndarray], baseline: Optional[np.ndarray],
    out_png: str, table_name: str = ""
) -> Optional[str]:
    """Overlay real→synthetic DCR distributions of every synthesizer.

    ``dcr_arrays`` = {synth: distances}; ``baseline`` = real→real distances.
    """
    arrays = {k: np.asarray(v).ravel() for k, v in dcr_arrays.items() if v is not None and len(v)}
    if not arrays:
        return None
    all_vals = np.concatenate(list(arrays.values()) + ([np.asarray(baseline).ravel()] if baseline is not None else []))
    hi = np.percentile(all_vals, 99) + 1e-9
    bins = np.linspace(0, hi, 50)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if baseline is not None and len(baseline):
        ax.hist(np.asarray(baseline).ravel(), bins=bins, density=True, alpha=0.30,
                color=_color_for("real"), label="real→real (baseline)")
    for i, (s, arr) in enumerate(arrays.items()):
        if _HAS_SNS and len(arr) > 1:
            try:
                sns.kdeplot(arr, ax=ax, color=_color_for(s, i), lw=1.8, label=s, clip=(0, hi))
                continue
            except Exception:
                pass
        ax.hist(arr, bins=bins, density=True, histtype="step", lw=1.8,
                color=_color_for(s, i), label=s)
    ax.set_xlabel("distance to closest record")
    ax.set_ylabel("density")
    ax.set_title(f"{table_name}: DCR distributions (further right of baseline = safer)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, out_png)


def plot_efficacy_comparison(efficacy_table: pd.DataFrame, out_png: str) -> Optional[str]:
    """TSTR grouped bars: test-set score per training source, per table.

    Expects the tidy frame from :func:`ml_efficacy_tstr` (multi-source mode):
    columns table / model / train_on / accuracy|r2.  'gap(...)' rows excluded.
    """
    if efficacy_table is None or efficacy_table.empty or "train_on" not in efficacy_table.columns:
        return None
    df = efficacy_table[~efficacy_table["train_on"].astype(str).str.startswith("gap(")].copy()
    if df.empty:
        return None
    tables = sorted(df["table"].dropna().unique())
    models = [m for m in ("RandomForest", "baseline") if m in set(df.get("model", []))]
    if not tables or not models:
        return None
    fig, axes = plt.subplots(len(models), len(tables),
                             figsize=(5.2 * len(tables), 4.2 * len(models)),
                             squeeze=False)
    for r, model in enumerate(models):
        for c, t in enumerate(tables):
            ax = axes[r][c]
            sub = df[(df["table"] == t) & (df["model"] == model)]
            if sub.empty:
                ax.axis("off")
                continue
            task = sub["task"].iloc[0]
            metric = "accuracy" if task == "classification" else "r2"
            if metric not in sub.columns:
                ax.axis("off")
                continue
            sources = list(sub["train_on"])
            vals = list(sub[metric])
            colors = [_color_for(s, i) for i, s in enumerate(sources)]
            bars = ax.bar(range(len(sources)), vals, color=colors)
            # emphasize the train-on-real reference with a dashed line
            ref = sub[sub["train_on"] == "real"][metric]
            if len(ref) and pd.notna(ref.iloc[0]):
                ax.axhline(float(ref.iloc[0]), color="black", ls="--", lw=1,
                           label="train-on-real reference")
                ax.legend(fontsize=7)
            _annotate_bars(ax)
            ax.set_xticks(range(len(sources)))
            ax.set_xticklabels(sources, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{t} · {model} · {metric} (target={sub['target'].iloc[0]})", fontsize=9)
            if metric == "accuracy":
                ax.set_ylim(0, 1.05)
    fig.suptitle("ML efficacy (TSTR): trained on each source, tested on the SAME real holdout", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, out_png)


def plot_efficacy_scores(efficacy_table: pd.DataFrame, out_png: str) -> Optional[str]:
    """Grouped bars for the sdmetrics ML-efficacy frame (one panel per
    table x metric): score per training source, dashed train-on-real line.
    """
    if efficacy_table is None or efficacy_table.empty or "score" not in efficacy_table.columns:
        return None
    df = efficacy_table[~efficacy_table["train_on"].astype(str).str.startswith("gap(")].copy()
    df = df[pd.notna(df["score"])]
    if df.empty:
        return None
    panels = list(df.groupby(["table", "metric"]).groups)
    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (t, mname) in zip(axes.ravel(), panels):
        sub = df[(df["table"] == t) & (df["metric"] == mname)]
        sources = list(sub["train_on"])
        vals = list(sub["score"])
        ax.bar(range(len(sources)), vals,
               color=[_color_for(s, i) for i, s in enumerate(sources)])
        ref = sub[sub["train_on"] == "real"]["score"]
        if len(ref):
            ax.axhline(float(ref.iloc[0]), color="black", ls="--", lw=1,
                       label="train-on-real reference")
            ax.legend(fontsize=7)
        _annotate_bars(ax)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{t}\n{mname} · target={sub['target'].iloc[0]}", fontsize=9)
        if "F1" in mname:
            ax.set_ylim(0, 1.05)
    fig.suptitle("sdmetrics ML efficacy (TSTR): trained per source, tested on the same real holdout",
                 y=1.02)
    fig.tight_layout()
    return _save_fig(fig, out_png)


def compute_leaderboard(
    quality_scores: Dict[str, Dict[str, Dict[str, float]]],
    privacy_all: Dict[str, Dict[str, Dict]],
    efficacy_table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """One row per synthesizer with 0-1 scores: fidelity / privacy / utility.

    * fidelity  = mean QualityReport overall score across tables
    * privacy   = mean of [1-2|AUC-0.5| (MIA), min(1, DCR ratio),
                  NewRowSynthesis, 1-exact_match_rate]
    * utility   = mean over tables of clip(synth_score / real_score, 0, 1)
                  using the RandomForest TSTR metric (accuracy or R²)
    """
    rows = []
    for s in quality_scores:
        row: Dict[str, object] = {"synthesizer": s}
        fid = [v.get("overall") for v in quality_scores[s].values() if v.get("overall") is not None]
        row["fidelity"] = float(np.mean(fid)) if fid else np.nan

        parts: List[float] = []
        for rep in (privacy_all.get(s) or {}).values():
            auc = rep.get("membership_inference", {}).get("auc")
            if auc is not None and not (isinstance(auc, float) and np.isnan(auc)):
                parts.append(max(0.0, 1.0 - 2.0 * abs(float(auc) - 0.5)))
            ratio = rep.get("dcr", {}).get("dcr_ratio_median")
            if ratio is not None:
                parts.append(float(min(1.0, ratio)))
            nrs = rep.get("sdmetrics", {}).get("NewRowSynthesis")
            if nrs is not None:
                parts.append(float(nrs))
            emr = rep.get("exact_match", {}).get("exact_match_rate")
            if emr is not None:
                parts.append(float(max(0.0, 1.0 - emr)))
        row["privacy"] = float(np.mean(parts)) if parts else np.nan

        utils: List[float] = []
        if efficacy_table is not None and not efficacy_table.empty and "train_on" in efficacy_table.columns:
            if "score" in efficacy_table.columns and "metric" in efficacy_table.columns:
                # sdmetrics-efficacy frame: one score column, panels = table x metric
                base = efficacy_table[~efficacy_table["train_on"].astype(str).str.startswith("gap(")]
                for (_, _), sub in base.groupby(["table", "metric"]):
                    r = sub[sub["train_on"] == "real"]["score"]
                    sv = sub[sub["train_on"] == s]["score"]
                    if len(r) and len(sv) and pd.notna(r.iloc[0]) and pd.notna(sv.iloc[0]) and float(r.iloc[0]) > 0:
                        utils.append(float(np.clip(float(sv.iloc[0]) / float(r.iloc[0]), 0.0, 1.0)))
            else:
                # legacy custom-TSTR frame (accuracy / r2 columns)
                for t in efficacy_table["table"].dropna().unique():
                    sub = efficacy_table[(efficacy_table["table"] == t)
                                         & (efficacy_table["model"] == "RandomForest")]
                    if sub.empty:
                        continue
                    metric = "accuracy" if sub["task"].iloc[0] == "classification" else "r2"
                    if metric not in sub.columns:
                        continue
                    r = sub[sub["train_on"] == "real"][metric]
                    sv = sub[sub["train_on"] == s][metric]
                    if len(r) and len(sv) and pd.notna(r.iloc[0]) and pd.notna(sv.iloc[0]) and float(r.iloc[0]) > 0:
                        utils.append(float(np.clip(float(sv.iloc[0]) / float(r.iloc[0]), 0.0, 1.0)))
        row["utility_tstr"] = float(np.mean(utils)) if utils else np.nan

        dims = [row["fidelity"], row["privacy"], row["utility_tstr"]]
        row["overall"] = float(np.nanmean([d for d in dims]))
        rows.append(row)
    lb = pd.DataFrame(rows).set_index("synthesizer")
    return lb.sort_values("overall", ascending=False)


def plot_leaderboard(leaderboard: pd.DataFrame, out_png: str) -> Optional[str]:
    """Annotated heatmap of the fidelity / privacy / utility leaderboard."""
    if leaderboard is None or leaderboard.empty:
        return None
    fig, ax = plt.subplots(figsize=(2.2 + 1.5 * leaderboard.shape[1], 1.2 + 0.7 * leaderboard.shape[0]))
    if _HAS_SNS:
        sns.heatmap(leaderboard, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", annot=True,
                    fmt=".3f", linewidths=0.6, cbar_kws={"label": "score (1 = best)"})
    else:
        im = ax.imshow(leaderboard.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(leaderboard.shape[1]))
        ax.set_xticklabels(leaderboard.columns)
        ax.set_yticks(range(leaderboard.shape[0]))
        ax.set_yticklabels(leaderboard.index)
        for i in range(leaderboard.shape[0]):
            for j in range(leaderboard.shape[1]):
                ax.text(j, i, f"{leaderboard.iloc[i, j]:.3f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)
    ax.set_title("Synthesizer leaderboard (fidelity · privacy · utility)")
    fig.tight_layout()
    return _save_fig(fig, out_png)


def plot_column_distribution_multi(
    real: pd.DataFrame,
    synths: Dict[str, pd.DataFrame],
    column: str,
    is_numeric: bool,
    out_png: str,
    table_name: str = "",
) -> str:
    """Real vs EVERY synthesizer for one column in a single figure.

    Numeric  -> real as filled histogram + one KDE/step line per synthesizer
    Category -> grouped proportion bars (real first, then each synthesizer)
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    real_c = real[column].dropna()
    if is_numeric:
        real_v = pd.to_numeric(real_c, errors="coerce").dropna()
        bins = min(40, max(10, int(np.sqrt(max(len(real_v), 1)))))
        ax.hist(real_v, bins=bins, alpha=0.35, density=True, label="real",
                color=_color_for("real"))
        for i, (s, sdf) in enumerate(synths.items()):
            if column not in sdf.columns:
                continue
            sv = pd.to_numeric(sdf[column], errors="coerce").dropna()
            if len(sv) < 2:
                continue
            drew = False
            if _HAS_SNS:
                try:
                    sns.kdeplot(sv, ax=ax, color=_color_for(s, i), lw=1.8, label=s)
                    drew = True
                except Exception:
                    pass
            if not drew:
                ax.hist(sv, bins=bins, density=True, histtype="step", lw=1.8,
                        color=_color_for(s, i), label=s)
        ax.set_ylabel("density")
    else:
        rp = real_c.astype(str).value_counts(normalize=True)
        cats = list(rp.index[:15])
        series = {"real": rp.reindex(cats, fill_value=0.0)}
        for s, sdf in synths.items():
            if column in sdf.columns:
                sp = sdf[column].dropna().astype(str).value_counts(normalize=True)
                series[s] = sp.reindex(cats, fill_value=0.0)
        x = np.arange(len(cats))
        w = 0.8 / max(len(series), 1)
        for i, (s, vals) in enumerate(series.items()):
            ax.bar(x + i * w - 0.4 + w / 2, vals.values, width=w, label=s,
                   color=_color_for(s, i))
        ax.set_xticks(x)
        ax.set_xticklabels([c[:14] for c in cats], rotation=45, ha="right")
        ax.set_ylabel("proportion")
    ax.set_title(f"{table_name}.{column} — real vs all synthesizers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, out_png)


def visualize_table_multi(
    real: pd.DataFrame,
    synths: Dict[str, pd.DataFrame],
    roles: ColumnRoles,
    table_name: str,
    reports_dir: str,
    max_cols: int = 12,
) -> List[Tuple[str, str]]:
    """Multi-synthesizer overlay plots for a table's top columns.

    Returns ``[(title, base64_png), ...]`` ready for the HTML report.
    """
    fig_dir = os.path.join(reports_dir, "figures", table_name)
    os.makedirs(fig_dir, exist_ok=True)
    out: List[Tuple[str, str]] = []
    for col in (roles.numeric + roles.categorical)[:max_cols]:
        png = os.path.join(fig_dir, f"multi_dist_{col}.png")
        b64 = plot_column_distribution_multi(
            real, synths, col, col in roles.numeric, png, table_name=table_name
        )
        out.append((f"{table_name}.{col} (all synthesizers)", b64))
    return out
