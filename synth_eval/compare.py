"""synth_eval.compare — cross-synthesizer comparison plots and leaderboard."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import plt, sns, _HAS_SNS, _save_fig, _color_for, SYNTH_PALETTE
from .columns import ColumnRoles, classify_columns
from .viz import plot_column_distribution, visualize_table


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


def quality_comparison_data(quality_scores) -> Optional[dict]:
    """Interactive-chart data for the fidelity QualityReport grouped bars.

    Returns {tables, synths, metrics:[[key,label]], values:{key:{synth:[per table]}}}
    or None.  Consumed by the dashboard's Plotly renderer; the PNG stays a fallback.
    """
    synths = list(quality_scores)
    tables = sorted({t for s in quality_scores.values() for t in s})
    if not synths or not tables:
        return None
    metrics = [("overall", "Overall quality"),
               ("column_shapes", "Column Shapes"),
               ("column_pair_trends", "Column Pair Trends")]

    def _v(x):
        return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    values = {}
    for key, _ in metrics:
        values[key] = {s: [_v(quality_scores[s].get(t, {}).get(key)) for t in tables]
                       for s in synths}
    return {"tables": tables, "synths": synths,
            "metrics": [[k, l] for k, l in metrics], "values": values}


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
    # rows = data columns, cols = synthesizers.  Transpose to a WIDE layout
    # (synthesizers as rows, data columns along the x-axis) so a table with many
    # columns renders as a wide, horizontally-scrollable band instead of a very
    # tall, thin strip that collapses when the browser scales it down.
    mat = pd.DataFrame(shape_scores).T
    if mat.empty:
        return None
    n_syn, n_col = mat.shape                       # rows, columns
    annot = n_col <= 20                            # numbers only when they fit
    width = max(6.0, 0.42 * n_col + 1.5)
    height = 1.4 + 0.6 * n_syn
    fig, ax = plt.subplots(figsize=(width, height))
    if _HAS_SNS:
        sns.heatmap(mat, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", annot=annot, fmt=".2f",
                    cbar_kws={"label": "shape score"}, linewidths=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    else:
        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_col)); ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(n_syn)); ax.set_yticklabels(mat.index)
        fig.colorbar(im, ax=ax, label="shape score")
    ax.set_title(f"{table_name}: per-column shape score by synthesizer")
    fig.tight_layout()
    return _save_fig(fig, out_png)


def _pair_trends_matrix(details):
    """Build the column x column pair-trend matrix from sdmetrics detail records.

    Returns (scored_columns, DataFrame) or None if fewer than 2 scored columns.
    """
    if not details:
        return None
    scores = {}
    cols = []
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
    scored_cols = [c for c in cols if any((c, o) in scores for o in cols)]
    if len(scored_cols) < 2:
        return None
    mat = pd.DataFrame(np.nan, index=scored_cols, columns=scored_cols, dtype=float)
    for (a, b), v in scores.items():
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = v
    np.fill_diagonal(mat.values, 1.0)
    return scored_cols, mat


def _matrix_to_z(mat):
    """DataFrame -> nested list with NaN replaced by None (JSON-safe for Plotly)."""
    return [[None if pd.isna(v) else float(v) for v in row] for row in mat.values]


def shapes_heatmap_data(shape_scores) -> Optional[dict]:
    """Interactive-chart data for the per-column shape-score heatmap.

    Returns {"x": data columns, "y": synthesizers, "z": [[score]]} — the same
    wide orientation as the PNG — or None.  Consumed by the dashboard's Plotly
    renderer; the PNG remains an offline fallback.
    """
    if not shape_scores:
        return None
    mat = pd.DataFrame(shape_scores).T          # rows = synths, cols = data columns
    if mat.empty:
        return None
    return {"x": [str(c) for c in mat.columns], "y": [str(i) for i in mat.index],
            "z": _matrix_to_z(mat)}


def pair_trends_heatmap_data(details) -> Optional[dict]:
    """Interactive-chart data for the pair-trends heatmap.

    Returns {"labels": scored columns, "z": [[similarity]]} (unscored pairs are
    None) or None.
    """
    built = _pair_trends_matrix(details)
    if built is None:
        return None
    scored_cols, mat = built
    return {"labels": [str(c) for c in scored_cols], "z": _matrix_to_z(mat)}


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
    built = _pair_trends_matrix(details)
    if built is None:
        return None
    scored_cols, mat = built
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
    """Three-panel privacy dashboard across synthesizers:
    NewRowSynthesis (sdmetrics, ideal 1), Membership-Inference attacker AUC
    (custom, ideal 0.5 with a safe band), CategoricalCAP (sdmetrics, ideal 1).

    ``privacy_all`` = {synth: {table: privacy_report dict}}.
    """
    synths = list(privacy_all)
    tables = sorted({t for s in privacy_all.values() for t in s})
    if not synths or not tables:
        return None
    x = np.arange(len(tables))
    w = 0.8 / max(len(synths), 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    def grab_sdm(s, t, key):
        v = privacy_all.get(s, {}).get(t, {}).get("sdmetrics", {}).get(key)
        return float(v) if isinstance(v, (int, float)) else np.nan

    def grab_auc(s, t):
        v = privacy_all.get(s, {}).get(t, {}).get("membership_inference", {}).get("auc")
        return float(v) if isinstance(v, (int, float)) else np.nan

    # panel 1: NewRowSynthesis
    ax = axes[0]
    ax.axhline(0.9, color="green", lw=1.0, ls="--", alpha=0.6, label="pass ≥ 0.9")
    for i, s in enumerate(synths):
        ax.bar(x + i * w - 0.4 + w / 2, [grab_sdm(s, t, "NewRowSynthesis") for t in tables],
               width=w, label=s, color=_color_for(s, i))
    _annotate_bars(ax); ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(tables, rotation=15)
    ax.set_title("NewRowSynthesis (novel rows, ideal = 1)", fontsize=10); ax.legend(fontsize=7)

    # panel 2: Membership Inference attacker AUC (ideal 0.5)
    ax = axes[1]
    ax.axhspan(0.4, 0.6, color="green", alpha=0.10, label="safe zone")
    ax.axhline(0.5, color="green", lw=1.2, ls="--")
    for i, s in enumerate(synths):
        ax.bar(x + i * w - 0.4 + w / 2, [grab_auc(s, t) for t in tables],
               width=w, label=s, color=_color_for(s, i))
    _annotate_bars(ax); ax.set_ylim(0, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(tables, rotation=15)
    ax.set_title("Membership Inference AUC (ideal = 0.5)", fontsize=10); ax.legend(fontsize=7)

    # panel 3: CategoricalCAP
    ax = axes[2]
    ax.axhline(0.9, color="green", lw=1.0, ls="--", alpha=0.6, label="pass ≥ 0.9")
    for i, s in enumerate(synths):
        ax.bar(x + i * w - 0.4 + w / 2, [grab_sdm(s, t, "CategoricalCAP") for t in tables],
               width=w, label=s, color=_color_for(s, i))
    _annotate_bars(ax); ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(tables, rotation=15)
    ax.set_title("CategoricalCAP (attribute privacy, ideal = 1)", fontsize=10); ax.legend(fontsize=7)

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
    * privacy   = mean of three 0-1 protection scores
                  (1-2|MIA AUC-0.5|, NewRowSynthesis, CategoricalCAP)
    * utility   = mean over tables of clip(synth_score / real_score, 0, 1)
                  using the RandomForest TSTR metric (accuracy or R²)
    """
    rows = []
    for s in quality_scores:
        row: Dict[str, object] = {"synthesizer": s}
        fid = [v.get("overall") for v in quality_scores[s].values() if v.get("overall") is not None]
        row["fidelity"] = float(np.mean(fid)) if fid else np.nan

        # privacy = mean of three 0-1 protection scores (higher = safer):
        #   MIA -> 1-2|AUC-0.5|, NewRowSynthesis, CategoricalCAP
        parts: List[float] = []
        for rep in (privacy_all.get(s) or {}).values():
            auc = rep.get("membership_inference", {}).get("auc")
            if auc is not None and not (isinstance(auc, float) and np.isnan(auc)):
                parts.append(max(0.0, 1.0 - 2.0 * abs(float(auc) - 0.5)))
            sdm = rep.get("sdmetrics", {})
            for key in ("NewRowSynthesis", "CategoricalCAP"):
                v = sdm.get(key)
                if v is not None:
                    parts.append(float(min(1.0, max(0.0, v))))
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
