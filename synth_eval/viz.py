"""synth_eval.viz — per-table distribution/correlation plots and HTML report."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._common import plt, sns, _HAS_SNS, _fig_to_base64, _save_fig, _single_table_metadata
from .columns import ColumnRoles, classify_columns


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

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


