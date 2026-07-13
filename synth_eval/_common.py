"""synth_eval._common — shared low-level helpers (figures, metadata, palette)."""
from __future__ import annotations

import base64
import io
import os
from typing import Dict, Optional

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



