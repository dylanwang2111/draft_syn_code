"""synth_eval.suite — multi-synthesizer generation (HMA + single-table models)."""
from __future__ import annotations

import warnings
from typing import Dict, Sequence

import pandas as pd

from ._common import SYNTH_PALETTE, _color_for, _single_table_metadata


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


