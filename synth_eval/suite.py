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


def build_constraints(specs, only_table=None, multitable=False):
    """Turn UI constraint specs into SDV ``sdv.cag`` constraint objects.

    ``specs`` is a list of dicts, e.g.::

        {"table": "PERSON", "type": "inequality",
         "low": "EFFECTIVE_DATE", "high": "END_DATE", "strict": False}
        {"table": "PERSON", "type": "range",
         "low": "START", "middle": "MID", "high": "END"}
        {"table": "CONTACT", "type": "fixed_combinations",
         "columns": ["STATUS_CD", "STATUS_DESC"]}
        {"table": "T", "type": "fixed_increments",
         "column": "AMOUNT", "increment": 5}

    ``only_table`` filters to specs for that one table (single-table path).
    ``multitable`` tags each constraint with its own table (HMA path); when
    False the constraint's ``table_name`` is left None (single-table path).
    Unknown / malformed specs are skipped with a warning.
    """
    from sdv import cag

    built = []
    for sp in specs or []:
        try:
            t = sp.get("table")
            if only_table is not None and t != only_table:
                continue
            tn = t if multitable else None
            kind = str(sp.get("type", "")).lower().replace("-", "_")
            if kind == "inequality":
                built.append(cag.Inequality(
                    low_column_name=sp["low"], high_column_name=sp["high"],
                    strict_boundaries=bool(sp.get("strict", False)), table_name=tn))
            elif kind == "range":
                built.append(cag.Range(
                    low_column_name=sp["low"], middle_column_name=sp["middle"],
                    high_column_name=sp["high"],
                    strict_boundaries=bool(sp.get("strict", True)), table_name=tn))
            elif kind in ("fixed_combinations", "fixedcombinations"):
                cols = [c for c in sp.get("columns", []) if c]
                if len(cols) >= 2:
                    built.append(cag.FixedCombinations(column_names=cols, table_name=tn))
            elif kind in ("fixed_increments", "fixedincrements"):
                built.append(cag.FixedIncrements(
                    column_name=sp["column"],
                    increment_value=int(sp["increment"]), table_name=tn))
            else:
                warnings.warn(f"unknown constraint type '{kind}' skipped")
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"constraint {sp!r} could not be built and was skipped: {e}")
    return built


def generate_synthetic_suite(
    train_tables: Dict[str, pd.DataFrame],
    metadata,
    synthesizers: Sequence[str] = ("HMA", "GaussianCopula", "CTGAN", "TVAE"),
    scale: float = 1.0,
    epochs: int = 300,
    verbose: bool = True,
    constraints=None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fit every requested synthesizer and sample synthetic data.

    'HMA' uses the multi-table HMASynthesizer over all tables at once; every
    other name is an SDV *single-table* synthesizer fitted per table (valid
    here because relationships were removed -> tables are independent).

    ``constraints`` is an optional list of UI constraint specs (see
    :func:`build_constraints`).  They are added to each synthesizer before
    fitting so the synthetic data satisfies them by construction.  If adding a
    constraint fails, that synthesizer is fitted without it (with a warning)
    rather than aborting the run.

    Returns ``{synthesizer_name: {table_name: synthetic_df}}``.  A synthesizer
    that fails (e.g. torch missing for CTGAN/TVAE) is skipped with a warning
    instead of aborting the whole run.
    """
    def _apply(syn, cons, label):
        if cons:
            try:
                syn.add_constraints(cons)
            except Exception as e:
                warnings.warn(f"constraints not applied ({label}): {e}")

    suite: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name in synthesizers:
        try:
            if name.upper() == "HMA":
                from sdv.multi_table import HMASynthesizer

                if verbose:
                    print(f"[{name}] fitting multi-table HMASynthesizer ...")
                # verbose=True so SDV emits phase/progress on stderr; the server
                # captures that stream and forwards it to the dashboard console.
                try:
                    syn = HMASynthesizer(metadata, verbose=True)
                except TypeError:  # older sdv without the verbose kwarg
                    syn = HMASynthesizer(metadata)
                _apply(syn, build_constraints(constraints, multitable=True), "HMA")
                syn.fit(train_tables)
                suite["HMA"] = syn.sample(scale=scale)
            else:
                tbls: Dict[str, pd.DataFrame] = {}
                for tname, df in train_tables.items():
                    if verbose:
                        print(f"[{name}] fitting {tname} ({len(df)} rows) ...")
                    single_meta = _single_table_metadata(metadata, tname)
                    syn = build_single_table_synthesizer(name, single_meta, epochs=epochs)
                    _apply(syn, build_constraints(constraints, only_table=tname), f"{name}/{tname}")
                    syn.fit(df)
                    tbls[tname] = syn.sample(num_rows=max(1, int(len(df) * scale)))
                suite[name] = tbls
            if verbose:
                shapes = {t: d.shape for t, d in suite[name].items()}
                print(f"[{name}] done: {shapes}")
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"Synthesizer '{name}' failed and was skipped: {e}")
    return suite


