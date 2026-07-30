"""synth_eval.suite — multi-synthesizer generation (HMA + single-table models)."""
from __future__ import annotations

import time
import warnings
from typing import Dict, Optional, Sequence

import pandas as pd

from ._common import SYNTH_PALETTE, _color_for, _single_table_metadata
from .privacy import filter_close_records, filter_close_records_multitable


def build_single_table_synthesizer(name: str, single_meta, epochs: int = 300):
    """Factory for SDV single-table synthesizers by (case-insensitive) name.

    No synthesizer constructor here accepts a seed directly (none of SDV's
    single-table classes take a ``random_state``/``seed`` kwarg) -- see
    :func:`_seed_synth` for how reproducibility is actually applied, after
    construction.
    """
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


def _seed_global(seed: int):
    """Seed the global RNGs that stochastic FITTING relies on (CTGAN/TVAE/
    CopulaGAN train a real neural net -- weight init, minibatch order,
    dropout -- all of which read numpy/torch's global state, not a
    per-instance one). Cheap no-op for GaussianCopula/HMA, which fit via
    closed-form estimation with no stochastic training step."""
    import numpy as np

    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _seed_synth(syn, seed: int):
    """Point a FITTED synthesizer's own model-local RNG at ``seed`` before a
    sample() call. SDV has no PUBLIC API for this (confirmed: none of the
    single-table constructors take a seed, and .fit() unconditionally resets
    any prior seeding); the closest thing is the private-but-stable
    ``_set_random_state`` (single-table) / ``_numpy_seed`` (HMA) hooks --
    the exact same ones SDV itself uses internally to auto-seed the first
    post-fit sample to its own hardcoded constant. Calling them directly
    just means OUR seed drives that instead of SDV's fixed default. Best
    effort: silently no-ops if a future SDV version renames/removes them,
    rather than aborting generation over a reproducibility nicety.
    """
    try:
        if hasattr(syn, "_set_random_state"):
            syn._set_random_state(seed)
        elif hasattr(syn, "_numpy_seed"):
            syn._numpy_seed = seed
    except Exception:
        pass


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
    should_cancel=None,
    timings: Optional[Dict[str, float]] = None,
    roles=None,
    filter_close_percentile: float = 5.0,
    close_filter_report: Optional[Dict[str, Dict[str, dict]]] = None,
    resample_timings: Optional[Dict[str, float]] = None,
    random_state: int = 0,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fit every requested synthesizer and sample synthetic data.

    'HMA' uses the multi-table HMASynthesizer over all tables at once; every
    other name is an SDV *single-table* synthesizer fitted per table (valid
    here because relationships were removed -> tables are independent).

    ``random_state`` seeds both the stochastic-fit path (CTGAN/TVAE/
    CopulaGAN training, via :func:`_seed_global`) and each synthesizer's own
    sample() calls (via :func:`_seed_synth`) -- the initial batch gets
    ``random_state`` exactly, and each reject-and-resample retry (see
    ``roles`` below) gets ``random_state`` offset by its attempt number, so
    reruns with the same config are reproducible without every retry just
    repeating the same rejected rows.

    ``constraints`` is an optional list of UI constraint specs (see
    :func:`build_constraints`).  They are added to each synthesizer before
    fitting so the synthetic data satisfies them by construction.  If adding a
    constraint fails, that synthesizer is fitted without it (with a warning)
    rather than aborting the run.

    ``timings``, if given, is filled in-place with wall-clock fit+sample
    seconds per synthesizer name (only for ones that actually succeeded,
    and NOT counting the reject-and-resample filter below, see
    ``resample_timings`` for that) -- the raw "how long did this take"
    number behind the time-savings story, measured the same way regardless
    of synthesizer so it's directly comparable across
    HMA/GaussianCopula/CTGAN/TVAE/CopulaGAN. ``resample_timings``, if given,
    is filled in-place with wall-clock seconds spent specifically in the
    close-record filter (encoding + nearest-neighbor search + any resample
    rounds) per synthesizer, split out from ``timings`` so "how long did
    generation take" and "how long did the privacy filter add on top" are
    each their own number.

    ``roles``, if given as ``{table_name: ColumnRoles}``, turns on the
    reject-and-resample privacy filter: rows that land closer to a real
    (training) row than real rows ever sit to each other get dropped and
    refilled from fresh draws of the same fitted model
    (:func:`synth_eval.privacy.filter_close_records` per table for the
    single-table synthesizers). For HMA, rows are tied together across
    tables by shared keys, so this instead checks only the root/"entity"
    table and cascades any rejection down to every descendant row that
    references it, refilling from whole fresh linked groups pulled out of a
    new full sample (:func:`synth_eval.privacy.filter_close_records_multitable`)
    -- a row being close in a leaf/child table alone, without its parent also
    being close, is not covered. ``close_filter_report``, if given, is filled
    in-place with ``{synth_name: {table_name: report}}`` from each table's
    filter run (HMA's entry is keyed by its root table(s)).

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
        if should_cancel is not None and should_cancel():
            break                                    # user cancelled — stop before the next model
        t0 = time.perf_counter()
        t_filter = 0.0   # time spent in the reject-and-resample filter, split out of fit+sample below
        try:
            if name.upper() == "HMA":
                from sdv.multi_table import HMASynthesizer

                if verbose:
                    print(f"[{name}] fitting multi-table HMASynthesizer ...")
                # verbose=True so SDV emits phase/progress on stderr; the server
                # captures that stream and forwards it to the dashboard console.
                _seed_global(random_state)
                try:
                    syn = HMASynthesizer(metadata, verbose=True)
                except TypeError:  # older sdv without the verbose kwarg
                    syn = HMASynthesizer(metadata)
                _apply(syn, build_constraints(constraints, multitable=True), "HMA")
                syn.fit(train_tables)
                _seed_synth(syn, random_state)
                sampled = syn.sample(scale=scale)
                if roles is not None:
                    tf0 = time.perf_counter()
                    rels = metadata.to_dict().get("relationships", [])
                    _retry = [0]
                    def _hma_resample(_syn=syn, _retry=_retry):
                        _retry[0] += 1
                        _seed_synth(_syn, random_state + _retry[0])
                        return _syn.sample(scale=scale)
                    result = filter_close_records_multitable(
                        train_tables, sampled, roles, rels,
                        resample_fn=_hma_resample,
                        percentile=filter_close_percentile)
                    t_filter += time.perf_counter() - tf0
                    sampled = result["data"]
                    if close_filter_report is not None:
                        close_filter_report.setdefault(name, {}).update(result["report"]["tables"])
                suite["HMA"] = sampled
            else:
                tbls: Dict[str, pd.DataFrame] = {}
                for tname, df in train_tables.items():
                    if verbose:
                        print(f"[{name}] fitting {tname} ({len(df)} rows) ...")
                    _seed_global(random_state)
                    single_meta = _single_table_metadata(metadata, tname)
                    syn = build_single_table_synthesizer(name, single_meta, epochs=epochs)
                    _apply(syn, build_constraints(constraints, only_table=tname), f"{name}/{tname}")
                    syn.fit(df)
                    _seed_synth(syn, random_state)
                    n_rows = max(1, int(len(df) * scale))
                    sampled = syn.sample(num_rows=n_rows)
                    if roles is not None and tname in roles:
                        tf0 = time.perf_counter()
                        _retry = [0]
                        def _single_resample(n, _syn=syn, _retry=_retry):
                            _retry[0] += 1
                            _seed_synth(_syn, random_state + _retry[0])
                            return _syn.sample(num_rows=n)
                        result = filter_close_records(
                            df, sampled, roles[tname],
                            resample_fn=_single_resample,
                            percentile=filter_close_percentile)
                        t_filter += time.perf_counter() - tf0
                        sampled = result["data"]
                        if close_filter_report is not None:
                            close_filter_report.setdefault(name, {})[tname] = result["report"]
                    tbls[tname] = sampled
                suite[name] = tbls
            if timings is not None:
                timings[name] = (time.perf_counter() - t0) - t_filter
            if resample_timings is not None:
                resample_timings[name] = t_filter
            if verbose:
                shapes = {t: d.shape for t, d in suite[name].items()}
                print(f"[{name}] done: {shapes}")
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"Synthesizer '{name}' failed and was skipped: {e}")
    return suite


