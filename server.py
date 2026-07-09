"""FastAPI backend for the Synthetic Data Studio dashboard (web/index.html).

Endpoints (single local user; state kept in memory):
    GET  /                      -> the dashboard SPA
    POST /api/upload            -> multipart CSVs; returns preview + detected metadata
    POST /api/sample            -> load bundled sdg/seed/*.csv instead of uploading
    POST /api/synthesize        -> start synthesis + evaluation in a worker thread
    GET  /api/progress          -> live progress log for the running job
    GET  /api/results           -> full evaluation report (JSON + base64 figures)
    GET  /api/download/{s}/{t}  -> synthetic CSV for synthesizer s, table t

Run with:
    uvicorn server:app --port 8000        (then open http://localhost:8000)
"""

from __future__ import annotations

import glob
import io
import os
import re
import sys
import threading
import traceback
import warnings


class _TqdmTee:
    """Wrap a stderr stream: intercept SDV/tqdm phase-progress lines and forward
    a compact summary to a callback (the job console), pass everything else
    through to the real stderr so normal logs still appear."""

    _PAT = re.compile(r"(Preprocess Tables|Learning relationships|Modeling Tables|"
                      r"Modeling|Sampling|Creating report)\D*?(\d+)\s*%")

    def __init__(self, real, progress_fn):
        self.real = real
        self.progress = progress_fn
        self.buf = ""

    def write(self, s):
        try:
            self.buf += s
        except Exception:
            return
        while True:
            i = -1
            for ch in ("\r", "\n"):
                j = self.buf.find(ch)
                if j >= 0 and (i < 0 or j < i):
                    i = j
            if i < 0:
                break
            seg, self.buf = self.buf[:i], self.buf[i + 1:]
            m = self._PAT.search(seg)
            if m:
                try:
                    self.progress(f"{m.group(1)} … {m.group(2)}%")
                except Exception:
                    pass
            elif seg.strip():
                try:
                    self.real.write(seg + "\n")
                except Exception:
                    pass

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Light blue-tinted figure style so matplotlib PNGs blend into the dashboard.
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "axes.facecolor": "#f4f8fc",
    "axes.edgecolor": "#d5e2f0",
    "axes.labelcolor": "#122c42",
    "axes.titlecolor": "#122c42",
    "text.color": "#122c42",
    "xtick.color": "#5b7288",
    "ytick.color": "#5b7288",
    "grid.color": "#e7eef7",
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#d5e2f0",
})

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_eval as se
import profiler

app = FastAPI(title="Synthetic Data Studio")

REPORTS_DIR = "reports"
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

#: Single in-memory session: {tables, meta_detected, job, results}
STATE: dict = {"tables": None, "meta_detected": None, "job": None, "results": None}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean(o):
    """Make numpy/pandas values JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return None  # raw arrays are never sent to the client
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _detect(tables: dict[str, pd.DataFrame]) -> dict:
    from sdv.metadata import Metadata

    return Metadata.detect_from_dataframes(tables).to_dict()


def _tables_payload():
    tables, meta = STATE["tables"], STATE["meta_detected"]
    out = {"tables": {}, "relationships": meta.get("relationships") or []}
    for t, df in tables.items():
        tmeta = meta["tables"][t]
        roles = se.classify_columns(df, None, t)
        cols = []
        for c in df.columns:
            ex = df[c].dropna()
            cols.append({
                "name": c,
                "sdtype": tmeta["columns"].get(c, {}).get("sdtype", "unknown"),
                "distinct": int(df[c].nunique(dropna=True)),
                "missing_pct": round(100 * float(df[c].isna().mean()), 1),
                "example": str(ex.iloc[0])[:40] if len(ex) else "",
            })
        out["tables"][t] = {
            "rows": len(df),
            "columns": cols,
            "primary_key": tmeta.get("primary_key"),
            "preview": df.head(5).astype(str).to_dict(orient="split"),
            "targets": roles.modelable,
        }
    # structural profile + recommended synthesis tier (safe: structure only)
    try:
        out["profile"] = _clean(profiler.profile_dataset(tables))
    except Exception as e:  # pragma: no cover - never block upload on profiling
        out["profile"] = {"error": str(e)[:200]}
    return out


def _build_metadata(tables_meta: dict, relationships: list[dict]):
    from sdv.metadata import Metadata

    return Metadata.load_from_dict({"tables": tables_meta, "relationships": relationships})


def _metadata_from_request(schema: dict, rels: list[dict]) -> dict:
    """Merge client sdtype/pk edits onto the detected metadata dict."""
    detected = STATE["meta_detected"]
    tables_meta = {}
    for t, tmeta in detected["tables"].items():
        edits = (schema.get(t) or {}).get("sdtypes", {})
        pk = (schema.get(t) or {}).get("primary_key")
        cols = {}
        for col, props in tmeta["columns"].items():
            new = edits.get(col, props.get("sdtype"))
            cols[col] = dict(props) if new == props.get("sdtype") else {"sdtype": new}
        entry = {"columns": cols}
        if pk:
            entry["primary_key"] = pk
            cols[pk] = {"sdtype": "id"}
        tables_meta[t] = entry
    return tables_meta


def _relational_split(tables, rels, holdout_frac, seed):
    """Split parents randomly; child rows follow their parent's split."""
    from sklearn.model_selection import train_test_split

    child_of = {r["child_table_name"]: r for r in rels}
    train, hold = {}, {}
    for t, df in tables.items():
        if t not in child_of:
            tr, ho = train_test_split(df, test_size=holdout_frac, random_state=seed)
            train[t], hold[t] = tr.reset_index(drop=True), ho.reset_index(drop=True)
    for _ in range(len(tables)):
        for t, df in tables.items():
            if t in train or t not in child_of:
                continue
            r = child_of[t]
            pt, pk, fk = r["parent_table_name"], r["parent_primary_key"], r["child_foreign_key"]
            if pt not in train:
                continue
            in_train = df[fk].isin(set(train[pt][pk]))
            in_hold = df[fk].isin(set(hold[pt][pk]))
            rand = np.random.default_rng(seed).random(len(df)) >= holdout_frac
            mask = np.where(in_train, True, np.where(in_hold, False, rand))
            train[t] = df[mask].reset_index(drop=True)
            hold[t] = df[~mask].reset_index(drop=True)
    for t, df in tables.items():
        if t not in train:
            tr, ho = train_test_split(df, test_size=holdout_frac, random_state=seed)
            train[t], hold[t] = tr.reset_index(drop=True), ho.reset_index(drop=True)
    return train, hold


def _relationships_hold(rels, tables) -> bool:
    """True iff every declared relationship is structurally valid in the data:
    parent table/key + child table/key exist, parent key is unique, and every
    non-null child FK value appears among the parent keys."""
    for r in rels:
        pt, pk = r["parent_table_name"], r["parent_primary_key"]
        ct, fk = r["child_table_name"], r["child_foreign_key"]
        if pt not in tables or ct not in tables:
            return False
        if pk not in tables[pt].columns or fk not in tables[ct].columns:
            return False
        parent_keys = tables[pt][pk]
        if parent_keys.duplicated().any() or parent_keys.isna().any():
            return False
        child = tables[ct][fk].dropna()
        if len(child) and not child.isin(set(parent_keys)).all():
            return False
    return True


def _child_pk(df):
    """Pick a unique per-row id column to serve as a table's primary key in the
    entity-hub schema (prefers a warehouse/surrogate id), else None."""
    cols = list(df.columns)
    for c in cols:
        if "warehouse" in c.lower() and df[c].is_unique:
            return c
    for c in cols:
        if c.lower().endswith("_id") and df[c].is_unique:
            return c
    for c in cols:
        if df[c].is_unique:
            return c
    return None


def _reduce_meta(tables_meta: dict, keep: dict) -> dict:
    """A copy of ``tables_meta`` restricted to the kept columns per table."""
    out = {}
    for t, tm in tables_meta.items():
        cols = {c: tm["columns"][c] for c in keep.get(t, []) if c in tm.get("columns", {})}
        entry = {"columns": cols}
        pk = tm.get("primary_key")
        if pk in cols:
            entry["primary_key"] = pk
        out[t] = entry
    return out


def _refill(synth: pd.DataFrame, real: pd.DataFrame, fill_cols, order, seed=0) -> pd.DataFrame:
    """Add ``fill_cols`` back to a synthetic table by resampling each from the
    real column's values (independent bootstrap, preserving the missing-rate),
    then reorder to the original column order.  Used for columns the synthesizer
    did not model (ids / dates / names / audit) so the output keeps all columns."""
    out = synth.copy()
    n = len(out)
    for i, c in enumerate(fill_cols):
        if c not in real.columns:
            continue
        if n == 0:
            out[c] = pd.Series([], dtype=real[c].dtype)
        else:
            out[c] = real[c].sample(n, replace=True, random_state=seed + i + 1).to_numpy()
    return out[[c for c in order if c in out.columns]]


def _referential_integrity(rels, real_tables, suite):
    rows = []
    for r in rels:
        pt, pk = r["parent_table_name"], r["parent_primary_key"]
        ct, fk = r["child_table_name"], r["child_foreign_key"]
        for src, tabs in {"real": real_tables, **suite}.items():
            if pt not in tabs or ct not in tabs or fk not in tabs[ct] or pk not in tabs[pt]:
                continue
            child = tabs[ct][fk].dropna()
            cov = float(child.isin(set(tabs[pt][pk])).mean()) if len(child) else float("nan")
            rows.append({
                "relationship": f"{ct}.{fk} → {pt}.{pk}", "source": src, "fk_coverage": cov,
                "status": "PASS" if cov >= 0.99 else "WARN" if cov >= 0.9 else "FAIL",
            })
    return rows


def _run_job(cfg: dict):
    job = STATE["job"]
    log = job["log"]

    def say(msg):
        log.append(msg)

    def progress(msg):
        """Update a single live '⏳' line (from captured tqdm) instead of spamming."""
        line = "⏳ " + msg
        if log and log[-1].startswith("⏳ "):
            log[-1] = line
        else:
            log.append(line)

    try:
        tables = STATE["tables"]
        tables_meta = _metadata_from_request(cfg.get("schema", {}), cfg.get("relationships", []))
        rels = cfg.get("relationships", [])
        # Fast, pandas-only check of whether the declared relationships actually
        # hold in the data (parent PK unique + every child FK present).  SDV's
        # own Metadata.validate() is skipped on purpose: it re-scans every faker
        # provider from disk per column and is pathologically slow under load —
        # fit() validates internally anyway.
        rels_ok = bool(rels) and _relationships_hold(rels, tables)
        if rels and not rels_ok:
            say("⚠ relationships don't hold in the data — synthesizing independent "
                "tables (relationships still used for the integrity check)")
        # full metadata drives EVALUATION (every column); a reduced schema drives
        # FITTING (only signal columns) so wide tables stay fast.
        full_metadata = _build_metadata(tables_meta, [])
        eval_meta_dict = full_metadata.to_dict()

        say("Splitting train / holdout (before any fitting)…")
        train, hold = _relational_split(tables, rels if rels_ok else [],
                                        cfg["holdout"], cfg.get("seed", 42))
        roles = {t: se.classify_columns(df, full_metadata, t) for t, df in train.items()}

        entity_key = (cfg.get("entity_key") or "").strip()
        entity_children = se.entity_key_tables(tables, entity_key) if entity_key else []
        child_pks = {t: _child_pk(train[t]) for t in entity_children} if entity_children else {}

        # columns that must survive pruning (keys), then model = keys + modelable
        keep_keys = {t: set() for t in train}
        for t in train:
            pk = tables_meta.get(t, {}).get("primary_key")
            if pk:
                keep_keys[t].add(pk)
        for r in (rels if rels_ok else []):
            keep_keys.setdefault(r["parent_table_name"], set()).add(r["parent_primary_key"])
            keep_keys.setdefault(r["child_table_name"], set()).add(r["child_foreign_key"])
        for t in entity_children:
            keep_keys[t].add(entity_key)
            if child_pks.get(t):
                keep_keys[t].add(child_pks[t])
        keep = {t: [c for c in train[t].columns
                    if c in set(roles[t].modelable) | keep_keys[t]] for t in train}
        fill = {t: [c for c in train[t].columns if c not in keep[t]] for t in train}
        reduced_train = {t: train[t][keep[t]].copy() for t in train}
        n_fill = sum(len(v) for v in fill.values())
        if n_fill:
            say(f"Modelling {sum(len(v) for v in keep.values())} signal columns; the other "
                f"{n_fill} id/date/name/audit columns are resampled after fitting "
                f"(kept in the output, keeps fitting fast).")

        # Entity-key (SCD hub) mode -> HMA on a derived parent so referential
        # integrity on the shared key is preserved; else fit the reduced tables.
        parent_name, hub_rels = None, []
        fit_synths = cfg["synths"]
        if entity_children:
            try:
                fit_tables, fit_metadata, hub_rels, hub_info = se.build_entity_hub(
                    reduced_train, entity_key, child_primary_keys=child_pks, lift_invariant=False)
                parent_name = hub_info["parent"]
                fit_synths = ["HMA"]
                say(f"Entity-key mode: built '{parent_name}' over {len(entity_children)} "
                    f"table(s) — {hub_info['n_entities']} distinct {entity_key}. Using HMA so "
                    f"referential integrity on {entity_key} is preserved; other synthesizers "
                    f"are skipped in this mode.")
            except Exception as e:
                say(f"⚠ entity-key mode failed ({e}); falling back to normal synthesis")
                parent_name, hub_rels, fit_synths = None, [], cfg["synths"]
                fit_tables = reduced_train
                fit_metadata = _build_metadata(_reduce_meta(tables_meta, keep), rels if rels_ok else [])
        else:
            fit_tables = reduced_train
            fit_metadata = _build_metadata(_reduce_meta(tables_meta, keep), rels if rels_ok else [])

        say(f"Fitting: {', '.join(fit_synths)} (epochs={cfg['epochs']}, scale={cfg['scale']})")
        _real_err = sys.stderr
        sys.stderr = _TqdmTee(_real_err, progress)     # forward fit progress to the console
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                suite = se.generate_synthetic_suite(
                    fit_tables, fit_metadata, synthesizers=fit_synths,
                    scale=cfg["scale"], epochs=cfg["epochs"], verbose=False,
                    constraints=cfg.get("constraints") or [])
        finally:
            sys.stderr = _real_err
        for w in caught:
            if any(k in str(w.message) for k in ("skipped", "constraints not applied")):
                say(f"⚠ {w.message}")
        if not suite:
            raise RuntimeError("every synthesizer failed — check the schema edits")
        say(f"Synthesis done: {', '.join(suite)}")

        # Referential integrity is measured while the derived parent is present,
        # then the parent is dropped so only real tables are evaluated.
        if parent_name:
            ri = _referential_integrity(hub_rels, fit_tables, suite)
            for s in list(suite):
                suite[s].pop(parent_name, None)
        else:
            ri = _referential_integrity(rels, reduced_train, suite) if rels else []

        # refill the non-modelled columns from real marginals -> every synthetic
        # table has all original columns, in the original order.
        seed0 = cfg.get("seed", 42)
        for s in suite:
            for t in list(suite[s]):
                if fill.get(t):
                    suite[s][t] = _refill(suite[s][t], train[t], fill[t],
                                          list(train[t].columns), seed0)

        # SCD timeline repair: per entity, tile non-overlapping effective/end
        # windows and mark one current row (needs an entity key + real dates).
        scd_eff = (cfg.get("scd_effective") or "").strip()
        scd_end = (cfg.get("scd_end") or "").strip()
        scd_cur = (cfg.get("scd_current") or "").strip() or None
        if entity_key and scd_eff and scd_end:
            noted = set()
            for s in suite:
                for t in list(suite[s]):
                    df = suite[s][t]
                    if entity_key in df.columns and scd_eff in df.columns and scd_end in df.columns:
                        rep, note = se.repair_scd_timeline(df, entity_key, scd_eff, scd_end, scd_cur)
                        suite[s][t] = rep
                        key = (t, note)
                        if note and key not in noted:
                            say(f"⚠ SCD timeline ({t}): {note}"); noted.add(key)
                        elif not note and t not in noted:
                            say(f"SCD timeline repaired for {t}: non-overlapping windows per {entity_key}")
                            noted.add(t)

        meta_dict = eval_meta_dict
        from sdmetrics.reports.single_table import QualityReport

        os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
        quality_scores, shape_details, pair_details, pair_full = {}, {}, {}, {}
        for s, tabs in suite.items():
            quality_scores[s] = {}
            for t, sdf in tabs.items():
                say(f"QualityReport · {s} · {t}")
                qr = QualityReport()
                qr.generate(train[t], sdf, meta_dict["tables"][t], verbose=False)
                props = qr.get_properties().set_index("Property")["Score"]
                quality_scores[s][t] = {
                    "overall": float(qr.get_score()),
                    "column_shapes": float(props.get("Column Shapes", np.nan)),
                    "column_pair_trends": float(props.get("Column Pair Trends", np.nan)),
                }
                det = qr.get_details("Column Shapes")
                shape_details.setdefault(t, {})[s] = det.set_index("Column")["Score"]
                pd_det = qr.get_details("Column Pair Trends")
                # full records feed the pair-trend heatmap; a trimmed copy is
                # sent to the client for the on-page details table.
                pair_full.setdefault(t, {})[s] = pd_det.to_dict(orient="records")
                pair_details.setdefault(t, {})[s] = pd_det.head(60).to_dict(orient="records")

        privacy_all = {}
        for s, tabs in suite.items():
            privacy_all[s] = {}
            for t, sdf in tabs.items():
                say(f"Privacy · {s} · {t}")
                privacy_all[s][t] = se.privacy_report(
                    train[t], hold[t], sdf, roles[t], t, REPORTS_DIR, full_metadata)

        eff_frames = []
        for t in tables:
            tgt = (cfg.get("targets") or {}).get(t) or "auto"
            if tgt == "auto":
                sel = se.auto_select_target(train[t], roles[t])
            else:
                task = "classification" if tgt in roles[t].categorical else "regression"
                sel = (tgt, task)
            if not sel:
                continue
            target, task = sel
            say(f"ML efficacy · {t} · target={target} ({task})")
            synth_dict = {s: tabs[t] for s, tabs in suite.items() if t in tabs}
            eff_frames.append(se.sdmetrics_ml_efficacy(
                train[t], hold[t], synth_dict, roles[t], target, task, table_name=t))
        efficacy = pd.concat(eff_frames, ignore_index=True) if eff_frames else pd.DataFrame()

        say("Leaderboard + comparison figures…")
        leaderboard = se.compute_leaderboard(quality_scores, privacy_all, efficacy)
        fig_dir = f"{REPORTS_DIR}/figures"
        figs = {
            "quality": se.plot_quality_comparison(quality_scores, f"{fig_dir}/web_quality.png"),
            "quality_data": se.quality_comparison_data(quality_scores),
            "privacy": se.plot_privacy_comparison(privacy_all, f"{fig_dir}/web_privacy.png"),
            "efficacy": se.plot_efficacy_scores(efficacy, f"{fig_dir}/web_efficacy.png"),
            # shapes/pairs: interactive Plotly data + a PNG fallback (offline).
            "shapes": {t: se.plot_column_shapes_heatmap(d, f"{fig_dir}/web_shapes_{t}.png", t)
                       for t, d in shape_details.items()},
            "shapes_data": {t: se.shapes_heatmap_data(d) for t, d in shape_details.items()},
            "pairs": {}, "pairs_data": {},
        }
        # Column Pair Trends heatmap per table, for the primary (first) synthesizer.
        primary = list(suite)[0]
        for t, per_synth in pair_full.items():
            recs = per_synth.get(primary)
            if recs:
                figs["pairs"][t] = se.plot_pair_trends_heatmap(
                    recs, f"{fig_dir}/web_pairs_{t}.png", t, primary)
                figs["pairs_data"][t] = se.pair_trends_heatmap_data(recs)

        STATE["suite"] = suite  # kept server-side for CSV downloads
        STATE["results"] = _clean({
            "synths": list(suite),
            "tables": list(tables),
            "palette": se.SYNTH_PALETTE,
            "leaderboard": leaderboard.reset_index().to_dict(orient="records"),
            "quality": quality_scores,
            "shape_details": {t: {s: ser.round(3).to_dict() for s, ser in d.items()}
                              for t, d in shape_details.items()},
            "pair_details": pair_details,
            "referential": ri,
            "relationships_modeled": rels_ok and bool(rels),
            "privacy": {s: {t: {k: v for k, v in rep.items() if k != "dcr_arrays"}
                            for t, rep in tabs.items()}
                        for s, tabs in privacy_all.items()},
            "efficacy": efficacy.to_dict(orient="records") if not efficacy.empty else [],
            "figures": figs,
            "config": {k: cfg[k] for k in ("synths", "epochs", "scale", "holdout")},
        })
        job["status"] = "done"
        say("Done ✓")
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{e}"
        log.append(f"✗ {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.post("/api/upload")
async def upload(files: list[UploadFile]):
    tables = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f.filename))[0].upper()
        tables[name] = pd.read_csv(io.BytesIO(await f.read()), low_memory=False)
    STATE.update(tables=tables, meta_detected=_detect(tables), results=None, job=None)
    return _tables_payload()


@app.post("/api/sample")
def sample():
    paths = sorted(glob.glob("sdg/seed/*.csv"))
    if not paths:
        return JSONResponse({"error": "no sample data found in sdg/seed/"}, status_code=404)
    tables = {os.path.splitext(os.path.basename(p))[0].upper(): pd.read_csv(p, low_memory=False)
              for p in paths}
    STATE.update(tables=tables, meta_detected=_detect(tables), results=None, job=None)
    return _tables_payload()


@app.post("/api/synthesize")
async def synthesize(cfg: dict):
    if STATE["tables"] is None:
        return JSONResponse({"error": "upload data first"}, status_code=400)
    if STATE["job"] and STATE["job"]["status"] == "running":
        return JSONResponse({"error": "a job is already running"}, status_code=409)
    STATE["job"] = {"status": "running", "log": [], "error": None}
    STATE["results"] = None
    threading.Thread(target=_run_job, args=(cfg,), daemon=True).start()
    return {"status": "running"}


@app.get("/api/progress")
def progress():
    job = STATE["job"]
    if not job:
        return {"status": "idle", "log": []}
    return {"status": job["status"], "log": job["log"], "error": job.get("error")}


@app.get("/api/results")
def results():
    if STATE["results"] is None:
        return JSONResponse({"error": "no results yet"}, status_code=404)
    return STATE["results"]


@app.get("/api/download/{synth}/{table}")
def download(synth: str, table: str):
    suite = STATE.get("suite") or {}
    if synth not in suite or table not in suite[synth]:
        return JSONResponse({"error": "not found"}, status_code=404)
    buf = io.StringIO()
    suite[synth][table].to_csv(buf, index=False)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=synthetic_{synth}_{table}.csv"})
