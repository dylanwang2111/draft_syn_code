"""Framework-agnostic core of the Synthetic Data Studio dashboard: in-memory
per-tab session state, upload/detection helpers, and the synthesis+evaluation
worker (_run_job / _start_job) driven by both the manual /api/synthesize
route (server.py) and the chat assistant's run_synthesis tool
(chat_assistant.py). No FastAPI app/routes live here.
"""

from __future__ import annotations

import os
import random
import re
import sys
import threading
import time
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
from fastapi import Request   # only for _sid's type hint below

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import synth_eval as se
from . import profiler

REPORTS_DIR = "reports"

#: every synthesizer except HMA: each fits tables independently, then (when a
#: relationship is declared) gets its foreign keys relinked post-fit so
#: referential integrity holds by construction, the same guarantee HMA gets
#: natively by fitting tables jointly (see synth_eval.link). If a
#: relationship was declared, every synthesizer should honor it -- there's
#: no reason to let one ignore it on purpose.
LINKABLE_SYNTHS = {"GAUSSIANCOPULA", "CTGAN", "TVAE", "COPULAGAN"}

class _Cancelled(Exception):
    """Raised inside a job when the user cancels, to unwind and stop all work."""

#: Per-tab in-memory sessions, keyed by the browser tab's session id so that
#: uploading / editing / synthesizing in one tab never touches another tab's
#: data or job.  Each value is {tables, meta_detected, job, results, suite, _ts}.
SESSIONS: dict[str, dict] = {}
_SESS_LOCK = threading.Lock()
_MAX_SESSIONS = 32


def _new_state() -> dict:
    return {"tables": None, "meta_detected": None, "job": None,
            "results": None, "suite": None, "_ts": time.time(),
            "chat_messages": [], "chat_plan": None}

def _session_for(sid: str) -> dict:
    """Return (creating if needed) the state dict for a tab session id.  Touches
    the last-used timestamp and evicts the oldest idle session past the cap."""
    sid = (sid or "default").strip() or "default"
    with _SESS_LOCK:
        st = SESSIONS.get(sid)
        if st is None:
            st = _new_state()
            SESSIONS[sid] = st
            if len(SESSIONS) > _MAX_SESSIONS:
                # evict the oldest session that isn't mid-run
                idle = [(k, v.get("_ts", 0.0)) for k, v in SESSIONS.items()
                        if k != sid and not (v.get("job") and v["job"].get("status") == "running")]
                if idle:
                    SESSIONS.pop(min(idle, key=lambda kv: kv[1])[0], None)
        st["_ts"] = time.time()
        return st

def _sid(request: Request) -> str:
    """The tab's session id, from the X-Session-Id header (fetch) or ?sid= (links)."""
    return request.headers.get("X-Session-Id") or request.query_params.get("sid") or "default"

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


def _read_csv_robust(src, **kwargs) -> pd.DataFrame:
    """pd.read_csv with an encoding fallback.

    Real-world CSVs -- especially Excel-exported ones, which this schema's own
    data already is (see the scientific-notation-ID / mangled-date caveat in
    docs/DATA_HANDLING.md) -- are frequently Windows-1252, not UTF-8. A hard
    UnicodeDecodeError on upload (e.g. byte 0xE8 mid-file, a smart quote or
    accented character) is a bad first experience for something otherwise
    fully readable. Try utf-8 first (the common case, and stricter, so a
    genuinely UTF-8 file is never silently misread as cp1252), then cp1252,
    then latin-1 as a last resort -- latin-1 maps every byte 0x00-0xFF to a
    character, so it never raises, but also isn't necessarily "correct" if
    the true encoding was something else entirely, it is a deliberate never-
    crash floor, not a best guess.
    """
    for encoding in ("utf-8", "cp1252"):
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            return pd.read_csv(src, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    if hasattr(src, "seek"):
        src.seek(0)
    return pd.read_csv(src, encoding="latin-1", **kwargs)


def _detect(tables: dict[str, pd.DataFrame]) -> dict:
    from sdv.metadata import Metadata

    return Metadata.detect_from_dataframes(tables).to_dict()


def _tables_payload(st: dict):
    tables, meta = st["tables"], st["meta_detected"]
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
            "categoricals": roles.categorical,           # CAP sensitive-column options
            "pii": se.detect_pii(df, roles.modelable),   # {col: kind} for the PII panel
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


def _metadata_from_request(schema: dict, rels: list[dict], st: dict) -> dict:
    """Merge client sdtype/pk edits onto the detected metadata dict."""
    detected = st["meta_detected"]
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


def _normalize_entity_cfg(cfg: dict) -> dict[str, list]:
    """{entity_key: [requested children]} regardless of which wire shape the
    request used. Today's UI only ever builds one hub at a time and sends the
    singular ``entity_key``/``entity_children`` (a flat table list); the API
    also accepts a plural ``entity_keys`` list + ``entity_children`` as
    ``{key: [tables]}`` for callers building more than one hub in one run
    (multi-hub isn't wired into the visual model editor yet — see
    docs/METRICS.md). An empty list of requested children means "every table
    that carries this key", resolved later by ``se.entity_key_tables``.
    """
    keys_field = cfg.get("entity_keys")
    if keys_field:
        children_field = cfg.get("entity_children") or {}
        return {k.strip(): (children_field.get(k) or []) for k in keys_field if k and k.strip()}
    key = (cfg.get("entity_key") or "").strip()
    if not key:
        return {}
    return {key: cfg.get("entity_children") or []}


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
    """Per relationship, per source (real + each synthesizer):

    * ``fk_coverage`` (forward) — share of child FK values that hit a real parent
      key.  1.0 = no orphan children; this drives the PASS/WARN/FAIL status.
    * ``parent_coverage`` (reverse) — share of parent keys that appear in the
      child (i.e. parents that have >=1 child row).  This is *not* expected to be
      1.0 (some entities legitimately have no child row); it's an informational
      fidelity signal — synthetic should match the real ratio, not maximise it.
    """
    rows = []
    for r in rels:
        pt, pk = r["parent_table_name"], r["parent_primary_key"]
        ct, fk = r["child_table_name"], r["child_foreign_key"]
        for src, tabs in {"real": real_tables, **suite}.items():
            if pt not in tabs or ct not in tabs or fk not in tabs[ct] or pk not in tabs[pt]:
                continue
            child = tabs[ct][fk].dropna()
            parent = tabs[pt][pk].dropna()
            cov = float(child.isin(set(parent)).mean()) if len(child) else float("nan")
            rcov = float(parent.isin(set(child)).mean()) if len(parent) else float("nan")
            rows.append({
                "relationship": f"{ct}.{fk} → {pt}.{pk}", "source": src,
                "fk_coverage": cov, "parent_coverage": rcov,
                "status": "PASS" if cov >= 0.99 else "WARN" if cov >= 0.9 else "FAIL",
            })
    return rows


def _merge_cross_table(reports: list[dict]) -> dict:
    """Combine one se.entity_cross_table_trends() dict per simultaneous hub
    key into the single dict the report/frontend expect (the multi-hub
    pipeline runs this once per key, but the wire shape predates multi-hub and
    is one dict per synthesizer, not one per synthesizer-per-key)."""
    reports = [r for r in reports if r]
    empty = {"score": None, "score_strong": None, "n_pairs": 0, "n_strong": 0,
             "n_entities": 0, "pairs": [], "unaligned": [], "note": None}
    if not reports:
        return empty
    if len(reports) == 1:
        return reports[0]
    scores = [r["score"] for r in reports if r.get("score") is not None]
    strong = [r["score_strong"] for r in reports if r.get("score_strong") is not None]
    notes = [r["note"] for r in reports if r.get("note")]
    return {
        "score": float(np.mean(scores)) if scores else None,
        "score_strong": float(np.mean(strong)) if strong else None,
        "n_pairs": sum(r.get("n_pairs") or 0 for r in reports),
        "n_strong": sum(r.get("n_strong") or 0 for r in reports),
        "n_entities": sum(r.get("n_entities") or 0 for r in reports),
        "pairs": [p for r in reports for p in (r.get("pairs") or [])],
        "unaligned": [u for r in reports for u in (r.get("unaligned") or [])],
        "note": " · ".join(notes) if notes else None,
    }


def _validate_relationships(tables: dict, cfg: dict) -> list[dict]:
    """Structurally validate a candidate data model against the *real* data:
    for each declared relationship check the parent key is unique and measure
    how many child FK values point at a real parent key; for the entity-key hub
    confirm the key is present in the chosen tables.  Read-only, no synthesis.
    Shared by the /api/validate_model route and the chat assistant's tools,
    so a link checked by clicking and one checked by the bot get the same answer."""
    results = []
    for r in cfg.get("relationships") or []:
        pt, pk = r.get("parent_table_name"), r.get("parent_primary_key")
        ct, fk = r.get("child_table_name"), r.get("child_foreign_key")
        label = f"{ct}.{fk} → {pt}.{pk}"
        if pt not in tables or ct not in tables or pk not in tables.get(pt, {}).columns or fk not in tables.get(ct, {}).columns:
            results.append({"label": label, "status": "FAIL", "detail": "table or column not found"})
            continue
        parent_keys = tables[pt][pk]
        child = tables[ct][fk].dropna()
        if parent_keys.duplicated().any():
            results.append({"label": label, "status": "FAIL",
                            "detail": f"parent key {pt}.{pk} is not unique — not a valid primary key"})
        elif not len(child):
            results.append({"label": label, "status": "WARN", "detail": "child foreign key is entirely null"})
        else:
            cov = float(child.isin(set(parent_keys.dropna())).mean())
            status = "PASS" if cov >= 0.99 else "WARN" if cov >= 0.90 else "FAIL"
            results.append({"label": label, "status": status,
                            "detail": f"{cov*100:.1f}% of child rows match a parent key"})
    for entity_key, requested in _normalize_entity_cfg(cfg).items():
        avail = se.entity_key_tables(tables, entity_key)
        chosen = [t for t in requested if t in avail] if requested else avail
        if not chosen:
            results.append({"label": f"hub {entity_key}", "status": "FAIL",
                            "detail": f"'{entity_key}' is not present in any selected table"})
        else:
            ids = set()
            for t in chosen:
                local_col = se._resolve_key_column(tables[t], entity_key) or entity_key
                ids |= set(tables[t][local_col].dropna().unique())
            results.append({"label": f"hub {entity_key} → {', '.join(chosen)}", "status": "PASS",
                            "detail": f"{len(ids)} distinct entities · select HMA to have referential "
                                      f"integrity on {entity_key} preserved by construction"})
    return results


def _run_job(cfg: dict, st: dict):
    job = st["job"]
    log = job["log"]
    t_job_start = time.perf_counter()
    phase_seconds: dict = {}   # {"prep","generate","resample","postprocess","report"} wall-clock

    def set_pct(p):
        """Monotonically advance the job's percent-complete (capped < 100)."""
        job["pct"] = max(float(job.get("pct", 0.0)), min(99.0, float(p)))

    def say(msg):
        log.append(msg)

    def cancelled():
        return bool(job.get("cancel"))

    def ck():
        """Cancellation checkpoint: raise if the user asked to cancel this job."""
        if cancelled():
            raise _Cancelled()

    def progress(msg):
        """Update a single live '⏳' line (from captured tqdm) instead of spamming.
        Also maps the fit phase/percent onto the overall progress bar (5–40%)."""
        line = "⏳ " + msg
        if log and log[-1].startswith("⏳ "):
            log[-1] = line
        else:
            log.append(line)
        m = re.search(r"(Preprocess|Learning|Modeling|Sampling|report)\D*?(\d+)\s*%", msg)
        if m:
            base = {"Preprocess": 5, "Learning": 10, "Modeling": 22,
                    "Sampling": 34, "report": 38}.get(m.group(1), 10)
            set_pct(base + 0.06 * float(m.group(2)))

    # A job-wide heartbeat: whenever visible progress stalls for a few seconds —
    # HMA's un-instrumented augment/sample, or a single slow metric on wide data —
    # show a live "still working — <last step> … <elapsed>" line so the run never
    # looks frozen.  In HMA's post-"Modeling Tables" dead zone (26–39%) it also
    # creeps the bar so it visibly moves.
    stop_beat = threading.Event()

    def _beat_line(msg):
        line = "⏳ " + msg
        if log and log[-1].startswith("⏳ "):
            log[-1] = line
        else:
            log.append(line)

    def _heartbeat():
        last_step, last_pct, quiet, step_t0 = None, 0.0, 0.0, time.time()
        while not stop_beat.wait(3.0):
            cur = job.get("pct", 0.0)
            reals = [l for l in log if not l.startswith("⏳ ")]
            top = reals[-1] if reals else None
            if top != last_step or cur > last_pct + 0.01:      # real progress -> reset
                last_step, last_pct, quiet, step_t0 = top, cur, 0.0, time.time()
                continue
            quiet += 3.0
            if quiet >= 6.0:
                if 26.0 <= cur < 39.0:                          # HMA fit dead zone: creep
                    set_pct(cur + (39.0 - cur) * 0.05)
                    last_pct = job.get("pct", 0.0)
                el = int(time.time() - step_t0)
                ctx = (top or "working")[:58]
                _beat_line(f"still working — {ctx} … {el // 60}m{el % 60:02d}s")

    threading.Thread(target=_heartbeat, daemon=True).start()

    try:
        t_prep_start = time.perf_counter()
        # Seed every global RNG this pipeline touches, once, up front: numpy
        # (train/holdout split, sklearn estimators with random_state=None
        # like sdmetrics' DecisionTreeClassifier efficacy scorer, which reads
        # the global numpy state since it takes no seed parameter of its
        # own), stdlib random (Faker-backed PII faking), and torch if CTGAN/
        # TVAE/CopulaGAN are in play. SDV's own synthesizers keep a
        # model-local RNG instead of using the global one, see
        # synth_eval.suite for the matching per-synthesizer seeding.
        run_seed = cfg.get("seed", 42)
        np.random.seed(run_seed)
        random.seed(run_seed)
        try:
            import torch
            torch.manual_seed(run_seed)
        except ImportError:
            pass
        # Scale-sensitive thresholds: hardcoded Python defaults tuned against
        # the demo-scale seed data, exposed here so real production data (a
        # different row count, a different cardinality distribution) doesn't
        # need a code change to get a sane result -- same defaults as before,
        # so nothing changes unless a run explicitly sets one.
        # * max_categorical_card: above this many distinct values, a would-be
        #   categorical column is treated as id-like and skipped instead.
        #   Too low on a real column set with genuinely wide categories
        #   silently drops it out of fidelity/privacy scoring entirely.
        # * min_target_rows: below this many rows, auto_select_target won't
        #   pick an ML-efficacy target for a table (holdout too small to
        #   mean anything).
        # * close_percentile: how aggressive the reject-and-resample filter
        #   and the nearest-record ceiling are. Nearest-neighbor distances
        #   shrink as row count grows (denser space), so the same fixed
        #   percentile reads as stricter on a much bigger real table than on
        #   the seed data this was tuned against.
        max_categorical_card = int(cfg.get("max_categorical_card") or 50)
        min_target_rows = int(cfg.get("min_target_rows") or 30)
        close_percentile = float(cfg.get("close_percentile") or 5.0)
        tables = st["tables"]
        tables_meta = _metadata_from_request(cfg.get("schema", {}), cfg.get("relationships", []), st)
        rels = cfg.get("relationships", [])

        # Entity key detection + naming-convention normalization, done here,
        # early, BEFORE full_metadata/roles get built below, so both are
        # already consistent with whichever child tables get a local key
        # variant (e.g. an X_-prefixed extension-field name) renamed to the
        # canonical form -- computing this later meant full_metadata/roles
        # were snapshotted against the OLD name, silently dropping that one
        # column out of quality/privacy scoring for the affected tables
        # (the metadata and the data no longer agreed on what it was called).
        # Detection only needs `tables`/`cfg`, both already available here;
        # only the entity_renames rename onto train/hold has to wait until
        # those exist (right after the split below).
        # entity_children_by_key: {entity_key: [chosen child tables]} -- one or
        # more simultaneous hubs (today's UI only ever builds one at a time, but
        # the pipeline and se.build_entity_hub support any number; see
        # _normalize_entity_cfg).
        entity_children_by_key: dict[str, list[str]] = {}
        for entity_key, requested in _normalize_entity_cfg(cfg).items():
            avail_children = se.entity_key_tables(tables, entity_key)
            entity_children_by_key[entity_key] = (
                [t for t in requested if t in avail_children] if requested else avail_children)
        entity_keys = list(entity_children_by_key)
        # every table that's a child of at least one hub -- most of the
        # downstream logic (pruning, keep-keys, child PKs) doesn't care WHICH
        # hub, just that the column has to survive
        entity_children = sorted({t for children in entity_children_by_key.values() for t in children})
        # {table: {local_col: canonical_key}} -- a table can carry more than one
        # hub's local key variant (e.g. PERSON could have both an X_-prefixed
        # occupation key and a contact-id key), so this is per (table, key), not
        # just per table.
        entity_renames: dict[str, dict[str, str]] = {}
        for entity_key, children in entity_children_by_key.items():
            for t in children:
                local_col = se._resolve_key_column(tables[t], entity_key)
                if local_col and local_col != entity_key:
                    entity_renames.setdefault(t, {})[local_col] = entity_key
                    cols_meta = tables_meta.get(t, {}).get("columns", {})
                    if local_col in cols_meta:
                        cols_meta[entity_key] = cols_meta.pop(local_col)
                    if tables_meta.get(t, {}).get("primary_key") == local_col:
                        tables_meta[t]["primary_key"] = entity_key

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

        set_pct(3)
        say("Splitting train / holdout (before any fitting)…")
        train, hold = _relational_split(tables, rels if rels_ok else [],
                                        cfg["holdout"], cfg.get("seed", 42))
        # apply the SAME rename to the working train/holdout copies -- never
        # to st["tables"] itself, so the Schema/Data tabs and downloads keep
        # showing exactly what was uploaded, only this run's internal working
        # copies see the canonical name
        for t, renames in entity_renames.items():
            train[t] = train[t].rename(columns=renames)
            hold[t] = hold[t].rename(columns=renames)
        roles = {t: se.classify_columns(df, full_metadata, t, max_categorical_card=max_categorical_card)
                 for t, df in train.items()}

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
        for entity_key, children in entity_children_by_key.items():
            for t in children:
                keep_keys[t].add(entity_key)
        for t in entity_children:
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

        # Entity-key (SCD hub) mode -> HMA on one derived parent per key so
        # referential integrity on each shared key is preserved; else fit the
        # reduced tables. parent_names maps {entity_key: hub table name} -- one
        # entry per hub, empty when no entity key is set at all.
        parent_names: dict[str, str] = {}
        hub_rels: list = []
        fit_synths = cfg["synths"]
        # only fit HMA if it was actually selected — the hub is what lets HMA preserve
        # referential integrity, but choosing an entity key must not conscript HMA.
        want_hma = any(s.upper() == "HMA" for s in cfg["synths"])
        if entity_children:
            try:
                fit_tables, fit_metadata, hub_rels, hub_info = se.build_entity_hub(
                    reduced_train, entity_keys, child_primary_keys=child_pks,
                    lift_invariant=False, child_tables=entity_children_by_key)
                parent_names = {k: info["parent"] for k, info in hub_info.items()}
                others_note = [s for s in cfg["synths"] if s.upper() != "HMA"]
                _v = "is" if len(others_note) == 1 else "are"
                for entity_key, info in hub_info.items():
                    say(f"Entity-key mode: built '{info['parent']}' over {len(info['children'])} "
                        f"table(s) — {info['n_entities']} distinct {entity_key}."
                        + (f" HMA models the hub so referential integrity on {entity_key} is "
                           f"preserved." if want_hma else
                           f" HMA was not selected, so no model learns the hub — referential "
                           f"integrity is measured, not enforced."))
                if others_note:
                    say(f"{', '.join(others_note)} {_v} fit per-table independently "
                        f"(single-table models can't preserve cross-table RI — shown for "
                        f"quality/privacy/utility comparison only).")
            except Exception as e:
                say(f"⚠ entity-key mode failed ({e}); falling back to normal synthesis")
                parent_names, hub_rels, fit_synths = {}, [], cfg["synths"]
                fit_tables = reduced_train
                fit_metadata = _build_metadata(_reduce_meta(tables_meta, keep), rels if rels_ok else [])
        else:
            fit_tables = reduced_train
            fit_metadata = _build_metadata(_reduce_meta(tables_meta, keep), rels if rels_ok else [])

        phase_seconds["prep"] = time.perf_counter() - t_prep_start

        # epochs only apply to the neural synthesizers; HMA / GaussianCopula have none
        _uses_epochs = any(s.upper() in ("CTGAN", "TVAE", "COPULAGAN") for s in fit_synths)
        say(f"Fitting: {', '.join(fit_synths)} (scale={cfg['scale']}"
            + (f", epochs={cfg['epochs']}" if _uses_epochs else "") + ")")
        _real_err = sys.stderr
        sys.stderr = _TqdmTee(_real_err, progress)     # forward fit progress to the console
        gen_timings: dict = {}   # {synth_name: fit+sample seconds} -- the time-savings metric
        resample_timings: dict = {}  # {synth_name: reject-and-resample filter seconds}
        close_filter_report: dict = {}  # {synth_name: {table: report}} -- reject-and-resample filter
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if parent_names:
                    # HMA models the derived hub (preserves cross-table RI on the
                    # key) — but only if the user actually asked for HMA.  Any other
                    # selected synths are single-table, so fit them independently on
                    # the reduced tables (no relationships); they still appear in the
                    # quality/privacy/utility comparison.
                    suite = {}
                    if want_hma:
                        suite = se.generate_synthetic_suite(
                            fit_tables, fit_metadata, synthesizers=["HMA"],
                            scale=cfg["scale"], epochs=cfg["epochs"], verbose=False,
                            constraints=cfg.get("constraints") or [], should_cancel=cancelled,
                            timings=gen_timings, roles=roles,
                            close_filter_report=close_filter_report,
                            resample_timings=resample_timings, random_state=run_seed,
                            filter_close_percentile=close_percentile)
                    others = [s for s in cfg["synths"] if s.upper() != "HMA"]
                    if others and not cancelled():
                        single_meta = _build_metadata(_reduce_meta(tables_meta, keep), [])
                        suite.update(se.generate_synthetic_suite(
                            reduced_train, single_meta, synthesizers=others,
                            scale=cfg["scale"], epochs=cfg["epochs"], verbose=False,
                            constraints=cfg.get("constraints") or [], should_cancel=cancelled,
                            timings=gen_timings, roles=roles,
                            close_filter_report=close_filter_report,
                            resample_timings=resample_timings, random_state=run_seed,
                            filter_close_percentile=close_percentile))
                else:
                    suite = se.generate_synthetic_suite(
                        fit_tables, fit_metadata, synthesizers=fit_synths,
                        scale=cfg["scale"], epochs=cfg["epochs"], verbose=False,
                        constraints=cfg.get("constraints") or [], should_cancel=cancelled,
                        roles=roles, close_filter_report=close_filter_report,
                        filter_close_percentile=close_percentile,
                        timings=gen_timings, resample_timings=resample_timings,
                        random_state=run_seed)
        finally:
            sys.stderr = _real_err
        # gen_timings/resample_timings are per-synthesizer wall-clock (see
        # synth_eval.suite.generate_synthetic_suite); synthesizers fit
        # sequentially in that loop, so summing them is the phase's real
        # wall-clock, not double-counted.
        phase_seconds["generate"] = sum(gen_timings.values())
        phase_seconds["resample"] = sum(resample_timings.values())
        ck()   # stop here if cancelled during/after fitting
        for w in caught:
            if any(k in str(w.message) for k in ("skipped", "constraints not applied")):
                say(f"⚠ {w.message}")
        if not suite:
            raise RuntimeError("every synthesizer failed — check the schema edits")
        say(f"Synthesis done: {', '.join(suite)}")
        t_postprocess_start = time.perf_counter()

        # Every synthesizer except HMA fits each table on its own, so a
        # child's foreign-key column is unrelated to the synthetic parent's
        # keys: relink it now (HMA and the entity-key hub path above don't
        # need this, they already model the link directly).
        linked_synths: list[str] = []
        if not parent_names and rels and rels_ok:
            linkable = [s for s in suite if s.upper() in LINKABLE_SYNTHS]
            if linkable:
                linked = se.link_relationships(rels, reduced_train, suite, linkable,
                                               seed=cfg.get("seed", 42))
                linked_synths = [s for s in linked if linked[s]]
                if linked_synths:
                    say(f"Linked foreign keys for {', '.join(linked_synths)} so referential "
                        f"integrity holds (cardinality resampled from the real per-parent shape).")

        # Referential integrity + cardinality are measured while the derived
        # parent is present (they need the parent table), then the parent is
        # dropped so only real tables are evaluated by the other metrics.
        cardinality = {}
        if parent_names:
            # HMA emits the hub itself; for the independent single-table synths,
            # derive an equivalent hub per key (the distinct keys they generated
            # across that key's child tables) so referential integrity +
            # cardinality are measured and comparable for every synthesizer, not
            # just HMA.
            for s, tabs in suite.items():
                for entity_key, pname in parent_names.items():
                    if pname in tabs:
                        continue
                    ids = set()
                    for t in entity_children_by_key.get(entity_key, []):
                        if t in tabs and entity_key in tabs[t].columns:
                            ids |= set(tabs[t][entity_key].dropna().unique())
                    tabs[pname] = pd.DataFrame(
                        {entity_key: sorted(ids, key=lambda v: (str(type(v)), str(v)))})
            # That per-key union is only a coherent shared pool once every child
            # is actually relinked against it: each single-table synth fit its
            # own copy of the key column independently, so before this the same
            # entity_key value in two children of the same hub (or, for a
            # multi-hub child like PERSON, its two DIFFERENT keys) are
            # unrelated numbers that happen to share a column name. Relinking
            # (real per-parent count shape, values drawn from the pool just
            # built above) is exactly what the plain-relationship path already
            # does for non-hub FKs (see LINKABLE_SYNTHS above) -- same
            # mechanism, applied to hub_rels instead of rels. HMA already
            # models each hub jointly with its children, so it's excluded here
            # the same way it's excluded from the "others" fit above.
            hub_linkable = [s for s in suite if s.upper() != "HMA" and s.upper() in LINKABLE_SYNTHS]
            if hub_linkable:
                hub_linked = se.link_relationships(hub_rels, fit_tables, suite, hub_linkable,
                                                    seed=cfg.get("seed", 42))
                hub_linked_synths = [s for s in hub_linked if hub_linked[s]]
                if hub_linked_synths:
                    say(f"Linked hub foreign keys for {', '.join(hub_linked_synths)} so "
                        f"referential integrity holds on the derived hub(s) too "
                        f"(cardinality resampled from the real per-parent shape).")
            ri = _referential_integrity(hub_rels, fit_tables, suite)
            try:
                cardinality = se.cardinality_report(fit_metadata, fit_tables, suite)
            except Exception as e:
                say(f"⚠ cardinality metrics skipped: {e}")
            for s in list(suite):
                for pname in parent_names.values():
                    suite[s].pop(pname, None)
        elif rels and rels_ok:
            ri = _referential_integrity(rels, reduced_train, suite)
            try:
                cardinality = se.cardinality_report(fit_metadata, reduced_train, suite)
            except Exception as e:
                say(f"⚠ cardinality metrics skipped: {e}")
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

        # PII policies: the bootstrap refill above re-deals REAL values, so name/
        # email/phone-like columns would carry real strings into the "synthetic"
        # output.  Default = fake (Faker values that never existed in the source);
        # the UI can opt a column into shuffle (keep bootstrap) or drop.
        pii_cfg = cfg.get("pii") or {}
        pii_plan, pii_log = {}, {"fake": [], "drop": [], "shuffle": []}
        for t in train:
            plan = {}
            for c, kind in se.detect_pii(train[t], roles[t].modelable).items():
                pol = str((pii_cfg.get(t) or {}).get(c, "fake")).lower()
                if pol in ("fake", "drop"):
                    plan[c] = (pol, kind)
                    pii_log[pol].append(f"{t}.{c}")
                else:
                    pii_log["shuffle"].append(f"{t}.{c}")
            if plan:
                pii_plan[t] = plan
        if any(pii_log.values()):
            parts = []
            if pii_log["fake"]:
                parts.append(f"faked {len(pii_log['fake'])} column(s): {', '.join(pii_log['fake'])}")
            if pii_log["drop"]:
                parts.append(f"dropped {', '.join(pii_log['drop'])}")
            if pii_log["shuffle"]:
                parts.append(f"⚠ kept REAL values (shuffled) in {', '.join(pii_log['shuffle'])}")
            say("PII: " + " · ".join(parts))
        for s in suite:
            for t in list(suite[s]):
                if pii_plan.get(t):
                    suite[s][t] = se.apply_pii_plan(suite[s][t], pii_plan[t], train[t], seed0)
        # faked/dropped columns are deliberately NOT faithful to the real
        # marginals (that's the point) — take them out of the evaluation
        # metadata so QualityReport neither crashes on a dropped column nor
        # scores fidelity we intentionally destroyed for privacy.  Shuffled
        # columns keep the real marginal, so they stay scoreable.
        for t, plan in pii_plan.items():
            tmeta = eval_meta_dict.get("tables", {}).get(t)
            if tmeta:
                for c in plan:
                    tmeta.get("columns", {}).pop(c, None)

        # SCD timeline repair: per entity, tile non-overlapping effective/end
        # windows and mark one current row (needs an entity key + real dates).
        scd_eff = (cfg.get("scd_effective") or "").strip()
        scd_end = (cfg.get("scd_end") or "").strip()
        scd_cur = (cfg.get("scd_current") or "").strip() or None
        if entity_keys and scd_eff and scd_end:
            noted = set()
            for s in suite:
                for t in list(suite[s]):
                    df = suite[s][t]
                    # a table versions against at most one hub key in practice,
                    # but this loops every configured key so it isn't assumed
                    for entity_key in entity_keys:
                        if entity_key in df.columns and scd_eff in df.columns and scd_end in df.columns:
                            df, note = se.repair_scd_timeline(df, entity_key, scd_eff, scd_end, scd_cur)
                            key = (t, entity_key, note)
                            if note and key not in noted:
                                say(f"⚠ SCD timeline ({t} · {entity_key}): {note}"); noted.add(key)
                            elif not note and (t, entity_key) not in noted:
                                say(f"SCD timeline repaired for {t}: non-overlapping windows per {entity_key}")
                                noted.add((t, entity_key))
                    suite[s][t] = df

        # Entity-level cross-table correlation: do a customer's attributes across
        # tables (marital status in PERSON vs province in CONTACT) hang together
        # like real?  Single-table Column Pair Trends can't see across tables and
        # the built-in Intertable Trends needs an attribute-bearing parent (our
        # hub is keys only), so this is measured directly per synthesizer.  A
        # model that doesn't keep the entity key consistent across tables reads
        # n/a here rather than being scored (see synth_eval.cross_table).
        cross_table = {}
        if entity_keys:
            cur_flags = {t: scd_cur for t in tables} if scd_cur else None
            eff_cols = {t: scd_eff for t in tables} if scd_eff else None
            for s, tabs in suite.items():
                cross_table[s] = _merge_cross_table([
                    se.entity_cross_table_trends(train, tabs, entity_key, roles, cur_flags, eff_cols)
                    for entity_key in entity_keys
                ])

        phase_seconds["postprocess"] = time.perf_counter() - t_postprocess_start
        t_report_start = time.perf_counter()

        meta_dict = eval_meta_dict
        from sdmetrics.reports.single_table import QualityReport

        set_pct(40)
        os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
        # step counts drive the progress bar across the evaluation phases
        n_st = sum(len(tabs) for tabs in suite.values())         # synth × table
        n_tab = len(tables)
        done_q = done_p = done_e = 0
        quality_scores, shape_details, pair_details, pair_full = {}, {}, {}, {}
        for s, tabs in suite.items():
            quality_scores[s] = {}
            for t, sdf in tabs.items():
                ck()
                say(f"QualityReport · {s} · {t}")
                qr = QualityReport()
                # slice both frames to the evaluation metadata's columns: PII
                # policies may have removed/faked columns, and sdmetrics wants
                # data and metadata to agree exactly.
                tmeta = meta_dict["tables"][t]
                mcols = [c for c in tmeta.get("columns", {})
                         if c in train[t].columns and c in sdf.columns]
                sub_meta = {**tmeta, "columns": {c: tmeta["columns"][c] for c in mcols}}
                qr.generate(train[t][mcols], sdf[mcols], sub_meta, verbose=False)
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
                done_q += 1; set_pct(40 + 20 * done_q / max(1, n_st))

        privacy_all = {}
        for s, tabs in suite.items():
            privacy_all[s] = {}
            for t, sdf in tabs.items():
                ck()
                say(f"Privacy · {s} · {t}")
                privacy_all[s][t] = se.privacy_report(
                    train[t], hold[t], sdf, roles[t], t, REPORTS_DIR, full_metadata,
                    cap_sensitive=(cfg.get("cap_sensitive") or {}).get(t) or None,
                    close_percentile=close_percentile)
                done_p += 1; set_pct(60 + 20 * done_p / max(1, n_st))

        eff_frames = []
        efficacy_skipped = []  # [{table, target, task, reason}] -- surfaced in the report, not just the log
        efficacy_notes = []    # [{table, target, task, note}] -- caution, score still computed

        def skip_eff(t, reason, target=None, task=None):
            say(f"ML efficacy · {t} · skipped ({reason})")
            efficacy_skipped.append({"table": t, "target": target, "task": task, "reason": reason})

        def note_eff(t, note, target=None, task=None):
            say(f"ML efficacy · {t} · note: {note}")
            efficacy_notes.append({"table": t, "target": target, "task": task, "note": note})

        for t in tables:
            ck()
            tgt = (cfg.get("targets") or {}).get(t) or "auto"
            if tgt == "none":
                skip_eff(t, "marked no-target — e.g. a dimension/lookup table")
                continue
            if tgt == "auto":
                sel = se.auto_select_target(train[t], roles[t], min_rows=min_target_rows)
                if not sel:
                    skip_eff(t, "no suitable target — table too small or too few columns "
                                "left to predict from once id/name/date fields are excluded")
                    continue
            else:
                task = "classification" if tgt in roles[t].categorical else "regression"
                sel = (tgt, task)
            target, task = sel
            # whether the target is actually predictable from the table's other
            # columns -- not a row-shape proxy (a table full of legitimate
            # low-cardinality codes looks identical to an SCD dimension table by
            # row-distinctness alone), but the real, direct signal check: does a
            # model beat its own noise floor on this exact target? Runs for BOTH
            # auto-picked and manually-forced targets -- auto_select_target's own
            # min-lift gate only skips targets with essentially NO signal, so a
            # borderline pick can still clear it and land here as a caution.
            sig_note = se.target_signal_note(train[t], target, roles[t], task)
            if sig_note:
                note_eff(t, sig_note, target, task)
            say(f"ML efficacy · {t} · target={target} ({task})")
            synth_dict = {s: tabs[t] for s, tabs in suite.items() if t in tabs}
            eff_frames.append(se.sdmetrics_ml_efficacy(
                train[t], hold[t], synth_dict, roles[t], target, task, table_name=t))
            done_e += 1; set_pct(80 + 12 * done_e / max(1, n_tab))
        efficacy = pd.concat(eff_frames, ignore_index=True) if eff_frames else pd.DataFrame()

        ck()
        set_pct(93)
        say("Leaderboard + comparison figures…")
        # Referential integrity feeds the fidelity score (as cardinality shape
        # similarity — see compare.structure_scores).  `derived_parent` says the
        # hub was built from the synthesizers' own keys, which makes forward FK
        # coverage 1.0 by construction: a diagnostic, never a score.
        derived = bool(parent_names)
        summary = se.compute_summary(quality_scores, privacy_all, efficacy, ri, cardinality, derived)
        leaderboard = se.compute_leaderboard(quality_scores, privacy_all, efficacy, ri,
                                             cardinality, derived)
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
        # Column Pair Trends heatmap per synthesizer per table (a pair matrix is
        # 2-D column×column, so it can't be stacked like the shapes heatmap — one
        # per synthesizer).  Shape: figs["pairs"][synth][table].
        for t, per_synth in pair_full.items():
            for s, recs in per_synth.items():
                if recs:
                    figs["pairs"].setdefault(s, {})[t] = se.plot_pair_trends_heatmap(
                        recs, f"{fig_dir}/web_pairs_{s}_{t}.png", t, s)
                    figs["pairs_data"].setdefault(s, {})[t] = se.pair_trends_heatmap_data(recs)

        phase_seconds["report"] = time.perf_counter() - t_report_start
        phase_seconds["total"] = time.perf_counter() - t_job_start

        ck()   # last checkpoint before publishing
        if st.get("job") is not job:
            return   # a newer job in this session superseded us; don't clobber it
        st["suite"] = suite  # kept server-side for CSV downloads
        st["results"] = _clean({
            "synths": list(suite),
            "tables": list(tables),
            "palette": se.SYNTH_PALETTE,
            "leaderboard": leaderboard.reset_index().to_dict(orient="records"),
            "summary": summary,
            "quality": quality_scores,
            "shape_details": {t: {s: ser.round(3).to_dict() for s, ser in d.items()}
                              for t, d in shape_details.items()},
            "pair_details": pair_details,
            "referential": ri,
            "cardinality": cardinality,
            "cross_table": cross_table,
            "relationships_modeled": rels_ok and bool(rels),
            "linked_synths": linked_synths,
            "gen_seconds": {s: gen_timings.get(s) for s in suite},
            "resample_seconds": {s: resample_timings.get(s) for s in suite},
            "close_filter": close_filter_report,
            "phase_seconds": phase_seconds,
            "privacy": {s: {t: {k: v for k, v in rep.items() if k != "dcr_arrays"}
                            for t, rep in tabs.items()}
                        for s, tabs in privacy_all.items()},
            "efficacy": efficacy.to_dict(orient="records") if not efficacy.empty else [],
            "efficacy_skipped": efficacy_skipped,
            "efficacy_notes": efficacy_notes,
            "figures": figs,
            "config": {k: cfg[k] for k in ("synths", "epochs", "scale", "holdout")},
        })
        job["pct"] = 100.0
        job["status"] = "done"
        say("Done ✓")
    except _Cancelled:
        job["status"] = "cancelled"
        say("■ cancelled — synthesis stopped, no report generated")
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{e}"
        log.append(f"✗ {e}")
        traceback.print_exc()
    finally:
        stop_beat.set()   # stop the heartbeat thread


def _start_job(cfg: dict, st: dict) -> dict:
    """Kick off a synthesis job on this session. Shared by the manual
    /api/synthesize route and the chat assistant's run_synthesis tool, so a
    run started by clicking and one started by telling the bot behave
    identically (same job/progress/results machinery, same report after)."""
    if st["tables"] is None:
        return {"error": "upload data first"}
    if st["job"] and st["job"]["status"] == "running":
        return {"error": "a job is already running in this tab"}
    st["job"] = {"status": "running", "log": [], "error": None, "pct": 0.0}
    st["results"] = None
    threading.Thread(target=_run_job, args=(cfg, st), daemon=True).start()
    return {"status": "running"}
