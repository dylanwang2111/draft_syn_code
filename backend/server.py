"""FastAPI backend for the Synthetic Data Studio dashboard (web/index.html).

Endpoints (single local user; state kept in memory):
    GET  /                      -> the dashboard SPA
    POST /api/upload            -> multipart CSVs; returns preview + detected metadata
    POST /api/sample            -> load bundled sdg/seed/*.csv instead of uploading
    POST /api/synthesize        -> start synthesis + evaluation in a worker thread
    GET  /api/progress          -> live progress log for the running job
    GET  /api/results           -> full evaluation report (JSON + base64 figures)
    GET  /api/download/{s}/{t}  -> synthetic CSV for synthesizer s, table t

Business logic (session state, upload/detection helpers, the synthesis+eval
worker) lives in dashboard_core.py; the chat assistant's LLM tool-calling
layer and its /api/chat/* routes live in chat_assistant.py (mounted below).
This file only wires up the thin HTTP layer for the core dashboard.

Run with:
    uvicorn backend.server:app --port 8000   (from the repo root, then open http://localhost:8000)
"""

from __future__ import annotations

import glob
import io
import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .dashboard_core import _detect, _session_for, _sid, _start_job, _tables_payload, _validate_relationships
from . import chat_assistant

app = FastAPI(title="Synthetic Data Studio")

# web/ lives at the repo root, a sibling of this backend/ package, not inside it
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

app.include_router(chat_assistant.router)

# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.post("/api/upload")
async def upload(files: list[UploadFile], request: Request):
    st = _session_for(_sid(request))
    tables = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f.filename))[0].upper()
        tables[name] = pd.read_csv(io.BytesIO(await f.read()), low_memory=False)
    st.update(tables=tables, meta_detected=_detect(tables), results=None, job=None, suite=None)
    return _tables_payload(st)


@app.post("/api/sample")
def sample(request: Request):
    st = _session_for(_sid(request))
    paths = sorted(glob.glob("sdg/seed/*.csv"))
    if not paths:
        return JSONResponse({"error": "no sample data found in sdg/seed/"}, status_code=404)
    tables = {os.path.splitext(os.path.basename(p))[0].upper(): pd.read_csv(p, low_memory=False)
              for p in paths}
    st.update(tables=tables, meta_detected=_detect(tables), results=None, job=None, suite=None)
    return _tables_payload(st)


@app.post("/api/validate_model")
def validate_model(cfg: dict, request: Request):
    tables = _session_for(_sid(request))["tables"]
    if tables is None:
        return JSONResponse({"error": "upload data first"}, status_code=400)
    return {"results": _validate_relationships(tables, cfg)}

@app.post("/api/synthesize")
async def synthesize(cfg: dict, request: Request):
    result = _start_job(cfg, _session_for(_sid(request)))
    if result.get("error"):
        status = 400 if result["error"] == "upload data first" else 409
        return JSONResponse(result, status_code=status)
    return result


@app.post("/api/cancel")
def cancel(request: Request):
    """Cancel this tab's running job. Flags it cancelled immediately (so the UI
    frees up and a new run is allowed); the worker thread unwinds all remaining
    synthesis/evaluation work at its next checkpoint and produces no report."""
    job = _session_for(_sid(request)).get("job")
    if not job or job.get("status") != "running":
        return {"status": (job or {}).get("status", "idle")}
    job["cancel"] = True
    job["status"] = "cancelled"
    return {"status": "cancelled"}


@app.get("/api/progress")
def progress(request: Request):
    job = _session_for(_sid(request))["job"]
    if not job:
        return {"status": "idle", "log": []}
    return {"status": job["status"], "log": job["log"], "error": job.get("error"),
            "pct": round(float(job.get("pct", 0.0)), 1)}


@app.get("/api/results")
def results(request: Request):
    res = _session_for(_sid(request))["results"]
    if res is None:
        return JSONResponse({"error": "no results yet"}, status_code=404)
    return res


@app.get("/api/download/{synth}/{table}")
def download(synth: str, table: str, request: Request):
    suite = _session_for(_sid(request)).get("suite") or {}
    if synth not in suite or table not in suite[synth]:
        return JSONResponse({"error": "not found"}, status_code=404)
    buf = io.StringIO()
    suite[synth][table].to_csv(buf, index=False)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=synthetic_{synth}_{table}.csv"})

