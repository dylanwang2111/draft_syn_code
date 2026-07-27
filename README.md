# Synth/Lab

A synthetic data studio: upload one or more related tables, generate
synthetic versions with several SDV synthesizers side by side, and get a
report on whether the output actually looks real, is safe to share, and is
useful for downstream work, before you trust it.

## What's here

- **`backend/`** + **`web/`** — the dashboard: a FastAPI backend and a
  vanilla-JS/HTML frontend (`uvicorn backend.server:app`). Upload CSVs,
  confirm the schema and any table relationships, pick synthesizers, run,
  and get a report with a Business (plain-language) and Technical (full
  metrics) view. Inside `backend/`: `server.py` (the thin HTTP routes),
  `dashboard_core.py` (session state, upload/detection helpers, the
  synthesis+evaluation worker), `chat_assistant.py` (see below), and
  `profiler.py` — structural profiling: detects table relationships,
  versioned/SCD data, and recommends a synthesis strategy without ever
  looking at cell values.
- **`synth_eval/`** — the evaluation suite: fidelity, privacy (membership
  inference, copy-check, attribute inference), and ML-efficacy metrics, plus
  referential-integrity scoring and the foreign-key relinking described
  below. See `METRICS.md` for the full metric-by-metric writeup. Shared by
  the dashboard and the standalone notebook (`notebook/synthetic_evaluation.ipynb`).
- **Chat assistant** (`web/chat.js`, `backend/chat_assistant.py`) — an
  LLM-driven conversational layer docked alongside the dashboard, a
  proof-of-concept for embedding this capability into a chatbot. It's not
  a separate app: it drives the exact same upload/synthesize/report
  machinery the manual controls do, in the same process, so a run started
  by chatting shows up in the same progress bar and the same report. Works
  with your own Azure AI Foundry / Azure OpenAI deployment, a direct OpenAI
  API key, or DeepSeek, picked automatically from whichever `.env` vars are
  set (see `.env.example`). Optional and fully decoupled: without one
  configured, the rest of the dashboard works exactly as before, the chat
  panel just says so.

## Synthesizers

Five SDV synthesizers. If a relationship is declared, all five preserve it,
just by two different mechanisms:

| Tier | Synthesizer(s) | Notes |
|---|---|---|
| Joint | HMA | Fits every table together; relationships are preserved by construction, including cross-table correlations. |
| Fit-then-linked | GaussianCopula, CTGAN, TVAE, CopulaGAN | Each fits every table independently, then has its foreign keys relinked to real synthetic parent rows afterward (`synth_eval/link.py`), so referential integrity holds too, just without HMA's joint cross-table modelling. |

Not shown as separate UI tiers (the dashboard just lists all five together),
the distinction above is real but mostly invisible day-to-day: pick whichever
synthesizer suits your data, referential integrity is handled either way as
long as a relationship or entity key is declared.

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python -m uvicorn backend.server:app --port 8000
```

Then open http://localhost:8000. Load the sample data or upload your own CSVs.

To enable the chat assistant too, copy `.env.example` to `.env` and fill in
one provider: `DEEPSEEK_API_KEY` (from platform.deepseek.com), `OPENAI_API_KEY`
(a direct key from platform.openai.com), or the `AZURE_OPENAI_*` variables for
your own Azure AI Foundry deployment (if more than one is set, priority is
Azure, then OpenAI, then DeepSeek), then restart the server.
