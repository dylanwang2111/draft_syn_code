"""LLM-driven conversational layer over the dashboard's existing
upload/synthesize/report machinery in dashboard_core.py / server.py. Optional
and fully decoupled: without an LLM provider configured, the endpoints below
just report they can't reach a language model, the rest of the dashboard is
unaffected.

Two providers are supported, auto-selected from whichever env vars are set
(see .env.example): your own Azure AI Foundry / Azure OpenAI deployment, or
DeepSeek. Azure wins if both are configured. Both speak the same OpenAI
chat-completions + tool-calling shape, so nothing below this block needs to
know or care which one is actually in use.

Mounted by server.py via:
    import chat_assistant
    app.include_router(chat_assistant.router)
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from openai import AzureOpenAI, OpenAI

from .dashboard_core import _session_for, _sid, _start_job, _tables_payload, _validate_relationships

load_dotenv()  # own copy: correct standalone regardless of server.py's import order

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")          # e.g. https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")      # the deployment name you chose in Foundry, e.g. "gpt-4.1"
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT:
    llm_client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT,
                             api_version=AZURE_OPENAI_API_VERSION)
    LLM_MODEL = AZURE_OPENAI_DEPLOYMENT   # Azure's `model=` argument is the deployment name, not "gpt-4.1" itself
elif DEEPSEEK_API_KEY:
    llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    LLM_MODEL = DEEPSEEK_MODEL
else:
    llm_client, LLM_MODEL = None, None

router = APIRouter()

CHAT_SYNTHS = ["HMA", "GaussianCopula", "CTGAN", "TVAE", "CopulaGAN"]
#: fixed best-practice holdout fraction (the manual UI no longer exposes this
#: as a tunable knob either, see web/app.js's HOLDOUT_FRAC) -- keep both in sync
CHAT_HOLDOUT_FRAC = 0.25
#: must match web/app.js's SDTYPES exactly (the Schema tab's dropdown options)
CHAT_SDTYPES = ["categorical", "numerical", "datetime", "boolean", "id", "unknown"]

CHAT_SYSTEM_PROMPT = """You are the Synth/Lab assistant: a friendly, plain-spoken helper that generates \
synthetic, privacy-safe versions of a user's tabular data, right inside the same dashboard the user is \
looking at. You are talking to someone who may not know anything about synthetic data generation, so \
avoid jargon and keep replies short (2-4 sentences), like real chat messages, not a report.

You have five synthesizers available, only mention them by name if asked or when explaining your pick. \
All five keep a declared relationship's referential integrity: HMA by fitting every table jointly, the \
other four by fitting tables independently and then relinking foreign keys to real synthetic parent \
rows afterward (so links hold, but unlike HMA they don't model correlations across tables).
- HMA: fits every table jointly. Use this when the tables are genuinely linked (a shared key), it's \
the one that actually models how tables relate to each other, not just patches up the keys after.
- GaussianCopula: a fast statistical model, fit per table independently. Good default in general, \
especially for a single table or a quick first look.
- CTGAN, TVAE, CopulaGAN: neural models, fit per table independently, slower (they actually train, so \
epochs matter, more epochs is slower but can fit better). Only suggest these if the user asks for \
something slower/fancier, or mentions a column with complex/skewed patterns, they're not needed by \
default. If the user gives a specific epoch count, pass it through via run_synthesis's epochs \
argument, otherwise leave it unset.

If the user asks what a synthesizer IS, how it works, or how two of them compare, call \
explain_synthesizer instead of writing your own technical explanation from memory, the dashboard \
already has the real step-by-step mechanism for each one written out and illustrated, no need to \
reinvent it in the chat bubble. Still say a quick sentence yourself too so the chat doesn't feel empty, \
just don't duplicate the detailed write-up.

After a file upload, you'll be given a structural analysis (table names, row counts, columns per \
table, whether a reliable link between tables was found, candidate shared columns if any were \
spotted structurally, a suggested synthesizer, and any columns that look like personal info). That \
suggested synthesizer is for your own reasoning only, never say it, hint at it, or say anything like \
"my recommendation is" or "I'd suggest" out loud yet, you're missing two things you need to ask about \
first, in order, and naming a synthesizer before then undercuts the whole point of asking. \
IMPORTANT: for both of those things you MUST actually call the ask_question tool, a real function \
call, not just phrase a question as plain text, plain text does not render as clickable buttons and \
leaves the user guessing what answers are even valid. This is true every single time you need a \
decision anywhere in the conversation, not just here.

STEP 1 - schema. Summarize the tables in plain language (names, row counts, mention you'll auto-fake \
any personal info found), then call ask_question with something like "Want to review the auto-detected \
column types before we go further?" and options like ["Looks good", "Let me edit it", "I'll describe changes"].
- "Looks good" / equivalent -> call confirm_schema(modified=false).
- "Let me edit it" / equivalent -> call open_schema_editor(), tell them it's open and to save when \
done. Once they say they're done (their save button sends a short confirmation automatically) call \
confirm_schema(modified=true).
- They describe specific column changes in words instead -> call set_column_types with what they \
described, confirm what you changed, ask if there's more, and once they're satisfied call \
confirm_schema(modified=true).
Do not move to step 2 before confirm_schema has been called.

STEP 2 - relationships. Only after schema is confirmed, and ALWAYS ask this via ask_question first, \
e.g. "Are these tables related to each other?" with options like ["No, independent", "Let me set it \
up", "I'll describe it"]. Tables are often related even when it can't be auto-detected (e.g. a \
customer ID that shows up in several tables but isn't unique in any single one), if candidate shared \
columns were spotted structurally, mention them here as a hint, worth asking about. But a shared \
column name is only ever a HINT for you to bring up, never treat it as the user's answer. Do not call \
set_entity_key, set_relationship, or confirm_relationships based on your own guess about candidate \
shared columns, only the user gets to decide there's a real relationship, and only after you've \
actually asked via ask_question and they've replied in THIS conversation.
- "No" / equivalent -> call confirm_relationships(has_relationships=false).
- "Let me set it up" / equivalent -> call open_data_model(), tell them it's open (drag-and-drop canvas \
or an entity-key hub builder) and to save when done. Once they confirm, call \
confirm_relationships(has_relationships=true).
- They describe it in words instead -> there are two kinds: an entity key (one column name present in \
several tables tying rows to the same real-world entity, rows don't need to be 1:1, call \
set_entity_key) or a formal relationship (one table's column is a unique parent key, another table's \
column points at it, call set_relationship). Apply it, then call \
confirm_relationships(has_relationships=true).
Do not move to step 3 before confirm_relationships has been called. Once a link is set, HMA becomes \
the better pick (it's the one that actually models links), fold that into your step 3 recommendation.

STEP 3 - only now, with both confirmed, give your synthesizer recommendation and a one-line reason. \
Ask via ask_question whether they want just your recommendation, or want to run a couple of \
synthesizers together to compare from the start (this is a real choice to offer, not a footnote), \
options like ["Go with your pick", "Compare a couple"].
- "Go with your pick" / equivalent -> call run_synthesis with just your recommended synthesizer.
- "Compare a couple" / equivalent, or they want to choose themselves -> call \
open_config_panel(panel="synthesizers"), this brings the chip picker (and run parameters below it) \
into the middle of the screen so they can multi-select there (ask_question's buttons are \
one-choice-at-a-time, not built for picking several at once). Tell them it's open and to click \
whichever ones they want, then say go. Once they say go, call run_synthesis with the synthesizer(s) \
they named, or omit the argument entirely to use whatever they clicked.

open_config_panel also covers "constraints", "pii", and "run_parameters" the same way, whenever a \
config choice needs more than a one-tap answer, e.g. they want to set a per-column value rule or \
change which columns get faked. Prefer ask_question for anything that's really just picking between a \
few named options.

Call the run_synthesis tool once the user agrees to proceed, with whichever synthesizer(s) they \
actually named (a list; more than one runs a real side-by-side comparison in one report). If they \
just say yes/go ahead without naming any, omit the synthesizers argument entirely rather than \
guessing, it automatically falls back to whatever's currently selected in the Synthesizers panel \
(which is your own recommendation unless they've clicked different chips themselves). After you call \
it, tell the user you're starting; a progress bar and the Synthesizers panel are already visible next \
to the chat and will show exactly what's running, so don't promise to report back with percentages \
yourself, just say it'll take a little while.

When you're later given the results of a run, the full detailed report has already opened in the \
dashboard next to you. Business language by default: lead with each synthesizer's business_summary \
(that's the exact plain-language line already printed on the report's exec summary card, e.g. "This \
synthetic data behaves like your real data, carries no privacy red flags, and is about 90% as useful \
as real data for analytics. Recommended for dev/test environments, vendor sharing, and model \
training."). Don't cite the raw decimal scores (fidelity_score_0_to_1 etc.) or say things like "0.85" \
unless the user specifically asks to go technical, they're there for that, not for the default reply. \
If more than one synthesizer ran, weave their business_summary lines into one short comparison (which \
one you'd actually go with and the one-line reason why), don't just paste both verbatim back to back. \
Be honest if something looks weak, use_cases has the per-scenario reasons (dev/test, vendor sharing, \
training) if a specific one needs flagging. If they ask for the technical view, THEN use the actual \
scores/verdicts to explain precisely.

After giving that answer, call ask_question with concrete next-step options, don't just leave it open \
("anything else?"), options like ["Explain a detail", "Download the data", "Adjust settings", "Try \
another synthesizer"].
- "Explain a detail" -> ask what, then explain using the real scores/verdicts.
- "Download the data" -> call open_config_panel(panel="synthetic_data"), that's where the per-table \
download links live.
- "Adjust settings" -> ask (or just call open_config_panel directly if they already said which) which \
one: constraints, PII handling, synthesizer choice, or run parameters, then open the matching panel.
- "Try another synthesizer" -> same as the compare flow in step 3, call open_config_panel(panel=\
"synthesizers") so they can pick, or run_synthesis directly if they name one.

Avoid markdown formatting where you can, this renders as a plain chat bubble, short plain sentences \
read better than headers or lists. Don't use em dashes, use commas or periods instead. Never refer to \
any specific company or bank by name."""

#: every tool the LLM can ever call, keyed by name. Which ones it actually
#: SEES on a given turn is decided by _tools_for(plan), not this dict, that's
#: the hard gate: run_synthesis isn't offered until schema_confirmed and
#: relationships_confirmed are both true, so the model can't skip ahead even
#: if it wanted to (the system prompt's step ordering is the soft version of
#: the same rule, this is the structural one).
_TOOL_DEFS = {
    "ask_question": {
        "type": "function",
        "function": {
            "name": "ask_question",
            "description": "Ask the user a specific question with a short list of clear-cut answer "
                           "options, whenever you need an explicit decision rather than guessing or "
                           "assuming. The options render as clickable buttons in the chat, but the "
                           "user can also just type a free-text reply instead of clicking, both work, "
                           "you don't need to handle that differently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question, as you'd say it in chat."},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4,
                               "description": "2-4 short button labels, e.g. "
                               "['Looks good', 'Let me edit it', \"I'll describe changes\"]."},
                },
                "required": ["question", "options"],
            },
        },
    },
    "open_schema_editor": {
        "type": "function",
        "function": {
            "name": "open_schema_editor",
            "description": "Open the Schema tab, visibly emphasized over the rest of the dashboard, so "
                           "the user can review/edit auto-detected column types and primary keys "
                           "themselves. Call this when they say they want to edit rather than describe "
                           "changes in words. You'll be told once they've saved.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "open_data_model": {
        "type": "function",
        "function": {
            "name": "open_data_model",
            "description": "Open the Data Model tab, visibly emphasized over the rest of the "
                           "dashboard, so the user can draw a table relationship or build an "
                           "entity-key hub themselves on its drag-and-drop canvas. Call this when they "
                           "say they want to set it up rather than describe it in words. You'll be "
                           "told once they've saved.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "set_column_types": {
        "type": "function",
        "function": {
            "name": "set_column_types",
            "description": "Apply column type changes the user described in words instead of using "
                           "the Schema tab, e.g. 'make STATUS categorical'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "changes": {
                        "type": "array", "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "table": {"type": "string"},
                                "column": {"type": "string"},
                                "sdtype": {"type": "string", "enum": CHAT_SDTYPES},
                            },
                            "required": ["table", "column", "sdtype"],
                        },
                    },
                },
                "required": ["changes"],
            },
        },
    },
    "confirm_schema": {
        "type": "function",
        "function": {
            "name": "confirm_schema",
            "description": "Call once the schema question is settled: either the user said it looks "
                           "fine as-is, or edits were made/described and they're happy with them. This "
                           "unlocks the relationships question, don't call it before actually asking.",
            "parameters": {
                "type": "object",
                "properties": {"modified": {"type": "boolean",
                               "description": "Whether anything was actually changed."}},
                "required": ["modified"],
            },
        },
    },
    "confirm_relationships": {
        "type": "function",
        "function": {
            "name": "confirm_relationships",
            "description": "Call once the relationships question is settled: a link was set up, or "
                           "the user confirmed the tables are independent. This unlocks giving a "
                           "synthesizer recommendation and run_synthesis, don't call it before "
                           "actually asking.",
            "parameters": {
                "type": "object",
                "properties": {"has_relationships": {"type": "boolean"}},
                "required": ["has_relationships"],
            },
        },
    },
    "set_entity_key": {
        "type": "function",
        "function": {
            "name": "set_entity_key",
            "description": "Declare a shared business key: one column name present in several tables "
                           "that ties their rows to the same real-world entity, even though it isn't a "
                           "unique key in any single one of them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "The shared column name."},
                    "tables": {"type": "array", "items": {"type": "string"},
                              "description": "Which tables to link on this column."},
                },
                "required": ["column", "tables"],
            },
        },
    },
    "set_relationship": {
        "type": "function",
        "function": {
            "name": "set_relationship",
            "description": "Declare a formal foreign-key relationship between two tables: the parent "
                           "column must be unique, the child column points at it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_table": {"type": "string"},
                    "parent_key": {"type": "string", "description": "Unique key column in the parent table."},
                    "child_table": {"type": "string"},
                    "child_key": {"type": "string", "description": "Foreign-key column in the child table."},
                },
                "required": ["parent_table", "parent_key", "child_table", "child_key"],
            },
        },
    },
    "run_synthesis": {
        "type": "function",
        "function": {
            "name": "run_synthesis",
            "description": "Start generating synthetic data. Call this once the user has agreed to "
                           "proceed. Pass more than one synthesizer to run a side-by-side comparison "
                           "in a single report. Omit synthesizers entirely if the user didn't name any "
                           "(e.g. just said 'go ahead'), that means 'use whatever's currently selected "
                           "in the Synthesizers panel'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "synthesizers": {"type": "array", "items": {"type": "string", "enum": CHAT_SYNTHS},
                                     "minItems": 1, "description": "Which synthesizer(s) to run. Omit "
                                     "to use the current chip selection instead."},
                    "epochs": {"type": "integer", "description": "Training epochs for CTGAN/TVAE/"
                              "CopulaGAN if the user asked for a specific number. Ignored by HMA/"
                              "GaussianCopula (they don't train). Defaults to 100 if not given."},
                },
                "required": [],
            },
        },
    },
    "open_config_panel": {
        "type": "function",
        "function": {
            "name": "open_config_panel",
            "description": "Bring one of the dashboard's config panels into the middle of the screen, "
                           "highlighted, so the user can set it up directly instead of describing it in "
                           "words. Use this any time a config choice needs more than a one-tap answer, "
                           "ask_question's buttons are one-choice-at-a-time, not built for a multi-select "
                           "or fiddly control like a chip picker, a slider, or a per-column table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "panel": {"type": "string", "enum": ["structure", "constraints", "pii",
                              "synthesizers", "run_parameters", "synthetic_data"], "description":
                              "structure: entity-key/relationship shortcuts (usually prefer the Data "
                              "Model tab instead). constraints: per-column value rules. pii: which "
                              "columns get faked and how. synthesizers: the multi-select chip picker, "
                              "opens alongside run_parameters. run_parameters: epochs/scale/etc. "
                              "synthetic_data: the finished per-table download links, once a run has "
                              "completed."},
                },
                "required": ["panel"],
            },
        },
    },
    "explain_synthesizer": {
        "type": "function",
        "function": {
            "name": "explain_synthesizer",
            "description": "Pop open the dashboard's own \"How the synthesizers work\" explainer "
                           "instead of writing your own explanation from scratch, whenever the user "
                           "asks what a synthesizer is, how it works, or how it compares to another. "
                           "It already has the real step-by-step mechanism for each one, illustrated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "synth": {"type": "string", "enum": ["HMA", "GaussianCopula", "CTGAN", "TVAE",
                              "CopulaGAN", "all"], "description": "Which one to scroll to and "
                              "highlight, or 'all' to just open the explainer without singling one "
                              "out (e.g. they asked about more than one, or asked generally)."},
                },
                "required": ["synth"],
            },
        },
    },
}

_CONFIG_PANELS = {"structure", "constraints", "pii", "synthesizers", "run_parameters", "synthetic_data"}


def _tools_for(plan: dict) -> list[dict]:
    names = ["ask_question", "open_schema_editor", "set_column_types", "confirm_schema",
             "open_config_panel", "explain_synthesizer"]
    if plan.get("schema_confirmed"):
        names += ["open_data_model", "set_entity_key", "set_relationship", "confirm_relationships"]
    if plan.get("schema_confirmed") and plan.get("relationships_confirmed"):
        names.append("run_synthesis")
    return [_TOOL_DEFS[n] for n in names]


def _chat_plan_from_profile(tables: dict, profile: dict) -> dict:
    """Structural facts for the LLM to reason from, the same signal the
    dashboard's advisor card is built from (profiler.py). NOT chat text,
    just data; the LLM writes what the user sees."""
    rec = (profile or {}).get("recommendation") or {}
    rels = rec.get("relationships") or []
    n_tables = len(tables)
    if rec.get("tier") == 1 and rels:
        synth = "HMA"
    else:
        synth = "GaussianCopula"
        rels = []

    pii_cols = [f"{t}.{col}" for t, info in tables.items() for col in (info.get("pii") or {})]
    primary_keys = {t: info.get("primary_key") for t, info in tables.items() if info.get("primary_key")}
    table_cols = {t: [c["name"] for c in info.get("columns", [])] for t, info in tables.items()}

    col_counts: dict[str, int] = {}
    for cols in table_cols.values():
        for c in set(cols):
            col_counts[c] = col_counts.get(c, 0) + 1
    candidate_shared = sorted(c for c, n in col_counts.items() if n >= 2)

    return {
        "tables": {t: {"rows": info["rows"], "columns": table_cols[t]} for t, info in tables.items()},
        "n_tables": n_tables,
        "relationships_found": rels,
        "candidate_shared_columns": candidate_shared,
        "suggested_synthesizer": synth,
        "pii_columns_detected": pii_cols,
        "synth": synth, "relationships": rels, "primary_keys": primary_keys,
        "entity_key": "", "entity_children": [],
        "selected_synths": [], "epochs": 100,
        "schema": {}, "schema_confirmed": False, "relationships_confirmed": False,
    }


def _chat_run_synthesis(st: dict, synthesizers: list[str], epochs=None) -> dict:
    plan = st.get("chat_plan")
    if not plan:
        return {"error": "no data has been uploaded yet"}
    # nothing named explicitly ("go ahead") -> whatever's currently selected
    # in the Synthesizers chips, then the bot's own single recommendation
    synthesizers = [s for s in (synthesizers or []) if s in CHAT_SYNTHS] \
        or plan.get("selected_synths") or [plan["synth"]]
    try:
        epochs = max(1, min(2000, int(epochs)))
    except (TypeError, ValueError):
        epochs = plan.get("epochs") or 100
    # start from the auto-detected primary keys, then layer on whatever the
    # user actually edited (via the Schema tab, set_column_types, or a
    # focus-mode save) so an untouched table still gets its detected pk
    schema = {t: {"primary_key": pk} for t, pk in plan.get("primary_keys", {}).items()}
    for t, edits in (plan.get("schema") or {}).items():
        entry = schema.setdefault(t, {"primary_key": plan.get("primary_keys", {}).get(t)})
        if edits.get("primary_key"):
            entry["primary_key"] = edits["primary_key"]
        if edits.get("sdtypes"):
            entry["sdtypes"] = edits["sdtypes"]
    cfg = {
        "schema": schema,
        "relationships": plan.get("relationships") or [],
        "entity_key": plan.get("entity_key") or "",
        "entity_children": plan.get("entity_children") or [],
        "synths": synthesizers, "scale": 1.0, "epochs": epochs, "holdout": CHAT_HOLDOUT_FRAC, "seed": 42,
    }
    result = _start_job(cfg, st)
    if result.get("error"):
        return result
    plan["selected_synths"], plan["epochs"] = synthesizers, epochs
    return {"status": "started", "synthesizers": synthesizers, "epochs": epochs}


def _chat_set_entity_key(st: dict, column: str, tables: list[str]) -> dict:
    plan = st.get("chat_plan")
    if not plan:
        return {"error": "no data has been uploaded yet"}
    column = (column or "").strip()
    tables = [t for t in (tables or []) if t in plan["tables"]]
    if not column or len(tables) < 2:
        return {"error": "need a column name and at least two real table names"}
    missing = [t for t in tables if column not in plan["tables"][t]["columns"]]
    if missing:
        return {"error": f"'{column}' isn't a column in: {', '.join(missing)}"}
    results = _validate_relationships(st["tables"], {"entity_key": column, "entity_children": tables})
    if any(r.get("status") == "FAIL" for r in results):
        return {"error": "that didn't check out against the real data", "details": results}
    plan["entity_key"], plan["entity_children"], plan["relationships"] = column, tables, []
    plan["synth"] = "HMA"
    return {"status": "linked", "kind": "entity_key", "column": column, "tables": tables,
            "validation": results}


def _chat_set_relationship(st: dict, parent_table: str, parent_key: str,
                           child_table: str, child_key: str) -> dict:
    plan = st.get("chat_plan")
    if not plan:
        return {"error": "no data has been uploaded yet"}
    if parent_table not in plan["tables"] or child_table not in plan["tables"]:
        return {"error": "unknown table name"}
    if parent_key not in plan["tables"][parent_table]["columns"]:
        return {"error": f"'{parent_key}' isn't a column in {parent_table}"}
    if child_key not in plan["tables"][child_table]["columns"]:
        return {"error": f"'{child_key}' isn't a column in {child_table}"}
    rel = {"parent_table_name": parent_table, "parent_primary_key": parent_key,
           "child_table_name": child_table, "child_foreign_key": child_key}
    results = _validate_relationships(st["tables"], {"relationships": [rel]})
    if any(r.get("status") == "FAIL" for r in results):
        return {"error": "that didn't check out against the real data", "details": results}
    plan["relationships"] = (plan.get("relationships") or []) + [rel]
    plan["entity_key"], plan["entity_children"] = "", []
    plan["synth"] = "HMA"
    return {"status": "linked", "kind": "relationship", "relationship": rel, "validation": results}


def _chat_set_column_types(st: dict, changes: list[dict]) -> dict:
    plan = st.get("chat_plan")
    if not plan:
        return {"error": "no data has been uploaded yet"}
    applied = []
    for ch in changes or []:
        table, column, sdtype = ch.get("table", ""), ch.get("column", ""), ch.get("sdtype", "")
        if table not in plan["tables"]:
            return {"error": f"unknown table '{table}'"}
        if column not in plan["tables"][table]["columns"]:
            return {"error": f"'{column}' isn't a column in {table}"}
        if sdtype not in CHAT_SDTYPES:
            return {"error": f"'{sdtype}' isn't a valid column type"}
        plan.setdefault("schema", {}).setdefault(table, {"sdtypes": {}, "primary_key": None})
        plan["schema"][table].setdefault("sdtypes", {})[column] = sdtype
        applied.append(f"{table}.{column} -> {sdtype}")
    if not applied:
        return {"error": "no changes given"}
    return {"status": "applied", "changes": applied}


def _chat_confirm_schema(st: dict, modified: bool) -> dict:
    plan = st.get("chat_plan")
    if not plan:
        return {"error": "no data has been uploaded yet"}
    plan["schema_confirmed"] = True
    return {"status": "confirmed", "modified": bool(modified)}


def _chat_confirm_relationships(st: dict, has_relationships: bool) -> dict:
    plan = st.get("chat_plan")
    if not plan:
        return {"error": "no data has been uploaded yet"}
    plan["relationships_confirmed"] = True
    return {"status": "confirmed", "has_relationships": bool(has_relationships)}


#: fallback (question, options) for each unconfirmed step -- used when the
#: model answers in plain prose instead of actually calling ask_question
#: (observed in testing: DeepSeek's tool_choice doesn't support "required"
#: in thinking mode, so this can't be forced API-side; the prompt asks for
#: it but LLM instruction-following isn't reliable enough to trust alone).
#: Since the model's own prose already asks roughly the right thing (the
#: system prompt walks it through the steps), we don't discard it, we just
#: guarantee the buttons for the CURRENT pending step exist regardless --
#: and if that prose doesn't actually end on a question (observed too: a
#: plain summary with no ask at all), append the real question so the
#: buttons don't show up looking orphaned/unexplained.
_STEP_FALLBACK_OPTIONS = {
    "schema": ["Looks good", "Let me edit it", "I'll describe changes"],
    "relationships": ["No, independent", "Let me set it up", "I'll describe it"],
}
_STEP_FALLBACK_QUESTIONS = {
    "schema": "Want to review the auto-detected column types before we go further?",
    "relationships": "Are these tables related to each other?",
}

#: some providers (observed: Azure/GPT-4.1) copy the system prompt's own
#: example phrasing -- 'options like ["A", "B"]' -- verbatim into the chat
#: reply's own text instead of actually calling ask_question, leaving a raw
#: bracketed list visible in the bubble with no real buttons behind it
#: (DeepSeek doesn't do this, it reliably calls the tool instead). Recover
#: structurally: if the tail of the text has exactly that shape, treat the
#: quoted strings as the options the model meant to offer, strip the
#: literal list out of what's displayed, and use them as real buttons.
_LEAKED_OPTIONS_RE = re.compile(r'\[\s*"[^"\]]{1,60}"(?:\s*,\s*"[^"\]]{1,60}"){1,3}\s*\]\s*\.?\s*$')


def _extract_leaked_options(text: str) -> tuple[str, list[str]]:
    m = _LEAKED_OPTIONS_RE.search(text)
    if not m:
        return text, []
    return text[:m.start()].rstrip(), re.findall(r'"([^"]+)"', m.group(0))[:4]


def _recent_history(messages: list[dict], limit: int = 24) -> list[dict]:
    """Last `limit` messages, but never starting mid-tool-exchange: a plain
    suffix slice can land on a 'tool' result whose owning assistant
    message (the one with tool_calls) fell just outside the window, and
    the API rejects a 'tool' message with no preceding tool_calls to
    answer. Drop any such orphaned leading tool messages instead."""
    window = messages[-limit:]
    i = 0
    while i < len(window) and window[i].get("role") == "tool":
        i += 1
    return window[i:]


def _chat_turn(st: dict, force_tool: bool = True, tools: list[dict] | None = None):
    """One assistant turn: call the LLM with the current history (offering
    only the tools _tools_for(plan) currently allows, the hard gate on
    run_synthesis lives there), execute any tool call it makes, and return
    (text, options, focus) for the route to hand to the frontend.

    force_tool: True on the first call reacting to fresh user/system input;
    False on the recursive call one level down (after a tool result has
    been appended and the model is just narrating/waiting for the user's
    actual reply, not to be pushed into another decision in the same
    breath). Only a force_tool=True turn is eligible for the fallback
    options below, an inner turn skipping ask_question is normal, not a
    compliance failure.

    tools: computed ONCE from plan at the top of a fresh request (None here)
    and threaded through every recursive call unchanged. Without this, a
    confirm_schema call flipping schema_confirmed mid-request would
    immediately unlock set_relationship/confirm_relationships for the very
    next (recursive, same-exchange) call, letting the model both ask AND
    unilaterally decide+confirm the relationships step in one breath before
    the user has said anything about it (observed in testing: it inferred a
    relationship from shared column names alone). Freezing the tool list
    for the life of one request means a just-unlocked step's mutating
    tools only become callable on the NEXT request, i.e. after the user
    has actually seen and answered that step's question -- ask_question
    itself stays available throughout, so the model can still ask the next
    question in the same breath, it just can't also answer it itself."""
    plan = st.get("chat_plan") or {}
    if llm_client is None:
        return ("I can't reach my language model right now (no LLM provider is configured on the "
                "server -- set DEEPSEEK_API_KEY, or the AZURE_OPENAI_* variables for your own "
                "deployment). Ask whoever's running this to set it and restart."), [], None
    if tools is None:
        tools = _tools_for(plan)
    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + _recent_history(st["chat_messages"]),
            tools=tools, tool_choice="auto", temperature=0.4,
        )
    except Exception as e:
        return f"My language model call failed: {e}", [], None

    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    st["chat_messages"].append({"role": "assistant", "content": msg.content or "",
                                "tool_calls": [c.model_dump() for c in calls] or None})

    if not calls:
        # the system prompt asks for no em dashes, but LLM style compliance
        # isn't guaranteed, so enforce it deterministically too
        text = re.sub(r"\s+,", ",", (msg.content or "…").replace("—", ","))
        text, options = _extract_leaked_options(text)
        if force_tool and not options:
            step = "schema" if not plan.get("schema_confirmed") \
                else "relationships" if not plan.get("relationships_confirmed") else None
            if step:
                options = _STEP_FALLBACK_OPTIONS[step]
                if "?" not in text[-80:]:
                    text = text.rstrip() + " " + _STEP_FALLBACK_QUESTIONS[step]
        return text, options, None

    # the hard gate lives here, not just in what tools= we hand the API:
    # nothing stops a model from EMITTING a tool_call for a function name
    # it isn't currently offered (most providers don't validate that server
    # side, and the system prompt mentions run_synthesis/confirm_* by name
    # throughout regardless of the current step, so the model can and did,
    # in testing, hallucinate a call to a tool it was never given this
    # turn). Refuse anything whose name isn't in THIS turn's frozen `tools`
    # before it ever reaches a handler -- dispatching by name match alone,
    # as this used to, would silently honor a call the gate meant to block.
    allowed_names = {t["function"]["name"] for t in tools}
    options, focus, question_like, asked_question = [], None, False, ""
    for c in calls:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        name = c.function.name
        if name not in allowed_names:
            result = {"error": f"'{name}' isn't available yet, follow the step order in your "
                               "instructions instead"}
        elif name == "run_synthesis":
            result = _chat_run_synthesis(st, args.get("synthesizers") or [], args.get("epochs"))
        elif name == "set_entity_key":
            result = _chat_set_entity_key(st, args.get("column", ""), args.get("tables") or [])
        elif name == "set_relationship":
            result = _chat_set_relationship(st, args.get("parent_table", ""), args.get("parent_key", ""),
                                            args.get("child_table", ""), args.get("child_key", ""))
        elif name == "set_column_types":
            result = _chat_set_column_types(st, args.get("changes") or [])
        elif name == "confirm_schema":
            result = _chat_confirm_schema(st, bool(args.get("modified")))
        elif name == "confirm_relationships":
            result = _chat_confirm_relationships(st, bool(args.get("has_relationships")))
        elif name == "ask_question":
            options = [str(o) for o in (args.get("options") or [])][:4]
            asked_question = str(args.get("question") or "").strip()
            result = {"status": "asked"}
            question_like = True
        elif name == "open_schema_editor":
            focus = "schema"
            result = {"status": "opened", "view": "schema"}
            question_like = True
        elif name == "open_data_model":
            focus = "data_model"
            result = {"status": "opened", "view": "data_model"}
            question_like = True
        elif name == "open_config_panel":
            panel = args.get("panel", "")
            if panel not in _CONFIG_PANELS:
                result = {"error": f"'{panel}' isn't a real config panel"}
            else:
                focus = panel
                result = {"status": "opened", "view": panel}
                question_like = True
        elif name == "explain_synthesizer":
            synth = args.get("synth", "all")
            if synth not in CHAT_SYNTHS and synth != "all":
                result = {"error": f"'{synth}' isn't a real synthesizer"}
            else:
                focus = f"docs:{synth}"
                result = {"status": "opened", "synth": synth}
                question_like = True
        else:
            result = {"error": "unknown tool"}
        st["chat_messages"].append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})

    # let the model react to the tool result(s), NOT forced this time (it
    # just acted, it may need to just narrate and wait for the user's
    # actual reply rather than being pushed into another tool call in the
    # same breath); SAME frozen tools list too, a step just confirmed this
    # request can be asked about right away (ask_question is always in the
    # list) but can't also be unilaterally answered until next request; if
    # THAT turn asks a question or opens a view too (e.g. right after
    # confirm_schema unlocks step 2, it may on its own initiative
    # immediately ask the relationships question), that's the more current
    # signal, prefer it over this level's
    text, inner_options, inner_focus = _chat_turn(st, force_tool=False, tools=tools)
    if question_like:
        # ask_question/open_*_editor calls are asked ALONGSIDE their own
        # rich lead-in text (the table summary, why we're asking, etc.) --
        # the next turn only sees a bare {"status": "asked"} tool result,
        # so its own reply tends to be a thin "go ahead and choose" filler.
        # Prefer this level's own content when the model actually wrote one.
        own_text = (msg.content or "").strip()
        if own_text:
            text = re.sub(r"\s+,", ",", own_text.replace("—", ","))
    # own_text (question_like) or the recursive call's own text can carry a
    # leaked bracket list the same way the no-tool-call branch can -- catch
    # it here too, and use it to fill in options if the tool call itself
    # didn't provide any (e.g. the model wrote ask_question-shaped text
    # without actually calling the tool this turn).
    text, leaked = _extract_leaked_options(text)
    if question_like:
        # the model's own text (own_text or the fallback above) frequently
        # summarizes WITHOUT ever actually asking anything -- the options
        # it chose for ask_question's buttons never get shown as a
        # question on their own otherwise (only the button labels do), so
        # the buttons look orphaned: "here's your data" ... [Looks good]
        # [Let me edit it], no explanation of what would be edited. Make
        # sure the literal question the model wrote is actually present.
        if asked_question and asked_question not in text:
            text = text.rstrip()
            if text and not text.endswith((".", "!", "?")):
                text += "."
            text = (text + " " + asked_question).strip() if text else asked_question
    return text, (inner_options or options or leaked), (inner_focus or focus)


@router.post("/api/chat/plan")
def chat_plan(request: Request):
    st = _session_for(_sid(request))
    if st["tables"] is None:
        return JSONResponse({"error": "upload data first"}, status_code=400)
    payload = _tables_payload(st)
    tables, profile = payload["tables"], payload.get("profile", {})

    plan = _chat_plan_from_profile(tables, profile)
    st["chat_plan"] = plan
    st["chat_messages"] = []

    analysis = {
        "tables": {t: {"rows": info["rows"], "columns": len(info["columns"])} for t, info in plan["tables"].items()},
        "relationships_found": plan["relationships_found"],
        "candidate_shared_columns": plan["candidate_shared_columns"],
        "suggested_synthesizer": plan["suggested_synthesizer"],
        "pii_columns_detected": plan["pii_columns_detected"],
    }
    st["chat_messages"].append({"role": "user", "content": f"I uploaded: {', '.join(tables)}"})
    st["chat_messages"].append({"role": "system",
                                "content": "Structural analysis of the upload: " + json.dumps(analysis)})
    message, options, focus = _chat_turn(st)
    return {"message": message, "options": options, "focus": focus, "tables": plan["tables"],
            "candidate_shared_columns": plan["candidate_shared_columns"],
            "has_link": bool(plan["relationships"])}


def _sync_ui_state(st: dict, body: dict) -> None:
    """Pick up whatever's currently live on the client -- a relationship/
    entity-key set visually in the Data Model tab, and/or the current
    Synthesizers chip selection + epochs field -- so a link drawn on the
    canvas, a chip clicked by hand, and one described in words to the bot
    all end up in the identical plan, no separate 'I'm done' step required.
    The frontend sends this with every /api/chat/message call."""
    plan = st.get("chat_plan")
    if not plan:
        return
    rels = body.get("relationships") or []
    entity_key = (body.get("entity_key") or "").strip()
    entity_children = body.get("entity_children") or []
    changed, desc = False, None
    if entity_key and (entity_key != plan.get("entity_key") or entity_children != plan.get("entity_children")):
        plan["entity_key"], plan["entity_children"], plan["relationships"] = entity_key, entity_children, []
        plan["synth"] = "HMA"
        changed, desc = True, {"kind": "entity_key", "column": entity_key, "tables": entity_children}
    elif not entity_key and rels and rels != plan.get("relationships"):
        plan["relationships"] = rels
        plan["entity_key"], plan["entity_children"] = "", []
        plan["synth"] = "HMA"
        changed, desc = True, {"kind": "relationship", "relationships": rels}
    elif not entity_key and not rels and (plan.get("entity_key") or plan.get("relationships")):
        # the user removed the link on the canvas (Remove hub / delete
        # relationship) -- the two branches above only ever fire on a NEW
        # link, so clearing one back to nothing needs its own check
        plan["entity_key"], plan["entity_children"], plan["relationships"] = "", [], []
        plan["synth"] = "GaussianCopula"
        changed, desc = True, {"kind": "cleared"}
    if changed:
        st["chat_messages"].append({"role": "system",
                                    "content": "The user just set this up visually in the Data Model "
                                    "tab (not by typing): " + json.dumps(desc) + ". Acknowledge it "
                                    "briefly next turn."})

    # synth/epochs/schema are just live UI state, not a decision worth
    # narrating -- sync silently so 'go ahead' with nothing named uses
    # what's selected, and the Schema tab's own edits are already visible
    # there, no need for the bot to repeat them back
    selected = [s for s in (body.get("selected_synths") or []) if s in CHAT_SYNTHS]
    if selected and selected != plan.get("selected_synths"):
        plan["selected_synths"] = selected
    try:
        epochs = int(body.get("epochs"))
        if epochs > 0:
            plan["epochs"] = epochs
    except (TypeError, ValueError):
        pass
    schema = body.get("schema") or {}
    if schema and schema != plan.get("schema"):
        plan["schema"] = schema


def _plan_sync_dict(plan: dict) -> dict:
    return {
        "relationships": plan.get("relationships") or [],
        "entity_key": plan.get("entity_key") or "",
        "entity_children": plan.get("entity_children") or [],
        "selected_synths": plan.get("selected_synths") or [],
        "epochs": plan.get("epochs") or 100,
        "schema": plan.get("schema") or {},
    }


@router.post("/api/chat/message")
async def chat_message(request: Request):
    st = _session_for(_sid(request))
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty message"}, status_code=400)
    plan = st.get("chat_plan")
    if plan is None:
        return JSONResponse({"error": "upload data first"}, status_code=400)
    _sync_ui_state(st, body)
    st["chat_messages"].append({"role": "user", "content": text})
    message, options, focus = _chat_turn(st)
    started = bool(st["job"] and st["job"]["status"] == "running")
    return {"message": message, "options": options, "focus": focus, "started": started,
            "sync": _plan_sync_dict(plan)}


@router.post("/api/chat/reset")
def chat_reset(request: Request):
    """Clear just the chat transcript (new data was likely uploaded/changed);
    tables/results are untouched."""
    st = _session_for(_sid(request))
    st["chat_messages"], st["chat_plan"] = [], None
    return {"status": "ok"}


@router.post("/api/chat/narrate_result")
async def chat_narrate_result(request: Request):
    """Called once the frontend's own poll() reaches 'done' and has already
    rendered the real report -- reads the SAME st['results'] that report was
    built from and asks the LLM for a short spoken summary + recommendation.

    The request body carries a `business` dict, one entry per synthesizer,
    precomputed client-side by the exact same bizDims/recommendation/
    bizUseCases functions (web/app.js) that build the report's own exec
    summary card -- so the chat's business-language framing is always the
    SAME text already on screen, not a separate LLM paraphrase of the raw
    scores that could drift from it. The raw scores are still included
    too, for when the user wants the technical numbers instead."""
    st = _session_for(_sid(request))
    res = st.get("results")
    if res is None:
        return JSONResponse({"error": "no results yet"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    business = body.get("business") or {}

    facts = []
    for s in res.get("synths") or []:
        summ = (res.get("summary") or {}).get(s) or {}
        worst = "ready"
        for tabs in ((res.get("privacy") or {}).get(s) or {}).values():
            for v in (tabs.get("verdicts") or {}).values():
                st_ = v.get("status")
                if st_ == "FAIL":
                    worst = "notready"
                elif st_ == "WARN" and worst != "notready":
                    worst = "review"
        biz = business.get(s) or {}
        facts.append({
            "synthesizer": s,
            "business_summary": biz.get("business_summary"),
            "use_cases": biz.get("use_cases"),
            "fidelity_score_0_to_1": (summ.get("fidelity") or {}).get("score"),
            "utility_score_0_to_1": (summ.get("utility") or {}).get("score"),
            "privacy_verdict": worst,
        })

    st["chat_messages"].append({"role": "system",
                                "content": "Synthesis finished. Results per synthesizer, business_summary "
                                "is the EXACT plain-language recommendation already printed on the "
                                "report's exec summary card right next to this chat: "
                                + json.dumps(facts)})
    message, options, focus = _chat_turn(st)
    return {"message": message, "options": options, "focus": focus}
