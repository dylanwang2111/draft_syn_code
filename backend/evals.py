"""Automated eval suite for the chat assistant's tool-calling behaviour.

This exercises chat_assistant's real dispatch logic against whatever LLM
provider/model is actually configured in .env (Azure, direct OpenAI --
including gpt-5.1, or DeepSeek), with NO mocking of the model itself: every
scenario below makes real API calls, so this is a live behavioural check,
not a unit test. It answers three questions per scenario:

  1. necessary tool call     -- did a tool get called when the system prompt
                                 says one is required at that decision point?
  2. correct tool             -- was it the RIGHT tool (and none forbidden)?
  3. no unnecessary tool call -- did it stay quiet when nothing needed doing?

Two tiers:
  - "reaction" scenarios build a session directly into a specific precondition
    (schema confirmed or not, tables linked or not, a synthetic prior
    assistant question already in history) and send ONE user message, so the
    assertion is about a single, deterministic decision point. This avoids
    the false-fail noise of chaining many live LLM turns together.
  - one "full_journey" scenario drives a whole realistic conversation
    start-to-finish and checks looser, global invariants (step order never
    skipped, no gate bypass, no leaked bracket text anywhere).

Two invariants are checked on EVERY turn regardless of scenario, since they
guard against bugs found earlier in this project:
  - run_synthesis is never called before both schema_confirmed and
    relationships_confirmed were already true BEFORE this turn (regression
    guard for the tool-gate bypass fixed earlier).
  - no leaked bracket-list text (regression guard for the GPT-4.1
    "options like [...]" bug fixed earlier) -- reuses chat_assistant's own
    _LEAKED_OPTIONS_RE so this stays in sync with the real detector.

Run from the repo root:
    .venv/bin/python -m backend.evals
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from . import chat_assistant as ca

DONT_CARE = object()  # sentinel: distinguishes "no expectation" from "expect None/False"


def _stub_start_job(cfg: dict, st: dict) -> dict:
    """Stand-in for dashboard_core._start_job: this eval suite is checking
    tool-call correctness, not running real SDV training in a background
    thread every time a scenario reaches run_synthesis. Same return shape,
    no actual work."""
    st["job"] = {"status": "running", "log": [], "error": None, "pct": 0.0}
    st["results"] = None
    return {"status": "running"}


ca._start_job = _stub_start_job


# ---------------------------------------------------------------------------
# fixtures: small REAL pandas tables (set_relationship/set_entity_key
# validate against actual data via dashboard_core._validate_relationships,
# not just column-name metadata)
# ---------------------------------------------------------------------------

def _fake_tables() -> dict[str, pd.DataFrame]:
    contact = pd.DataFrame({
        "CONT_ID": [1, 2, 3, 4, 5, 6],
        "CONTACT_NAME": ["Alice A", "Bob B", "Cara C", "Dan D", "Eve E", "Fay F"],
        "SOLICIT_IND": ["Y", "N", "Y", "N", "Y", "N"],
    })
    person = pd.DataFrame({
        "CONT_ID": [1, 2, 3, 4, 5, 6],
        "MARITAL_ST_TP_CD": ["S", "M", "M", "S", "D", "M"],
        "GENDER_TP_CODE": ["F", "M", "F", "M", "F", "M"],
    })
    return {"CONTACT": contact, "PERSON": person}


def _profile_tables_meta(tables: dict[str, pd.DataFrame]) -> dict:
    return {t: {"rows": len(df), "columns": [{"name": c} for c in df.columns]} for t, df in tables.items()}


def _fake_profile(linked: bool) -> dict:
    rels = [{"parent_table_name": "CONTACT", "parent_primary_key": "CONT_ID",
             "child_table_name": "PERSON", "child_foreign_key": "CONT_ID"}] if linked else []
    return {"recommendation": {"tier": 1 if linked else 2, "relationships": rels}}


def _fresh_session(linked: bool) -> dict:
    tables = _fake_tables()
    st = {"tables": tables, "chat_messages": [], "chat_plan": None, "job": None, "results": None}
    plan = ca._chat_plan_from_profile(_profile_tables_meta(tables), _fake_profile(linked))
    st["chat_plan"] = plan
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
    return st


def _seed_schema_asked(st: dict) -> None:
    """Fast-forward past the upload-analysis turn with a synthetic prior
    assistant message, instead of spending a real API call to get there."""
    st["chat_messages"].append({"role": "assistant",
                                "content": "You have two tables, CONTACT and PERSON. Some columns look like "
                                "personal info, those will be auto-faked. Want to review the auto-detected "
                                "column types before we go further?"})


def _seed_schema_confirmed(st: dict) -> None:
    _seed_schema_asked(st)
    st["chat_messages"].append({"role": "user", "content": "Looks good"})
    st["chat_messages"].append({"role": "assistant", "content": "Great, moving on."})
    st["chat_plan"]["schema_confirmed"] = True


def _seed_relationships_asked(st: dict) -> None:
    _seed_schema_confirmed(st)
    st["chat_messages"].append({"role": "assistant",
                                "content": "Are these tables related to each other? CONT_ID shows up in "
                                "both CONTACT and PERSON, which might be the link."})


def _seed_relationships_confirmed(st: dict, linked: bool) -> None:
    _seed_relationships_asked(st)
    st["chat_messages"].append({"role": "user", "content": "No, independent" if not linked else
                                "Yes, CONT_ID links them"})
    st["chat_messages"].append({"role": "assistant", "content": "Got it."})
    st["chat_plan"]["relationships_confirmed"] = True
    if linked:
        st["chat_plan"]["relationships"] = _fake_profile(True)["recommendation"]["relationships"]
        st["chat_plan"]["synth"] = "HMA"


def _seed_recommendation_asked(st: dict, linked: bool) -> None:
    _seed_relationships_confirmed(st, linked)
    synth = st["chat_plan"]["synth"]
    st["chat_messages"].append({"role": "assistant",
                                "content": f"Based on that, I'd recommend {synth}. Want to go with that "
                                "pick, or compare a couple of options side by side?"})


def _seed_results_narrated(st: dict, linked: bool) -> None:
    _seed_recommendation_asked(st, linked)
    st["chat_messages"].append({"role": "user", "content": "Go with your pick"})
    st["chat_plan"]["selected_synths"] = [st["chat_plan"]["synth"]]
    st["chat_messages"].append({"role": "system",
                                "content": "Synthesis finished. Results per synthesizer: "
                                + json.dumps([{"synthesizer": st["chat_plan"]["synth"],
                                              "business_summary": "This synthetic data behaves like your "
                                              "real data, carries no privacy red flags, and is about 90% "
                                              "as useful as real data for analytics. Recommended for "
                                              "dev/test environments, vendor sharing, and model training.",
                                              "use_cases": {"dev_test": True, "vendor_sharing": True,
                                                            "model_training": True},
                                              "fidelity_score_0_to_1": 0.91, "utility_score_0_to_1": 0.9,
                                              "privacy_verdict": "ready"}])})
    st["chat_messages"].append({"role": "assistant",
                                "content": f"{st['chat_plan']['synth']} produced synthetic data with no "
                                "privacy red flags. What would you like to do next?"})


# ---------------------------------------------------------------------------
# scenario/turn definitions
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    note: str
    user: str | None = None                 # None only for a turn with no new user message to append
    expect_tool: object = DONT_CARE         # True: >=1 tool call must happen; False: none must happen
    require_tools: object = DONT_CARE       # set[str]: all of these must appear among the calls
    forbidden_tools: object = DONT_CARE     # set[str]: none of these may appear among the calls
    expect_focus: object = DONT_CARE        # expected focus value (None is a valid expectation)
    expect_options_nonempty: object = DONT_CARE   # True: options must render as real buttons


@dataclass
class Scenario:
    name: str
    linked: bool
    setup: object          # callable(st) -> None, mutates st into the starting precondition
    turns: list[Turn]


SCENARIOS: list[Scenario] = [
    Scenario("upload_asks_schema_question", linked=False, setup=lambda st: None, turns=[
        Turn("fresh upload must ask about schema via a real tool call, nothing else",
             require_tools={"ask_question"}, forbidden_tools={"run_synthesis", "confirm_schema",
             "confirm_relationships", "open_schema_editor", "open_data_model"},
             expect_focus=None, expect_options_nonempty=True),
    ]),
    Scenario("schema_looks_good", linked=False, setup=_seed_schema_asked, turns=[
        Turn("'Looks good' must confirm schema, must not open anything or touch relationships",
             user="Looks good", require_tools={"confirm_schema"}, expect_focus=None,
             forbidden_tools={"open_schema_editor", "confirm_relationships", "run_synthesis"}),
    ]),
    Scenario("schema_edit_opens_editor", linked=False, setup=_seed_schema_asked, turns=[
        Turn("'Let me edit it' must open the real schema editor panel, must not self-confirm",
             user="Let me edit it", require_tools={"open_schema_editor"}, expect_focus="schema",
             forbidden_tools={"confirm_schema"}),
    ]),
    Scenario("schema_edit_by_words", linked=False, setup=_seed_schema_asked, turns=[
        Turn("describing a column change in words must call set_column_types, "
             "must not jump straight to confirm_schema without checking for more",
             user="Please mark SOLICIT_IND on CONTACT as categorical",
             require_tools={"set_column_types"}, forbidden_tools={"confirm_schema"}),
    ]),
    Scenario("relationships_no", linked=False, setup=_seed_relationships_asked, turns=[
        Turn("'No, independent' must confirm relationships=false, must not open anything or run",
             user="No, independent", require_tools={"confirm_relationships"}, expect_focus=None,
             forbidden_tools={"open_data_model", "run_synthesis"}),
    ]),
    Scenario("relationships_setup_opens_data_model", linked=False, setup=_seed_relationships_asked, turns=[
        Turn("'Let me set it up' must open the real Data Model panel, must not self-confirm",
             user="Let me set it up", require_tools={"open_data_model"}, expect_focus="data_model",
             forbidden_tools={"confirm_relationships"}),
    ]),
    Scenario("relationships_by_words", linked=True, setup=_seed_relationships_asked, turns=[
        Turn("describing the FK relationship in words must call set_relationship AND "
             "confirm_relationships together (per the system prompt's words-branch)",
             user="CONTACT.CONT_ID is the parent key and PERSON.CONT_ID points at it",
             require_tools={"set_relationship", "confirm_relationships"}),
    ]),
    Scenario("recommendation_go_with_pick", linked=True, setup=lambda st: _seed_recommendation_asked(st, True),
              turns=[
        Turn("'Go with your pick' must call run_synthesis",
             user="Go with your pick", require_tools={"run_synthesis"}),
    ]),
    Scenario("recommendation_compare", linked=True, setup=lambda st: _seed_recommendation_asked(st, True),
              turns=[
        Turn("'Compare a couple' must open the synthesizers panel, must NOT auto-run",
             user="I'd like to compare a couple of them myself",
             require_tools={"open_config_panel"}, expect_focus="synthesizers",
             forbidden_tools={"run_synthesis"}),
    ]),
    Scenario("explain_synthesizer_midflow", linked=False, setup=_seed_schema_asked, turns=[
        Turn("asking what a synthesizer is mid-flow must call explain_synthesizer, "
             "must NOT touch schema/relationships state",
             user="what is HMA and how is it different from GaussianCopula?",
             require_tools={"explain_synthesizer"},
             forbidden_tools={"confirm_schema", "confirm_relationships", "run_synthesis"}),
    ]),
    Scenario("offtopic_question_no_state_mutation", linked=False, setup=_seed_schema_asked, turns=[
        Turn("an off-topic informational question must not silently advance/confirm any step",
             user="How long does this whole process usually take?",
             forbidden_tools={"confirm_schema", "confirm_relationships", "run_synthesis",
                              "set_column_types", "set_relationship", "set_entity_key"}),
    ]),
    Scenario("download_after_results", linked=False, setup=lambda st: _seed_results_narrated(st, False),
              turns=[
        Turn("asking to download after results must open the synthetic_data panel",
             user="Can I download the synthetic data now?",
             require_tools={"open_config_panel"}, expect_focus="synthetic_data"),
    ]),
    Scenario("closing_remark_no_forced_tool", linked=False, setup=lambda st: _seed_results_narrated(st, False),
              turns=[
        Turn("a plain closing remark after everything is done should NOT force another "
             "ask_question loop -- tests over-triggering, not just under-triggering",
             user="Perfect, thanks, that's all for now!", expect_tool=False),
    ]),
    Scenario("full_journey_linked", linked=True, setup=lambda st: None, turns=[
        Turn("upload analysis asks about schema", require_tools={"ask_question"}),
        Turn("user accepts schema as-is", user="Looks good", require_tools={"confirm_schema"}),
        Turn("user describes the relationship in words", user="CONT_ID links CONTACT and PERSON, "
             "CONTACT.CONT_ID is the parent", forbidden_tools={"run_synthesis"}),
        Turn("user goes with the recommendation", user="Sure, go with your pick",
             require_tools={"run_synthesis"}),
    ]),
]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _tool_calls_since(st: dict, start_idx: int) -> list[tuple[str, bool]]:
    """(name, ok) per tool call made since start_idx -- ok=False means the
    dispatcher's own gate refused it (its "tool" result contained an
    "error" key), e.g. confirm_schema bundled into the same turn as
    open_schema_editor. The model DID attempt the call, but it never took
    effect, so callers should treat it as if it hadn't happened."""
    calls = []
    msgs = st["chat_messages"][start_idx:]
    i = 0
    while i < len(msgs):
        m = msgs[i]
        tcs = m.get("tool_calls") if m.get("role") == "assistant" else None
        if tcs:
            names = [c["function"]["name"] for c in tcs]
            results = msgs[i + 1:i + 1 + len(names)]
            for name, res_msg in zip(names, results):
                try:
                    ok = "error" not in json.loads(res_msg.get("content") or "{}")
                except (json.JSONDecodeError, TypeError):
                    ok = True
                calls.append((name, ok))
            i += 1 + len(names)
        else:
            i += 1
    return calls


def _run_turn(st: dict, turn: Turn) -> dict:
    plan = st["chat_plan"]
    pre_schema_confirmed = bool(plan.get("schema_confirmed"))
    pre_rel_confirmed = bool(plan.get("relationships_confirmed"))

    if turn.user is not None:
        st["chat_messages"].append({"role": "user", "content": turn.user})
    start_idx = len(st["chat_messages"])
    text, options, focus = ca._chat_turn(st)
    calls_with_status = _tool_calls_since(st, start_idx)
    attempted_names = [n for n, _ in calls_with_status]      # for display/debugging
    tool_names = [n for n, ok in calls_with_status if ok]    # actually took effect -- what assertions use

    problems = []
    if turn.expect_tool is True and not tool_names:
        problems.append("expected a tool call, none happened")
    if turn.expect_tool is False and tool_names:
        problems.append(f"expected NO tool call, got {tool_names}")
    if turn.require_tools is not DONT_CARE:
        missing = turn.require_tools - set(tool_names)
        if missing:
            problems.append(f"missing required tool(s) {missing}, got {tool_names}")
    if turn.forbidden_tools is not DONT_CARE:
        present = turn.forbidden_tools & set(tool_names)
        if present:
            problems.append(f"forbidden tool(s) called: {present}")
    if turn.expect_focus is not DONT_CARE and focus != turn.expect_focus:
        problems.append(f"expected focus={turn.expect_focus!r}, got {focus!r}")
    if turn.expect_options_nonempty is True and not options:
        problems.append("expected clickable options, got none")
    if turn.expect_options_nonempty is False and options:
        problems.append(f"expected no options, got {options}")

    # always-on regression guards, independent of what this specific turn was testing
    if "run_synthesis" in tool_names and not (pre_schema_confirmed and pre_rel_confirmed):
        problems.append("GATE BYPASS: run_synthesis called before both confirmations were true")
    if "confirm_relationships" in tool_names and not pre_schema_confirmed:
        problems.append("STEP-ORDER BYPASS: confirm_relationships called before schema was confirmed")
    leak = ca._LEAKED_OPTIONS_RE.search(text or "")
    if leak:
        problems.append(f"leaked bracket-list text survived in reply: ...{text[-100:]!r}")
    if ca._looks_like_tool_call_json(text or ""):
        problems.append(f"leaked raw tool-call JSON survived in reply: {text[:120]!r}")

    return {"note": turn.note, "user": turn.user, "text": text, "options": options, "focus": focus,
            "tool_calls": tool_names, "attempted": attempted_names, "problems": problems}


def run_scenario(scenario: Scenario) -> list[dict]:
    st = _fresh_session(scenario.linked)
    scenario.setup(st)
    return [_run_turn(st, turn) for turn in scenario.turns]


def main():
    if ca.llm_client is None:
        print("No LLM provider configured -- set an API key in .env first (see .env.example).")
        return
    print(f"Model under test: {ca.LLM_MODEL}\n")

    total_turns, failed_turns, failed_scenarios = 0, 0, 0
    for scenario in SCENARIOS:
        print(f"=== {scenario.name} ({'linked' if scenario.linked else 'unlinked'} tables) ===")
        results = run_scenario(scenario)
        scenario_ok = True
        for r in results:
            total_turns += 1
            status = "PASS" if not r["problems"] else "FAIL"
            if r["problems"]:
                failed_turns += 1
                scenario_ok = False
            print(f"  [{status}] {r['note']}")
            if r["user"] is not None:
                print(f"         user: {r['user']!r}")
            print(f"         tool_calls: {r['tool_calls']}  focus: {r['focus']!r}  "
                  f"options: {r['options']}")
            refused = [n for n in r["attempted"] if n not in r["tool_calls"]]
            if refused:
                print(f"         (also attempted but refused by the dispatcher's own gate: {refused})")
            print(f"         reply: {r['text'][:160]!r}")
            for p in r["problems"]:
                print(f"         PROBLEM: {p}")
        if not scenario_ok:
            failed_scenarios += 1
        print()

    print(f"{'='*60}\n{total_turns - failed_turns}/{total_turns} turns passed across "
          f"{len(SCENARIOS)} scenarios ({failed_scenarios} scenario(s) had a failure).")


if __name__ == "__main__":
    main()
