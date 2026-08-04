"""Regression suite for synth_eval's detection/metric logic.

Targeted at the scale-sensitivity and naming-convention work done to make the
pipeline behave on real production-shaped data (a handful of tables, wide
column sets, SCD-versioned dimension tables) instead of just the small demo
seed data it was originally tuned against: key-name normalization, the
auto_select_target guards + dim-table density note, entity-hub building
(single and multiple simultaneous keys), and the close-record filter /
nearest-record ceiling's percentile sensitivity.

No pytest, script-based like backend/evals.py -- every check runs against
real small pandas DataFrames (never mocks), is fast (no model fitting, no
network calls), and is meant to catch a future edit silently reintroducing
one of these bugs. Run from the repo root:

    .venv/bin/python -m synth_eval.regression_suite
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .columns import ColumnRoles, classify_columns
from .efficacy import auto_select_target, dim_table_density_note
from .entity import _normalize_key_name, _resolve_key_column, build_entity_hub, entity_key_tables
from .privacy import filter_close_records, filter_close_records_multitable, nearest_real_examples

CHECKS: list[tuple[str, "object"]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# key-name normalization (synth_eval.entity)
# ---------------------------------------------------------------------------

@check("normalize_key_name: strips the known X_ prefix, case-insensitive")
def _c_normalize_prefix():
    _assert(_normalize_key_name("X_OCCUPATION_TP_CD") == "OCCUPATION_TP_CD", "prefix not stripped")
    _assert(_normalize_key_name("OCCUPATION_TP_CD") == "OCCUPATION_TP_CD", "already-canonical changed")
    _assert(_normalize_key_name("occupation_tp_cd") == "OCCUPATION_TP_CD", "not case-folded")
    _assert(_normalize_key_name("x_occupation_tp_cd") == "OCCUPATION_TP_CD", "prefix+case not both handled")


@check("normalize_key_name: does not strip a coincidental X_-looking prefix into nothing")
def _c_normalize_no_overstrip():
    # "X_" alone would strip to "" -- the length guard must prevent that
    _assert(_normalize_key_name("X_") == "X_", "stripped down to an empty/degenerate key")
    # a column that merely STARTS with X but isn't the X_ convention (no underscore)
    _assert(_normalize_key_name("XAVIER_ID") == "XAVIER_ID", "false-positive prefix strip")


@check("resolve_key_column: finds a table's own local variant, or None")
def _c_resolve():
    df = pd.DataFrame({"X_OCCUPATION_TP_CD": [1, 2]})
    _assert(_resolve_key_column(df, "OCCUPATION_TP_CD") == "X_OCCUPATION_TP_CD", "local variant not found")
    df_literal = pd.DataFrame({"OCCUPATION_TP_CD": [1, 2]})
    _assert(_resolve_key_column(df_literal, "OCCUPATION_TP_CD") == "OCCUPATION_TP_CD", "literal match broken")
    df_none = pd.DataFrame({"UNRELATED": [1, 2]})
    _assert(_resolve_key_column(df_none, "OCCUPATION_TP_CD") is None, "should be no match")


@check("entity_key_tables: only lists tables that actually carry the key (literal or normalized)")
def _c_entity_key_tables():
    tables = {
        "A": pd.DataFrame({"CONT_ID": [1, 2]}),
        "B": pd.DataFrame({"X_CONT_ID": [1, 2]}),
        "C": pd.DataFrame({"OTHER": [1, 2]}),
    }
    found = set(entity_key_tables(tables, "CONT_ID"))
    _assert(found == {"A", "B"}, f"expected {{A, B}}, got {found}")


# ---------------------------------------------------------------------------
# auto_select_target guards (synth_eval.efficacy)
# ---------------------------------------------------------------------------

@check("auto_select_target: returns None below min_rows, even with a perfectly good target")
def _c_min_rows_guard():
    # threshold-based, not alternating: a greedy CART split can actually learn
    # this (an alternating Y/N/Y/N... pattern is a parity function, which
    # axis-aligned threshold splits can't learn without full memorization --
    # that's a real limitation of the tree model _predictive_signal uses, not
    # something a "perfectly good target" fixture should rely on)
    df = pd.DataFrame({"FLAG_CD": ["N"] * 10 + ["Y"] * 10, "AMOUNT": range(20)})
    roles = ColumnRoles(numeric=["AMOUNT"], categorical=["FLAG_CD"])
    _assert(auto_select_target(df, roles, min_rows=30) is None, "min_rows guard did not trigger")
    _assert(auto_select_target(df, roles, min_rows=10) is not None, "valid target rejected above min_rows")


@check("auto_select_target: returns None when the only categorical target has no other feature")
def _c_no_features_guard():
    # a lookup/dimension table: one id-like column (skipped) + one categorical,
    # nothing left to predict FROM once the id is excluded
    df = pd.DataFrame({"CODE_CD": ["A", "B", "C"] * 20})
    roles = ColumnRoles(numeric=[], categorical=["CODE_CD"])   # no other modelable column
    _assert(auto_select_target(df, roles, min_rows=10) is None, "picked a target with zero features")


@check("auto_select_target: picks a valid categorical target once a feature column exists")
def _c_picks_classification():
    # threshold-based, not modular (see _c_min_rows_guard's comment)
    df = pd.DataFrame({"CODE_CD": ["A"] * 20 + ["B"] * 20 + ["C"] * 20, "AMOUNT": range(60)})
    roles = ColumnRoles(numeric=["AMOUNT"], categorical=["CODE_CD"])
    sel = auto_select_target(df, roles, min_rows=10)
    _assert(sel == ("CODE_CD", "classification"), f"expected ('CODE_CD', 'classification'), got {sel}")


@check("auto_select_target: falls back to regression when no categorical target qualifies")
def _c_picks_regression():
    df = pd.DataFrame({"ID_CD": list(range(40)), "AMOUNT": np.linspace(0, 1000, 40)})
    # ID_CD has 40 distinct values on 40 rows -- outside the 2-20 class window, so it's
    # never picked as classification; AMOUNT is the only viable numeric target
    roles = ColumnRoles(numeric=["ID_CD", "AMOUNT"], categorical=[])
    sel = auto_select_target(df, roles, min_rows=10)
    _assert(sel is not None and sel[1] == "regression", f"expected a regression pick, got {sel}")


@check("dim_table_density_note: flags an SCD-style table (repeated attribute combos across versions)")
def _c_density_flags_dim_table():
    # 5 real-world entities, ~5 SCD-history rows each, attributes constant per
    # entity except a changing CURRENT_IND -- same shape as OCCUPATION.csv
    rows = []
    for code in range(5):
        for version in range(6):
            rows.append({"CODE_CD": code, "CATEGORY_CD": code % 2, "CURRENT_IND": "Y" if version == 5 else "N"})
    df = pd.DataFrame(rows)
    roles = ColumnRoles(numeric=[], categorical=["CODE_CD", "CATEGORY_CD", "CURRENT_IND"])
    note = dim_table_density_note(df, roles, "CATEGORY_CD")
    _assert(note is not None, "expected a low-distinctness note on SCD-repeated data")


@check("dim_table_density_note: stays silent on a table where rows are jointly unique")
def _c_density_silent_on_fact_table():
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "AMOUNT": rng.normal(size=n),
        "CHANNEL_CD": rng.choice(["A", "B", "C"], size=n),
        "STATUS_CD": rng.choice(["OPEN", "CLOSED"], size=n),
    })
    roles = ColumnRoles(numeric=["AMOUNT"], categorical=["CHANNEL_CD", "STATUS_CD"])
    note = dim_table_density_note(df, roles, "STATUS_CD")
    _assert(note is None, f"expected no note on a fact-shaped table, got: {note!r}")


# ---------------------------------------------------------------------------
# entity-hub building, single and multiple simultaneous keys (synth_eval.entity)
# ---------------------------------------------------------------------------

def _hub_fixture() -> dict[str, pd.DataFrame]:
    contact = pd.DataFrame({
        "CONT_ID": [1, 2, 3, 4, 5, 6],
        "SOLICIT_IND": ["Y", "N", "Y", "N", "Y", "N"],
    })
    # PERSON carries CONT_ID literally, and an X_-prefixed local variant of a
    # SECOND, independent key (occupation) -- the two-hub, mixed-naming case
    person = pd.DataFrame({
        "CONT_ID": [1, 2, 3, 4, 5, 6],
        "X_OCCUPATION_TP_CD": [10, 10, 20, 20, 30, 30],
        "GENDER_TP_CODE": ["F", "M", "F", "M", "F", "M"],
    })
    occupation = pd.DataFrame({
        "OCCUPATION_TP_CD": [10, 20, 30],
        "OCCUPATION_NAME": ["Teacher", "Engineer", "Nurse"],
    })
    return {"CONTACT": contact, "PERSON": person, "OCCUPATION": occupation}


@check("build_entity_hub: single key builds one parent with the right entity count + relationships")
def _c_single_hub():
    tables = _hub_fixture()
    new_tables, metadata, rels, info = build_entity_hub(
        tables, ["CONT_ID"], lift_invariant=False, child_tables={"CONT_ID": ["CONTACT", "PERSON"]})
    _assert("CONT_ID_HUB" in new_tables, "parent hub table missing")
    _assert(info["CONT_ID"]["n_entities"] == 6, f"expected 6 entities, got {info['CONT_ID']['n_entities']}")
    _assert(len(rels) == 2, f"expected 2 relationships (CONTACT, PERSON), got {len(rels)}")
    _assert({r["child_table_name"] for r in rels} == {"CONTACT", "PERSON"}, "wrong children in relationships")


@check("build_entity_hub: original caller tables are never mutated")
def _c_hub_no_mutation():
    tables = _hub_fixture()
    before_cols = list(tables["PERSON"].columns)
    build_entity_hub(tables, ["CONT_ID"], lift_invariant=False, child_tables={"CONT_ID": ["CONTACT", "PERSON"]})
    _assert(list(tables["PERSON"].columns) == before_cols, "caller's PERSON columns were mutated in place")
    _assert("X_OCCUPATION_TP_CD" in tables["PERSON"].columns, "local key variant renamed on the caller's own data")


@check("build_entity_hub: raises when the key is in none of the selected tables")
def _c_hub_missing_key():
    tables = _hub_fixture()
    try:
        build_entity_hub(tables, ["NOT_A_REAL_KEY"], lift_invariant=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a key present in no table")


@check("build_entity_hub: two simultaneous keys build two independent hubs in one pass")
def _c_multi_hub():
    tables = _hub_fixture()
    new_tables, metadata, rels, info = build_entity_hub(
        tables, ["CONT_ID", "OCCUPATION_TP_CD"], lift_invariant=False,
        child_tables={"CONT_ID": ["CONTACT", "PERSON"], "OCCUPATION_TP_CD": ["OCCUPATION", "PERSON"]})
    _assert("CONT_ID_HUB" in new_tables and "OCCUPATION_TP_CD_HUB" in new_tables, "missing a hub parent")
    _assert(info["CONT_ID"]["n_entities"] == 6, "CONT_ID hub entity count wrong")
    _assert(info["OCCUPATION_TP_CD"]["n_entities"] == 3, "OCCUPATION_TP_CD hub entity count wrong")
    _assert(len(rels) == 4, f"expected 4 relationships total across both hubs, got {len(rels)}")
    # PERSON is a child of BOTH hubs -- both its local key variant AND its
    # literal key must both resolve correctly on the SAME working copy
    _assert("CONT_ID" in new_tables["PERSON"].columns, "PERSON missing CONT_ID after multi-hub build")
    _assert("OCCUPATION_TP_CD" in new_tables["PERSON"].columns,
            "PERSON's X_-prefixed local variant was not normalized in multi-hub mode")
    md = metadata.to_dict()
    _assert(md["tables"]["CONT_ID_HUB"]["primary_key"] == "CONT_ID", "CONT_ID hub primary key not forced")
    _assert(md["tables"]["OCCUPATION_TP_CD_HUB"]["primary_key"] == "OCCUPATION_TP_CD",
            "OCCUPATION_TP_CD hub primary key not forced")


@check("build_entity_hub: multi-key metadata keeps BOTH hubs' relationships (no clobbering)")
def _c_multi_hub_relationships_survive():
    tables = _hub_fixture()
    _, metadata, rels, _ = build_entity_hub(
        tables, ["CONT_ID", "OCCUPATION_TP_CD"], lift_invariant=False,
        child_tables={"CONT_ID": ["CONTACT", "PERSON"], "OCCUPATION_TP_CD": ["OCCUPATION", "PERSON"]})
    md_rels = metadata.to_dict()["relationships"]
    _assert(len(md_rels) == len(rels) == 4, "metadata relationships don't match the returned rels list")
    parents = {r["parent_table_name"] for r in md_rels}
    _assert(parents == {"CONT_ID_HUB", "OCCUPATION_TP_CD_HUB"}, f"expected both hub parents, got {parents}")


# ---------------------------------------------------------------------------
# close-record filter + nearest-record ceiling, percentile sensitivity
# (synth_eval.privacy)
# ---------------------------------------------------------------------------

def _close_fixture(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    real = pd.DataFrame({
        "AMOUNT": rng.normal(loc=100, scale=20, size=n),
        "FLAG_CD": rng.choice(["A", "B"], size=n),
    })
    roles = ColumnRoles(numeric=["AMOUNT"], categorical=["FLAG_CD"])
    # synth = half near-exact copies of real rows (should be flagged as too
    # close), half far-away rows (should survive)
    close_half = real.iloc[: n // 2].copy()
    close_half["AMOUNT"] += 1e-6
    far_half = pd.DataFrame({
        "AMOUNT": rng.normal(loc=1000, scale=5, size=n // 2),
        "FLAG_CD": rng.choice(["A", "B"], size=n // 2),
    })
    synth = pd.concat([close_half, far_half], ignore_index=True)
    return real, synth, roles


@check("filter_close_records: rejects near-duplicate rows, keeps far ones")
def _c_filter_close_basic():
    real, synth, roles = _close_fixture()
    out = filter_close_records(real, synth, roles, percentile=50.0)
    report = out["report"]
    _assert(report["n_rejected"] > 0, "expected at least some rows rejected as too close")
    _assert(len(out["data"]) <= len(synth), "filtered output should not exceed input size without a resample_fn")


@check("filter_close_records: percentile is a real knob -- higher percentile rejects at least as much")
def _c_filter_percentile_monotonic():
    real, synth, roles = _close_fixture()
    low = filter_close_records(real, synth, roles, percentile=1.0)["report"]["n_rejected"]
    high = filter_close_records(real, synth, roles, percentile=50.0)["report"]["n_rejected"]
    _assert(high >= low, f"expected percentile=50 to reject >= percentile=1 (got {high} < {low})")


@check("filter_close_records_multitable: an entity-hub root (no roles entry) doesn't block its children")
def _c_filter_multitable_hub_root():
    real, synth, roles_df = _close_fixture(n=40)
    real_tables = {"CHILD": real}
    synth_tables = {"CHILD": synth, "SOME_HUB": pd.DataFrame({"KEY": range(20)})}
    roles = {"CHILD": roles_df}   # SOME_HUB deliberately has NO roles entry, like a derived hub
    relationships = [{"parent_table_name": "SOME_HUB", "parent_primary_key": "KEY",
                      "child_table_name": "CHILD", "child_foreign_key": "KEY"}]
    out = filter_close_records_multitable(real_tables, synth_tables, roles, relationships, percentile=50.0)
    _assert("CHILD" in out["report"]["tables"], "hub child fell through unchecked (the bug this guards against)")


@check("nearest_real_examples: percentile changes the reported ceiling, not just cosmetically")
def _c_nearest_percentile():
    real, synth, roles = _close_fixture()
    holdout = real.sample(frac=0.5, random_state=1)
    tight = nearest_real_examples(real, synth, roles, holdout=holdout, percentile=5.0)
    loose = nearest_real_examples(real, synth, roles, holdout=holdout, percentile=50.0)
    c_tight = tight.get("holdout_bootstrap_min_p05")
    c_loose = loose.get("holdout_bootstrap_min_p05")
    _assert(c_tight is not None and c_loose is not None, "ceiling missing from nearest_real_examples output")
    _assert(c_loose >= c_tight, f"expected percentile=50 ceiling >= percentile=5 ceiling ({c_loose} < {c_tight})")


# ---------------------------------------------------------------------------
# classify_columns cardinality threshold (max_categorical_card), used
# together with the above by every caller in backend/dashboard_core.py
# ---------------------------------------------------------------------------

@check("classify_columns: max_categorical_card demotes a wide *_CD column to skipped, not categorical")
def _c_max_categorical_card():
    df = pd.DataFrame({"WIDE_TP_CD": [f"V{i}" for i in range(30)]})
    narrow = classify_columns(df, {}, "T", max_categorical_card=50)
    wide = classify_columns(df, {}, "T", max_categorical_card=5)
    _assert("WIDE_TP_CD" in narrow.categorical, "expected categorical under the default (50) threshold")
    _assert("WIDE_TP_CD" in wide.skipped, "expected skipped once max_categorical_card is tightened below its cardinality")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- report every failure, don't stop the run
            failed += 1
            print(f"[FAIL] {name}\n       {type(e).__name__}: {e}")
        else:
            print(f"[PASS] {name}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
