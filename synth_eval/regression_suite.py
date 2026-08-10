"""Regression suite for synth_eval's detection/metric logic.

Targeted at the scale-sensitivity and naming-convention work done to make the
pipeline behave on real production-shaped data (a handful of tables, wide
column sets, SCD-versioned dimension tables) instead of just the small demo
seed data it was originally tuned against: key-name normalization, the
auto_select_target guards, entity-hub building
(single and multiple simultaneous keys), and the close-record filter /
nearest-record ceiling's percentile sensitivity.

No pytest, script-based like backend/evals.py -- every check runs against
real small pandas DataFrames (never mocks), is fast (no model fitting, no
network calls -- one deliberate exception: a tiny/few-epoch TabSyn fit+sample
smoke test, since that model's fit/sample contract is this repo's own code,
not an external library's, and deserves the same regression coverage as
everything else here), and is meant to catch a future edit silently
reintroducing one of these bugs. Run from the repo root:

    .venv/bin/python -m synth_eval.regression_suite
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .columns import (ColumnRoles, auto_categorical_threshold, best_refill_group_column,
                      classify_columns, group_diversity_reduction, suffix_sdtype_overrides)
from .efficacy import auto_select_target
from .entity import _normalize_key_name, _resolve_key_column, build_entity_hub, entity_key_tables
from .link import link_relationships
from .privacy import filter_close_records, filter_close_records_multitable, nearest_real_examples

# imported from backend, not synth_eval -- _refill's conditional-grouping
# dispatch is where the actual bug this section guards against lives (see
# below), the synth_eval-level building blocks alone don't exercise it
from backend.dashboard_core import _detect, _metadata_from_request, _refill

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


@check("link_relationships: relinks a multi-parent child's two hub FKs independently, 100% coverage")
def _c_link_relationships_multi_parent_hub():
    # backend/dashboard_core.py builds a per-key union pool for single-table
    # synths (HMA models hubs jointly and doesn't need this), then relinks
    # every hub relationship against it -- this is what makes that pool
    # coherent for PERSON, which sits under BOTH hubs at once, instead of each
    # child's independently-fit FK column holding unrelated numbers that just
    # happen to share a column name.
    tables = _hub_fixture()
    fit_tables, _, hub_rels, hub_info = build_entity_hub(
        tables, ["CONT_ID", "OCCUPATION_TP_CD"], lift_invariant=False,
        child_tables={"CONT_ID": ["CONTACT", "PERSON"], "OCCUPATION_TP_CD": ["OCCUPATION", "PERSON"]})
    parent_names = {k: info["parent"] for k, info in hub_info.items()}

    # worst case: a "synth" whose per-table models generated FK-ish columns
    # totally disjoint from every other table's, including PERSON's own two
    # keys -- nothing here should coincidentally already line up
    suite = {"GaussianCopula": {
        "CONTACT": pd.DataFrame({"CONT_ID": [101, 102, 103]}),
        "PERSON": pd.DataFrame({"CONT_ID": [201, 202], "OCCUPATION_TP_CD": [901, 902]}),
        "OCCUPATION": pd.DataFrame({"OCCUPATION_TP_CD": [301]}),
    }}
    entity_children_by_key = {"CONT_ID": ["CONTACT", "PERSON"], "OCCUPATION_TP_CD": ["OCCUPATION", "PERSON"]}

    for s, tabs in suite.items():
        for entity_key, pname in parent_names.items():
            ids = {v for t in entity_children_by_key[entity_key] for v in tabs[t][entity_key]}
            tabs[pname] = pd.DataFrame({entity_key: sorted(ids)})

    linked = link_relationships(hub_rels, fit_tables, suite, list(suite), seed=0)
    _assert(len(linked["GaussianCopula"]) == 4, f"expected all 4 hub relationships linked, got {linked}")

    tabs = suite["GaussianCopula"]
    for r in hub_rels:
        pt, pk = r["parent_table_name"], r["parent_primary_key"]
        ct, fk = r["child_table_name"], r["child_foreign_key"]
        cov = tabs[ct][fk].isin(set(tabs[pt][pk])).mean()
        _assert(cov == 1.0, f"{ct}.{fk} -> {pt}.{pk} coverage {cov}, expected 1.0 after relinking")


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


@check("auto_categorical_threshold: scales with row count instead of a fixed number")
def _c_auto_categorical_threshold_scales():
    _assert(auto_categorical_threshold(100) == 90, "expected 90% of 100 rows = 90")
    _assert(auto_categorical_threshold(10) == 9, "expected 90% of 10 rows = 9")
    _assert(auto_categorical_threshold(1) == 1, "must never round down to 0 on a tiny table")


@check("classify_columns: with max_categorical_card left unset, auto-detects PER TABLE from row count")
def _c_classify_columns_auto_detect():
    # 80 distinct codes over 1000 rows (8% distinct) -- a real wide production
    # code, well past the OLD fixed default of 50 but nowhere near id-like
    wide_real = pd.DataFrame({"OCCUPATION_TP_CD": (list(range(80)) * 13)[:1000]})
    roles = classify_columns(wide_real, {}, "T")   # no max_categorical_card passed at all
    _assert("OCCUPATION_TP_CD" in roles.categorical,
            "an 80-value code on a 1000-row table is 8% distinct -- auto-detection must not fall back "
            "to the old fixed 50 and wrongly skip it as id-like")

    # 950 distinct values over 1000 rows (95% distinct) -- genuinely id-like,
    # regardless of its *_TP_CD suffix, must still be excluded automatically
    id_like = pd.DataFrame({"WAREHOUSE_TP_CD": [f"W{i}" for i in range(950)] + [f"W{i}" for i in range(50)]})
    roles2 = classify_columns(id_like, {}, "T")
    _assert("WAREHOUSE_TP_CD" in roles2.skipped,
            "95% distinct on a *_TP_CD column must still auto-detect as id-like and be skipped")


@check("suffix_sdtype_overrides: promotes a *_TP_CD column SDV left 'numerical' back to categorical")
def _c_suffix_sdtype_overrides_promotes():
    df = pd.DataFrame({"OCCUPATION_TP_CD": list(range(21)) * 5, "AMOUNT": np.random.rand(105)})
    sdtypes = {"OCCUPATION_TP_CD": "numerical", "AMOUNT": "numerical"}
    fixed = suffix_sdtype_overrides(df, sdtypes)
    _assert(fixed["OCCUPATION_TP_CD"] == "categorical",
            f"expected OCCUPATION_TP_CD promoted to categorical, got {fixed['OCCUPATION_TP_CD']}")
    _assert(fixed["AMOUNT"] == "numerical", "a plain numeric column with no code-like suffix must stay numerical")


@check("suffix_sdtype_overrides: leaves a high-cardinality *_TP_CD column numerical (id-like, not a code)")
def _c_suffix_sdtype_overrides_respects_cardinality():
    df = pd.DataFrame({"COMPANY_TP_CD": list(range(83))})
    sdtypes = {"COMPANY_TP_CD": "numerical"}
    fixed = suffix_sdtype_overrides(df, sdtypes, max_categorical_card=50)
    _assert(fixed["COMPANY_TP_CD"] == "numerical",
            "83 distinct values exceeds the categorical-cardinality guard, must not be promoted")


@check("suffix_sdtype_overrides: never touches a column SDV already got right")
def _c_suffix_sdtype_overrides_leaves_correct_alone():
    df = pd.DataFrame({"STATUS_CD": ["A", "B", "A", "C"]})
    sdtypes = {"STATUS_CD": "categorical"}
    fixed = suffix_sdtype_overrides(df, sdtypes)
    _assert(fixed["STATUS_CD"] == "categorical", "already-categorical sdtype must be left untouched")


@check("_metadata_from_request: re-applies suffix_sdtype_overrides at the run's OWN max_categorical_card")
def _c_metadata_from_request_rechecks_cardinality():
    # integer-typed code (matches real *_TP_CD columns, e.g. 348820.0 in the
    # seed data), 55 distinct values on a 60-row table -- 91.7% distinct, so
    # even the ratio-based AUTO default (90% of 60 = 54) leaves it
    # 'numerical', same as a genuine near-id column would. A user who
    # explicitly knows this code space goes up to ~100 values sets
    # max_categorical_card=100 for their run; that RUN-TIME choice
    # (documented in _run_job as exactly the knob for "real production data,
    # a different cardinality distribution") must reach the sdtype actually
    # fed to the synthesizer, not just classify_columns' role decision --
    # otherwise the column is fit as a continuous distribution and rounded
    # back, inventing values and wrecking its column-shape score regardless
    # of anything downstream.
    df = pd.DataFrame({"OCCUPATION_TP_CD": list(range(340000, 340055))
                        + list(range(340000, 340005))})  # 55 distinct, 60 rows
    tables = {"OCCUPATION": df}
    st = {"tables": tables, "meta_detected": _detect(tables)}
    stale = _metadata_from_request({}, [], st)
    _assert(stale["OCCUPATION"]["columns"]["OCCUPATION_TP_CD"]["sdtype"] == "numerical",
            "sanity check: 55/60 = 91.7% distinct should stay numerical even under auto-detection")
    fixed = _metadata_from_request({}, [], st, tables=tables, max_categorical_card=100)
    _assert(fixed["OCCUPATION"]["columns"]["OCCUPATION_TP_CD"]["sdtype"] == "categorical",
            "an explicit run-time max_categorical_card=100 must promote a 55-value *_TP_CD "
            "column to categorical even though this table's own auto-detected ratio wouldn't")


@check("_metadata_from_request: an explicit user sdtype edit still wins over the re-check")
def _c_metadata_from_request_user_edit_wins():
    df = pd.DataFrame({"OCCUPATION_TP_CD": list(range(340000, 340080)) * 3})
    tables = {"OCCUPATION": df}
    st = {"tables": tables, "meta_detected": _detect(tables)}
    edited = _metadata_from_request({"OCCUPATION": {"sdtypes": {"OCCUPATION_TP_CD": "numerical"}}},
                                     [], st, tables=tables, max_categorical_card=100)
    _assert(edited["OCCUPATION"]["columns"]["OCCUPATION_TP_CD"]["sdtype"] == "numerical",
            "a user's explicit schema-editor override must not be clobbered by the cardinality re-check")


# ---------------------------------------------------------------------------
# conditional refill: a filled-in column (NAME/DESC/...) resampled from real
# rows that share a modeled column's value, instead of from the whole table,
# with a minimum-group-size privacy floor as the safety valve
# ---------------------------------------------------------------------------

def _refill_fixture() -> pd.DataFrame:
    """9 well-populated codes (12 real rows each, well above any sane
    min_group_size) each perfectly determining their own label, plus one
    sparse code D with only 3 real rows -- deliberately below the default
    min_group_size=10, to exercise the privacy-floor fallback.

    With G distinct codes each 1:1 with their own label, the theoretical max
    of group_diversity_reduction is 1 - 1/G (a group can narrow LABEL_DESC
    down to one value, but LABEL_DESC still has G distinct values overall) --
    10 groups here keeps that ceiling close to 1 (0.9) without a fixture the
    size of the real OCCUPATION table this mirrors.
    """
    codes, names = [], []
    for code, count in (("A", 12), ("B", 12), ("C", 12), ("D", 3), ("E", 12),
                        ("F", 12), ("G", 12), ("H", 12), ("I", 12), ("J", 12)):
        codes += [code] * count
        names += [f"Name-{code}"] * count
    return pd.DataFrame({
        "CODE_CD": codes,             # modeled (categorical, by suffix)
        "LABEL_DESC": names,          # filled-in, tied to CODE_CD
        "AUDIT_USER": ["SYS"] * len(codes),  # filled-in, unrelated to CODE_CD
    })


@check("group_diversity_reduction: near its ceiling when the group column fully determines the target")
def _c_group_diversity_reduction_high():
    score = group_diversity_reduction(_refill_fixture(), "CODE_CD", "LABEL_DESC")
    # ceiling is 1 - 1/10 = 0.9 for this fixture's 10 distinct codes/labels
    _assert(score > 0.85, f"expected close to the 0.9 ceiling for a perfect 1:1 mapping, got {score}")


@check("group_diversity_reduction: 0 when the target has no diversity for any grouping to reduce")
def _c_group_diversity_reduction_low():
    score = group_diversity_reduction(_refill_fixture(), "CODE_CD", "AUDIT_USER")
    _assert(score == 0.0, f"a constant target column has nothing to reduce, expected 0.0, got {score}")


@check("best_refill_group_column: picks the column a fill column is actually tied to, not an unrelated one")
def _c_best_refill_group_column_picks_right_one():
    real = _refill_fixture()
    real["NOISE_CD"] = (["X", "Y"] * len(real))[:len(real)]  # alternates independently of CODE_CD's grouping
    best = best_refill_group_column(real, "LABEL_DESC", ["CODE_CD", "NOISE_CD"])
    _assert(best == "CODE_CD", f"expected CODE_CD, got {best}")


@check("best_refill_group_column: returns None when nothing clears the association threshold")
def _c_best_refill_group_column_none():
    best = best_refill_group_column(_refill_fixture(), "AUDIT_USER", ["CODE_CD"])
    _assert(best is None, f"expected None (no real relationship worth conditioning on), got {best}")


@check("_refill: conditions a fill column on its matching modeled column once the real group is large enough")
def _c_refill_conditions_on_matching_group():
    real = _refill_fixture()
    synth = pd.DataFrame({"CODE_CD": ["A"] * 30})
    out = _refill(synth, real, ["LABEL_DESC", "AUDIT_USER"], ["CODE_CD", "LABEL_DESC", "AUDIT_USER"],
                  seed=0, group_candidates=["CODE_CD"], min_group_size=10)
    _assert((out["LABEL_DESC"] == "Name-A").all(),
            "every synthetic row with CODE_CD=A should get A's real label once conditioning is on")


def _noisy_group_fixture() -> pd.DataFrame:
    """REGION_CD only PARTIALLY narrows SEGMENT_DESC -- an individual-level
    attribute that correlates with the group but isn't determined by it,
    unlike _refill_fixture's CODE_CD/LABEL_DESC (a true 1:1 lookup label).
    Association clears min_association (0.6 >= 0.5) but sits at 75% of its
    own ceiling (0.8), well under the near-total bar -- so region Z's small
    group (3 rows, itself not even uniform) must still get the individual-
    level privacy floor, not the lookup-table bypass.
    """
    codes, segs = [], []
    for code, pattern in {
        "V": ["B1"] * 8 + ["B2"] * 4, "W": ["B2"] * 8 + ["B3"] * 4,
        "X": ["B3"] * 8 + ["B4"] * 4, "Y": ["B4"] * 8 + ["B5"] * 4,
    }.items():
        codes += [code] * len(pattern)
        segs += pattern
    codes += ["Z"] * 3
    segs += ["B1", "B5", "B1"]
    return pd.DataFrame({"REGION_CD": codes, "SEGMENT_DESC": segs})


@check("_refill: privacy floor still blocks a small group when the dependency isn't near-total")
def _c_refill_privacy_floor_blocks_small_group():
    real = _noisy_group_fixture()  # region Z has only 3 real rows, itself non-uniform
    synth = pd.DataFrame({"REGION_CD": ["Z"] * 30})
    out = _refill(synth, real, ["SEGMENT_DESC"], ["REGION_CD", "SEGMENT_DESC"],
                  seed=0, group_candidates=["REGION_CD"], min_group_size=10)
    distinct = set(out["SEGMENT_DESC"])
    _assert(len(distinct) > 2,
            f"Z's real group (3 rows) is below min_group_size=10 and the REGION_CD->SEGMENT_DESC "
            f"dependency isn't near-total, expected a whole-table fallback mix (5 possible segments), "
            f"got only {distinct} -- the privacy floor isn't blocking a too-small, non-deterministic group")


@check("_refill: privacy floor is bypassed for a near-total (lookup-table) dependency on a small group")
def _c_refill_deterministic_bypasses_floor():
    real = _refill_fixture()  # code D has only 3 real rows, but CODE_CD->LABEL_DESC is a true 1:1 lookup
    synth = pd.DataFrame({"CODE_CD": ["D"] * 30})
    out = _refill(synth, real, ["LABEL_DESC", "AUDIT_USER"], ["CODE_CD", "LABEL_DESC", "AUDIT_USER"],
                  seed=0, group_candidates=["CODE_CD"], min_group_size=10)
    _assert((out["LABEL_DESC"] == "Name-D").all(),
            "D's CODE_CD->LABEL_DESC dependency is a near-total 1:1 lookup (public reference data, not "
            "individual-level information) -- conditioning should bypass the min_group_size=10 floor "
            "despite D's real group having only 3 rows, same as OCCUPATION_TP_CD->OCCUPATION_NAME in prod")


@check("_refill: conditioning kicks in once min_group_size is lowered to match the real group size")
def _c_refill_conditions_once_floor_lowered():
    real = _refill_fixture()
    synth = pd.DataFrame({"CODE_CD": ["D"] * 30})
    out = _refill(synth, real, ["LABEL_DESC", "AUDIT_USER"], ["CODE_CD", "LABEL_DESC", "AUDIT_USER"],
                  seed=0, group_candidates=["CODE_CD"], min_group_size=3)
    _assert((out["LABEL_DESC"] == "Name-D").all(),
            "with min_group_size lowered to D's actual real count (3), every row should condition correctly")


@check("_refill: an unseen synthetic code value falls back to the whole table instead of crashing")
def _c_refill_unseen_code_falls_back():
    real = _refill_fixture()
    synth = pd.DataFrame({"CODE_CD": ["ZZZ"] * 10})
    out = _refill(synth, real, ["LABEL_DESC", "AUDIT_USER"], ["CODE_CD", "LABEL_DESC", "AUDIT_USER"],
                  seed=0, group_candidates=["CODE_CD"], min_group_size=10)
    _assert(out["LABEL_DESC"].notna().all(), "an unseen code must still get a real fallback value, not NaN")


# ---------------------------------------------------------------------------
# TabSyn (synth_eval.tabsyn) -- fit()/sample() contract smoke test. Tiny data
# and 2 epochs deliberately don't test generation QUALITY (that needs real
# training time and is validated separately, not on every regression run),
# just that fitting and sampling completes and honors the same interface
# shape SDV's own single-table synthesizers do.
# ---------------------------------------------------------------------------

@check("TabSynSynthesizer: fits and samples without crashing, honoring the SDV single-table contract")
def _c_tabsyn_fit_sample_smoke():
    from .tabsyn import TabSynSynthesizer

    rng = np.random.RandomState(0)
    n = 40
    df = pd.DataFrame({
        "AMOUNT": rng.normal(100, 20, n),
        "STATUS_CD": rng.choice(["A", "B", "C"], n),
    })
    df.loc[rng.choice(n, 4, replace=False), "AMOUNT"] = np.nan

    class _FakeMeta:
        def to_dict(self):
            return {"columns": {"AMOUNT": {"sdtype": "numerical"}, "STATUS_CD": {"sdtype": "categorical"}}}

    syn = TabSynSynthesizer(_FakeMeta(), epochs=2, d_token=8, d_latent=2, denoiser_hidden=16,
                            denoiser_depth=1, sample_steps=5)
    syn._set_random_state(0)
    syn.fit(df)
    out = syn.sample(25)
    _assert(list(out.columns) == list(df.columns), f"column order must match the input, got {list(out.columns)}")
    _assert(len(out) == 25, f"expected 25 sampled rows, got {len(out)}")
    _assert(set(out["STATUS_CD"].dropna().unique()) <= set(df["STATUS_CD"].unique()),
            "sampled categories must come from the real column's own vocabulary")
    _assert(pd.api.types.is_numeric_dtype(out["AMOUNT"]), "a numerical column must stay numeric, not object")


@check("TabSynSynthesizer: same seed (fit-time global + _set_random_state) reproduces the same sample")
def _c_tabsyn_reproducible():
    from .suite import _seed_global, _seed_synth
    from .tabsyn import TabSynSynthesizer

    rng = np.random.RandomState(1)
    n = 40
    df = pd.DataFrame({"AMOUNT": rng.normal(0, 1, n), "STATUS_CD": rng.choice(["A", "B"], n)})

    class _FakeMeta:
        def to_dict(self):
            return {"columns": {"AMOUNT": {"sdtype": "numerical"}, "STATUS_CD": {"sdtype": "categorical"}}}

    def run():
        _seed_global(7)
        syn = TabSynSynthesizer(_FakeMeta(), epochs=2, d_token=8, d_latent=2, denoiser_hidden=16,
                                denoiser_depth=1, sample_steps=5)
        syn.fit(df)
        _seed_synth(syn, 7)
        return syn.sample(20)

    out1, out2 = run(), run()
    _assert(out1.equals(out2), "same seed via the real suite.py seeding flow must reproduce identically")


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
