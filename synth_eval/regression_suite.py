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
from .compare import compute_summary, shapes_heatmap_data, structure_scores
from .efficacy import (InsufficientHoldoutError, _predictive_signal, _quasi_identifier_group_col,
                       auto_select_target, real_feature_importance, sdmetrics_ml_efficacy)
from .entity import (_normalize_key_name, _resolve_key_column, build_entity_hub,
                     derive_synthetic_hub_pool, entity_key_tables)
from .link import _normalize_key_values, _real_parent_counts, link_relationships, link_table
from .pii import apply_pii_plan, fake_series
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


def _small_many_class_fixture():
    """Real OCCUPATION.csv's own shape: 21 codes, ~6 rows each (131 total).
    An 80/20-ish split leaves several codes' entire small cluster on one
    side -- the real trigger for InsufficientHoldoutError, not a contrived
    edge case."""
    rng = np.random.default_rng(0)
    codes, other = [], []
    for i in range(21):
        codes += [f"C{i}"] * 6
        other += rng.choice(["A", "B", "C"], 6).tolist()
    df = pd.DataFrame({"CODE": codes, "OTHER_CD": other})
    train_real = df.iloc[:105].reset_index(drop=True)
    holdout_real = df.iloc[105:].reset_index(drop=True)
    roles = ColumnRoles(categorical=["CODE", "OTHER_CD"])
    return train_real, holdout_real, roles


@check("sdmetrics_ml_efficacy: raises InsufficientHoldoutError instead of scoring a table of NaNs")
def _c_sdmetrics_ml_efficacy_insufficient_holdout():
    train_real, holdout_real, roles = _small_many_class_fixture()
    synth = {"TVAE": train_real.copy()}
    try:
        out = sdmetrics_ml_efficacy(train_real, holdout_real, synth, roles, "CODE",
                                    "classification", table_name="OCCUPATION")
        raise AssertionError(f"expected InsufficientHoldoutError, got a frame instead:\n{out}")
    except InsufficientHoldoutError as e:
        _assert("too few to score reliably" in str(e), f"unexpected message: {e}")


@check("sdmetrics_ml_efficacy: still scores normally when the holdout shares enough classes")
def _c_sdmetrics_ml_efficacy_normal_case_unaffected():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"CODE": rng.choice(["A", "B"], n), "OTHER_CD": rng.choice(["X", "Y", "Z"], n)})
    train_real, holdout_real = df.iloc[:1600].reset_index(drop=True), df.iloc[1600:].reset_index(drop=True)
    roles = ColumnRoles(categorical=["CODE", "OTHER_CD"])
    out = sdmetrics_ml_efficacy(train_real, holdout_real, {"TVAE": train_real.copy()},
                                roles, "CODE", "classification", table_name="BIGTABLE")
    _assert(not out.empty and out["score"].notna().all(),
            f"expected a normally-scored, non-empty frame, got:\n{out}")


@check("real_feature_importance: ranks the genuinely predictive column above a pure-noise one")
def _c_real_feature_importance_ranks_signal():
    rng = np.random.default_rng(0)
    n = 2000
    skill = rng.choice(["LOW", "HIGH"], n)
    # FLAG genuinely depends on SKILL; NOISE_CD carries no information at all
    flag = [("Y" if (s == "HIGH" and rng.random() < 0.85) else
             ("Y" if (s == "LOW" and rng.random() < 0.10) else "N")) for s in skill]
    df = pd.DataFrame({"SKILL_LEVEL_CD": skill, "NOISE_CD": rng.choice(["A", "B", "C"], n), "FLAG": flag})
    roles = ColumnRoles(categorical=["SKILL_LEVEL_CD", "NOISE_CD", "FLAG"])
    ranked = real_feature_importance(df, roles, "FLAG", "classification")
    _assert(ranked is not None and ranked[0][0] == "SKILL_LEVEL_CD",
            f"expected SKILL_LEVEL_CD (the genuinely predictive column) ranked first, got {ranked}")
    total = sum(v for _, v in ranked)
    _assert(abs(total - 1.0) < 1e-6, f"importances should sum to ~1 (sklearn's own convention), got {total:.3f}")


@check("real_feature_importance: aggregates one-hot slots back to the ORIGINAL column, not fragments")
def _c_real_feature_importance_aggregates_onehot():
    rng = np.random.default_rng(1)
    n = 1500
    cat = rng.choice(["A", "B", "C", "D"], n)   # 4 one-hot slots once encoded
    target = np.where(cat == "A", "Y", "N")     # fully determined by CAT_CD
    df = pd.DataFrame({"CAT_CD": cat, "OTHER_CD": rng.choice(["X", "Y"], n), "TARGET": target})
    roles = ColumnRoles(categorical=["CAT_CD", "OTHER_CD", "TARGET"])
    ranked = real_feature_importance(df, roles, "TARGET", "classification")
    cols = [c for c, _ in ranked]
    _assert(cols.count("CAT_CD") == 1,
            f"expected ONE aggregated 'CAT_CD' entry, not one per one-hot value, got {cols}")


@check("real_feature_importance: not enough rows or no feature columns returns None, not a crash")
def _c_real_feature_importance_none_when_unusable():
    tiny = pd.DataFrame({"CD": ["A", "B"], "TARGET": ["Y", "N"]})
    roles = ColumnRoles(categorical=["CD", "TARGET"])
    _assert(real_feature_importance(tiny, roles, "TARGET", "classification") is None,
            "too few rows to fit reliably must return None, not raise or fabricate a ranking")

    n = 100
    df = pd.DataFrame({"TARGET": np.random.default_rng(0).choice(["Y", "N"], n)})
    roles2 = ColumnRoles(categorical=["TARGET"])   # no OTHER feature columns at all
    _assert(real_feature_importance(df, roles2, "TARGET", "classification") is None,
            "a target with no feature columns to predict FROM must return None")


def _versioned_dimension_fixture():
    """A reference/dimension table shaped like OCCUPATION.csv: CODE is the
    entity's own versioning key (multiple history-rows per code), CATEGORY
    is a static per-code attribute (perfectly constant within a code, no
    real relationship to anything else), SKILL genuinely varies WITHIN a
    code and has a real, generalizable relationship to FLAG."""
    codes, category, skill, flag = [], [], [], []
    rng = np.random.default_rng(0)
    for i in range(15):
        rows = 6
        codes += [f"C{i}"] * rows
        category += [f"CAT{i % 4}"] * rows          # constant per code -- the leak
        sk = rng.choice(["LOW", "HIGH"], rows)
        skill += list(sk)
        flag += [("Y" if (s == "HIGH" and rng.random() < 0.8) else
                  ("Y" if (s == "LOW" and rng.random() < 0.1) else "N")) for s in sk]
    return pd.DataFrame({"CODE": codes, "CATEGORY": category, "SKILL": skill, "FLAG": flag})


@check("_quasi_identifier_group_col: finds the entity key when it near-determines the target")
def _c_quasi_identifier_detects_entity_key():
    df = _versioned_dimension_fixture()
    roles = ColumnRoles(categorical=["CODE", "CATEGORY", "SKILL", "FLAG"])
    feature_roles = ColumnRoles(categorical=["CODE", "SKILL", "FLAG"])
    gcol = _quasi_identifier_group_col(df, "CATEGORY", feature_roles)
    _assert(gcol == "CODE", f"CATEGORY is constant per CODE (the classic SCD leak) -- expected 'CODE', got {gcol}")


@check("_quasi_identifier_group_col: finds nothing for a target with no near-deterministic feature")
def _c_quasi_identifier_none_for_real_signal():
    df = _versioned_dimension_fixture()
    feature_roles = ColumnRoles(categorical=["CODE", "CATEGORY", "SKILL"])
    gcol = _quasi_identifier_group_col(df, "FLAG", feature_roles)
    _assert(gcol is None, f"FLAG has no near-deterministic feature (only a real, partial SKILL "
                          f"relationship) -- expected None, got {gcol}")


@check("_predictive_signal: entity-aware split rejects a target only 'predictable' via code memorization")
def _c_predictive_signal_rejects_memorization():
    df = _versioned_dimension_fixture()
    feature_roles = ColumnRoles(categorical=["CODE", "SKILL", "FLAG"])
    sig = _predictive_signal(df, "CATEGORY", feature_roles, "classification")
    _assert(sig is not None, "expected a result, not None (enough rows/classes here)")
    real, floor = sig
    lift = real - floor
    _assert(lift < 0.05,
            f"CATEGORY is only 'predictable' by memorizing CODE (constant per code) -- a random row "
            f"split lets that leak through; expected the entity-aware split to bring lift near zero, "
            f"got real={real:.3f} floor={floor:.3f} lift={lift:.3f}")


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


@check("derive_synthetic_hub_pool: keeps the union as-is when it's a real, learned vocabulary")
def _c_hub_pool_keeps_real_vocabulary():
    # a *_TP_CD-style code column: each child's own model can only ever
    # reproduce a value it saw during training, so the union across children
    # IS the real vocabulary -- must be left untouched
    real_ids = pd.Series(["A", "B", "C", "D", "E"])
    children_values = {
        "T1": pd.Series(["A", "A", "B", "C"]),
        "T2": pd.Series(["B", "D", "E", "A"]),
    }
    pool = derive_synthetic_hub_pool(children_values, real_ids, "CD")
    _assert(set(pool) == {"A", "B", "C", "D", "E"},
            f"high-overlap union must be kept as-is, got {sorted(pool)}")


@check("derive_synthetic_hub_pool: resizes to the real entity count when the union is fabricated noise")
def _c_hub_pool_resizes_fabricated_ids():
    # an id-typed surrogate key: each child fabricates a fresh, fully-unique
    # value per row with ZERO overlap with real identity or with the other
    # children's own fabrication -- the union is uninformative noise whose
    # only useful property was ever its SIZE, which is wrong (inflated
    # toward the sum of the children's row counts, not the real entity count)
    real_ids = pd.Series(range(50))            # 50 real distinct entities
    children_values = {
        "T1": pd.Series([f"fake_t1_{i}" for i in range(80)]),   # 80 fabricated, unique
        "T2": pd.Series([f"fake_t2_{i}" for i in range(60)]),   # 60 fabricated, unique
    }
    pool = derive_synthetic_hub_pool(children_values, real_ids, "CD")
    _assert(len(pool) == 50, f"expected the pool resized to the real entity count (50), got {len(pool)}")
    _assert(not (set(pool) & set(real_ids)), "resized pool should use fresh placeholders, not real ids")


@check("derive_synthetic_hub_pool: an empty real hub is a no-op (nothing to resize against)")
def _c_hub_pool_empty_real_noop():
    real_ids = pd.Series([], dtype=object)
    children_values = {"T1": pd.Series(["x", "y"])}
    pool = derive_synthetic_hub_pool(children_values, real_ids, "CD")
    _assert(set(pool) == {"x", "y"}, f"no real ids to compare against -- union should pass through, got {sorted(pool)}")


@check("derive_synthetic_hub_pool: a wide code column with PARTIAL real coverage is still kept, not resized")
def _c_hub_pool_partial_recall_still_kept():
    # Regression for a real bug: recall-based overlap (does the union cover
    # the FULL real vocabulary?) wrongly flagged a genuinely-learned 220-code
    # column as "fabricated" just because two child tables' worth of rows
    # didn't happen to reproduce every single code -- every value the model
    # DID emit was still a real one (100% precision), which is what actually
    # distinguishes a real vocabulary from fabricated noise. Confirmed live:
    # this turned every PERSON.OCCUPATION_TP_CD value into a meaningless
    # "OCCUPATION_TP_CD__synth_N" placeholder on a column that previously
    # matched real values exactly.
    real_ids = pd.Series([f"CODE{i}" for i in range(220)])
    rng = np.random.default_rng(0)
    # each child only reproduces a subset of the 220 codes (a realistic
    # generation pattern for a long-tail categorical) -- union recall is
    # well under 50%, but every emitted value is still a real code
    children_values = {
        "OCCUPATION": pd.Series(rng.choice(real_ids[:90], size=300)),
        "PERSON": pd.Series(rng.choice(real_ids[:70], size=500)),
    }
    union_recall = len(set().union(*[set(v) for v in children_values.values()])) / len(real_ids)
    _assert(union_recall < 0.5, f"fixture setup check: expected recall < 0.5, got {union_recall:.2f}")
    pool = derive_synthetic_hub_pool(children_values, real_ids, "OCCUPATION_TP_CD")
    _assert(all(not str(v).endswith(tuple(f"__synth_{i}" for i in range(220))) for v in pool),
            f"a real (if incomplete) vocabulary must never be replaced with fabricated placeholders, got sample {sorted(pool)[:5]}")
    _assert(set(pool) <= set(real_ids), f"pool must stay within the real vocabulary, got {sorted(pool)[:5]}")


def _skewed_code_fixture(n_codes: int = 30, seed: int = 0):
    """A handful of very common codes (e.g. common occupations) plus many
    rare ones -- the shape a real *_TP_CD column's own popularity usually
    takes. Used to prove link_table keeps which SPECIFIC code is common,
    not just the aggregate shape of the popularity distribution."""
    rng = np.random.default_rng(seed)
    pop = np.concatenate([[3000, 2000, 1000], rng.integers(5, 60, size=n_codes - 3)]).astype(float)
    codes = [f"C{i}" for i in range(n_codes)]
    child_codes = np.repeat(codes, pop.astype(int))
    rng.shuffle(child_codes)
    real_child = pd.DataFrame({"CD": child_codes})
    real_parent = pd.DataFrame({"CD": codes})
    return real_parent, real_child, codes


@check("link_table: keeps WHICH code is popular, not just the shape of the popularity distribution")
def _c_link_table_preserves_value_popularity():
    # Before the value-matching fix, link_table bootstrap-sampled a count
    # independently of which synthetic parent key it landed on: the
    # AGGREGATE distribution of counts matched real (good
    # CardinalityShapeSimilarity) but WHICH code got which count was
    # scrambled, wrecking that column's own marginal frequency (Column
    # Shapes) even though referential integrity looked fine. Real-world
    # trigger: an occupation code, PERSON.OCCUPATION_TP_CD, showing solid
    # red on Column Shapes for every synthesizer even after the sdtype fix,
    # because relinking (not the model) scrambled it.
    real_parent, real_child, codes = _skewed_code_fixture()
    real_counts = _real_parent_counts(real_parent, real_child, "CD", "CD")
    # the synthesizer's OWN pre-relink guess: it got the vocabulary right
    # (same real codes) but has no reason to know their relative popularity
    # -- uniform-random, the worst case for the hybrid rank-swap+rebalance
    # to still get value-popularity right regardless of what it started from
    rng = np.random.default_rng(2)
    child_df = pd.DataFrame({"CD": rng.choice(codes, size=len(real_child))})
    linked = link_table(child_df, "CD", codes, real_counts, seed=1)
    real_top3 = set(real_child["CD"].value_counts().head(3).index)
    linked_top3 = set(linked["CD"].value_counts().head(3).index)
    _assert(real_top3 == linked_top3,
            f"the 3 truly most common real codes {real_top3} must still be the 3 most common "
            f"after relinking, got {linked_top3} -- popularity got scrambled")


@check("link_table: matches real/synthetic keys by value even across an int/float dtype mismatch")
def _c_link_table_dtype_mismatch_still_matches():
    # *_TP_CD columns are frequently read from CSV as float64 (e.g. 348820.0)
    # while a synthesizer's own categorical decoder can emit a plain int for
    # the same value -- naive equality (or a naive str() cast) would treat
    # every key as "no match" and silently fall back to the old scrambling
    # behavior for 100% of keys, defeating the fix above without erroring.
    real_parent, real_child, codes = _skewed_code_fixture()
    real_parent["CD"] = real_parent["CD"].map(lambda c: 340000.0 + int(c[1:]))
    real_child["CD"] = real_child["CD"].map(lambda c: 340000.0 + int(c[1:]))
    real_counts = _real_parent_counts(real_parent, real_child, "CD", "CD")
    synth_keys = [int(v) for v in real_parent["CD"]]   # dtype mismatch: int, not float64
    rng = np.random.default_rng(2)
    child_df = pd.DataFrame({"CD": rng.choice(synth_keys, size=len(real_child))})
    linked = link_table(child_df, "CD", synth_keys, real_counts, seed=1)
    real_top3 = set(real_child["CD"].value_counts().head(3).index.astype(int))
    linked_top3 = set(linked["CD"].value_counts().head(3).index)
    _assert(real_top3 == linked_top3,
            f"expected the dtype-normalized match to still find the 3 real most-common codes "
            f"{real_top3}, got {linked_top3}")


@check("link_table: rank-swap preserves a row's OWN correlation with other columns, not just aggregate shape")
def _c_link_table_preserves_row_level_correlation():
    # A pure popularity-weighted reshuffle (the pre-hybrid approach) assigns
    # each row's new code independently of everything else about that row,
    # destroying any relationship the model's own joint fit learned between
    # this column and the REST of the row (e.g. occupation code vs gender).
    # The hybrid rank-swap relabels the model's OWN groupings instead of
    # reshuffling rows, so that relationship should survive, at least for
    # codes the model's own generation was reasonably close to real on.
    codes = [f"C{i}" for i in range(5)]
    pop = np.array([500, 400, 300, 200, 150], dtype=float)
    real_child = pd.DataFrame({"CD": np.repeat(codes, pop.astype(int))})
    real_parent = pd.DataFrame({"CD": codes})
    real_counts = _real_parent_counts(real_parent, real_child, "CD", "CD")

    # model's own pre-relink generation: SAME counts as real (so rebalancing
    # barely has to move anything), each code strongly paired with GENDER
    rng = np.random.default_rng(3)
    model_codes, gender = [], []
    for i, c in enumerate(codes):
        cnt = int(pop[i])
        fem_p = 0.9 if i % 2 == 0 else 0.1
        model_codes += [c] * cnt
        gender += list(rng.choice(["F", "M"], size=cnt, p=[fem_p, 1 - fem_p]))
    child_df = pd.DataFrame({"CD": model_codes, "GENDER": gender})

    linked = link_table(child_df, "CD", codes, real_counts, seed=1)
    linked["GENDER"] = child_df["GENDER"].to_numpy()   # same row order/index throughout
    fem_share = linked.groupby("CD")["GENDER"].apply(lambda s: (s == "F").mean())
    for i, c in enumerate(codes):
        expected_high = i % 2 == 0
        got_high = fem_share[c] >= 0.5
        _assert(got_high == expected_high,
                f"{c}: expected {'mostly F' if expected_high else 'mostly M'} to survive relinking "
                f"(model's own count matched real here, so rebalancing barely touches it), "
                f"got F share {fem_share[c]:.2f}")


@check("link_table: largest-remainder apportionment doesn't systematically undercount weight-1 keys")
def _c_link_table_apportionment_not_undercounted():
    # floor(weight * n/total), the old target computation, zeroes out EVERY
    # key whose real weight is exactly 1 (the common case for a near-unique
    # key, e.g. a surrogate id) any time the resampled total lands even
    # slightly above n -- the ordinary case, not an edge case. A random
    # weighted top-up then only restores a random SUBSET of those zeroed
    # keys. Real-world trigger: a 998-key hub where 84% of real keys have
    # exactly 1 child -- the old code covered only 53% of keys after
    # relinking; largest-remainder apportionment should land much closer to
    # the real 84%, deterministically (not by lucky redraw).
    n_keys = 200
    codes = [f"K{i}" for i in range(n_keys)]
    rng = np.random.default_rng(0)
    # 84% of real parents have exactly 1 child, the rest have 0 -- mirrors
    # the real CONT_ID-style key shape that exposed the bug
    has_child = rng.random(n_keys) < 0.84
    real_child = pd.DataFrame({"CD": [c for c, h in zip(codes, has_child) if h]})
    real_parent = pd.DataFrame({"CD": codes})
    real_counts = _real_parent_counts(real_parent, real_child, "CD", "CD")

    # synthetic parent pool is the SAME size as the real one (this test
    # isolates the rounding bug, not the separate pool-sizing bug) but the
    # keys are fabricated placeholders with zero value-overlap with real --
    # forces every weight through the random-fallback + apportionment path
    synth_keys = [f"SYNTH{i}" for i in range(n_keys)]
    child_df = pd.DataFrame({"CD": rng.choice(synth_keys, size=len(real_child), replace=False)})
    n_covered_real = int(has_child.sum())
    linked = link_table(child_df, "CD", synth_keys, real_counts, seed=7)
    covered = pd.Series(synth_keys).isin(set(linked["CD"])).sum()
    real_ratio = n_covered_real / n_keys
    got_ratio = covered / n_keys
    _assert(got_ratio >= real_ratio - 0.10,
            f"expected apportionment to land within ~10pts of the real coverage ratio "
            f"({real_ratio:.2f}), got {got_ratio:.2f} ({covered}/{n_keys} keys covered) -- "
            f"floor-based rounding would land far below this")


@check("_normalize_key_values: 348820.0 and 348820 normalize to the same value")
def _c_normalize_key_values_numeric():
    out = _normalize_key_values([348820.0, 348820, "C0", "C0"])
    _assert(out[0] == out[1], f"float and int forms of the same number must normalize equal, got {out}")
    _assert(out[2] == out[3], "identical strings must normalize equal")
    _assert(out[0] != out[2], "a numeric value and a text code must not collide")


@check("structure_scores: cardinality_shape_baseline is attached per synth when given")
def _c_structure_scores_baseline_attached():
    # cardinality_report's own end-to-end behavior (a real holdout scoring
    # below 1.0 on a lopsided parent-child fan-out, exactly the OCCUPATION-
    # shaped production case this was built for) is validated by hand against
    # real sdmetrics calls, not re-run here (heavy: needs sdv+sdmetrics) --
    # this only checks structure_scores' OWN plumbing: the single baseline
    # float reaches every synth's row, unchanged from what was passed in.
    ri_rows = [
        {"relationship": "CHILD.CODE -> PARENT.CODE", "source": "real",
         "fk_coverage": 1.0, "parent_coverage": 0.9},
        {"relationship": "CHILD.CODE -> PARENT.CODE", "source": "SomeSynth",
         "fk_coverage": 1.0, "parent_coverage": 0.85},
    ]
    cardinality = {"SomeSynth": {"shape": 0.5, "statistic": 0.6}}
    out = structure_scores(ri_rows, cardinality, derived_parent=False, cardinality_baseline=0.75)
    _assert(out["SomeSynth"]["cardinality_shape_baseline"] == 0.75,
            f"expected the baseline passed in to be attached verbatim, got {out['SomeSynth']}")
    _assert(out["SomeSynth"]["cardinality_shape"] == 0.5, "the synth's own score must be untouched")


@check("structure_scores: cardinality_shape_baseline is None when not given (backward compatible)")
def _c_structure_scores_baseline_defaults_none():
    ri_rows = [
        {"relationship": "CHILD.CODE -> PARENT.CODE", "source": "real",
         "fk_coverage": 1.0, "parent_coverage": 0.9},
        {"relationship": "CHILD.CODE -> PARENT.CODE", "source": "SomeSynth",
         "fk_coverage": 1.0, "parent_coverage": 0.85},
    ]
    cardinality = {"SomeSynth": {"shape": 0.5, "statistic": 0.6}}
    out = structure_scores(ri_rows, cardinality, derived_parent=False)
    _assert(out["SomeSynth"]["cardinality_shape_baseline"] is None,
            "a caller that doesn't pass a baseline must not see one appear from nowhere")


@check("compute_summary: NewRowSynthesis near a low real-holdout ceiling doesn't tank the privacy score")
def _c_privacy_score_judges_newrow_against_baseline():
    # Regression for a real case: a reference table with few distinct value
    # combinations has a genuinely low achievable NewRowSynthesis ceiling
    # even for real, unseen rows (0.087 in the live case that triggered
    # this) -- a synthetic score of 0.000 there is WARN (0.087 below
    # ceiling), not a real problem. The composite privacy score must judge
    # it the same way the verdict does (gap from the ceiling), not average
    # in the raw 0.000 as if the ceiling were 1.0.
    quality_scores = {"S": {"T": {"overall": 0.9}}}
    privacy_all = {"S": {"T": {
        "membership_inference": {"auc": 0.5},
        "sdmetrics": {"NewRowSynthesis": 0.0, "NewRowSynthesis_baseline": 0.087,
                      "CategoricalCAP": 0.9, "CategoricalCAP_baseline": 0.9},
        "nearest_record_examples": {},
    }}}
    out = compute_summary(quality_scores, privacy_all, None)
    privacy = out["S"]["privacy"]["score"]
    _assert(privacy > 0.85,
            f"expected the near-ceiling NewRowSynthesis (gap=0.087) to barely dent the privacy "
            f"score, got {privacy:.3f} -- the raw 0.000 must not be averaged in directly")
    _assert(out["S"]["privacy"]["new_row_synthesis"] == 0.0,
            "the DISPLAYED raw new_row_synthesis number must stay the true raw score, "
            "only the composite score should be baseline-adjusted")


@check("compute_summary: no baseline falls back to the raw NewRowSynthesis/CategoricalCAP score")
def _c_privacy_score_no_baseline_uses_raw():
    quality_scores = {"S": {"T": {"overall": 0.9}}}
    privacy_all = {"S": {"T": {
        "membership_inference": {"auc": 0.5},
        "sdmetrics": {"NewRowSynthesis": 0.4, "CategoricalCAP": 0.6},
        "nearest_record_examples": {},
    }}}
    out = compute_summary(quality_scores, privacy_all, None)
    # mean(mia_prot=1.0, new_row=0.4, cap=0.6, nearest=nan->dropped) == mean(1.0, 0.4, 0.6)
    _assert(abs(out["S"]["privacy"]["score"] - (1.0 + 0.4 + 0.6) / 3) < 1e-9,
            f"without a baseline the composite must fall back to the raw scores, got {out['S']['privacy']['score']:.3f}")


@check("shapes_heatmap_data: carries an error reason for an unscoreable (not just unevaluated) cell")
def _c_shapes_heatmap_data_carries_errors():
    # a blank Column Shapes cell can mean "not evaluated" or "sdmetrics tried
    # and couldn't compute a similarity at all" (e.g. IncomputableMetricError
    # on a sparse real column whose synthetic side came back 100% null) --
    # the two must not collapse into the same unexplained blank cell
    shape_scores = {
        "TVAE": pd.Series({"COL_A": 0.9, "COL_B": float("nan")}),
        "TabSyn": pd.Series({"COL_A": 0.8, "COL_B": 0.3}),
    }
    shape_errors = {"TVAE": {"COL_B": "IncomputableMetricError: ... 1 or more non-null values."}}
    out = shapes_heatmap_data(shape_scores, shape_errors)
    i, j = out["y"].index("TVAE"), out["x"].index("COL_B")
    _assert(out["z"][i][j] is None, "an uncomputable cell must still be a null score, not a fake number")
    _assert(out["err"][i][j] is not None and "Incomputable" in out["err"][i][j],
            f"expected the error reason to reach the same cell position, got {out['err'][i][j]}")
    i2, j2 = out["y"].index("TabSyn"), out["x"].index("COL_A")
    _assert(out["err"][i2][j2] is None, "a normally-scored cell must not carry a stray error")


@check("shapes_heatmap_data: no shape_errors given still returns a same-shape all-None err matrix")
def _c_shapes_heatmap_data_no_errors():
    shape_scores = {"TVAE": pd.Series({"COL_A": 0.9})}
    out = shapes_heatmap_data(shape_scores, None)
    _assert(out["err"] == [[None]], f"expected an all-None err matrix when no errors are given, got {out['err']}")


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
# PII faking (synth_eval.pii) -- entity-consistent fake values on an
# SCD-versioned table (several history rows per real customer)
# ---------------------------------------------------------------------------

@check("fake_series: rows sharing a group id get the SAME fake value and missing status")
def _c_fake_series_entity_consistent():
    # mirrors a real customer with several SCD-versioned CONTACT rows: same
    # CONT_ID repeated, real name is either always null or always the same
    # non-null value for that customer -- the fake name must follow that
    # same shape, not re-roll independently per row (confirmed live: without
    # this, the same real customer showed a different fake name per row)
    group_ids = pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3])
    like = pd.Series(["Real A", "Real A", "Real A", None, None, "Real C", "Real C", "Real C", "Real C"])
    out = fake_series("name", len(group_ids), like=like, seed=0, column_name="CONTACT_NAME",
                      group_ids=group_ids)
    for g in group_ids.unique():
        vals = out[group_ids == g]
        _assert(vals.nunique(dropna=False) == 1,
                f"entity {g}'s rows must all get the SAME fake value (or all-NaN), got {vals.tolist()}")


@check("fake_series: a row with no group id still gets an independent draw, not crash or share")
def _c_fake_series_ungrouped_row_independent():
    group_ids = pd.Series([1, 1, np.nan, np.nan])
    out = fake_series("name", len(group_ids), like=None, seed=0, column_name="CONTACT_NAME",
                      group_ids=group_ids)
    _assert(out.notna().all(), f"expected 4 non-null fake names (no missing rate given), got {out.tolist()}")
    _assert(out.iloc[0] == out.iloc[1], "entity 1's two rows must share the same fake value")
    _assert(len({out.iloc[2], out.iloc[3]}) == 2,
            "two DIFFERENT ungrouped rows must not be forced to share a value")


@check("fake_series: without group_ids, behaves exactly as before (row-independent)")
def _c_fake_series_no_group_ids_unchanged():
    out = fake_series("name", 20, like=None, seed=0, column_name="CONTACT_NAME")
    _assert(out.nunique() > 1, "row-independent faking (no group_ids) must still vary row to row")


@check("apply_pii_plan: threads group_col through so a table's own entity key ties fakes together")
def _c_apply_pii_plan_entity_consistent():
    df = pd.DataFrame({
        "CONT_ID": [10, 10, 20, 20, 20],
        "CONTACT_NAME": ["x", "y", "z", "w", "v"],   # whatever the refill bootstrap put here
    })
    real = pd.DataFrame({"CONT_ID": [10, 20], "CONTACT_NAME": ["Real A", "Real B"]})
    out = apply_pii_plan(df, {"CONTACT_NAME": ("fake", "name")}, real, seed=0, group_col="CONT_ID")
    _assert(out.loc[out["CONT_ID"] == 10, "CONTACT_NAME"].nunique() == 1,
            "CONT_ID 10's two rows must share one faked name")
    _assert(out.loc[out["CONT_ID"] == 20, "CONTACT_NAME"].nunique() == 1,
            "CONT_ID 20's three rows must share one faked name")


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


@check("TabSynSynthesizer: nhead not dividing d_token falls back to a valid divisor instead of crashing")
def _c_tabsyn_nhead_autocorrect():
    from .tabsyn import TabSynSynthesizer

    class _FakeMeta:
        def to_dict(self):
            return {"columns": {"AMOUNT": {"sdtype": "numerical"}}}

    # 48 % 8 == 0 (fine); a caller-supplied combo like d_token=50, nhead=8
    # does NOT divide evenly -- nn.TransformerEncoderLayer would raise
    # instead of silently doing something wrong, so this must self-correct
    # at construction time, before fit() ever builds the model.
    syn = TabSynSynthesizer(_FakeMeta(), d_token=50, nhead=8)
    _assert(50 % syn.nhead == 0, f"nhead={syn.nhead} must evenly divide d_token=50")
    _assert(syn.nhead <= 8, f"expected the largest valid divisor <= the requested 8, got {syn.nhead}")


@check("build_single_table_synthesizer: threads tabsyn_params through, ignores them for other synths")
def _c_build_single_table_tabsyn_params():
    from .suite import build_single_table_synthesizer

    class _FakeMeta:
        def to_dict(self):
            return {"columns": {"AMOUNT": {"sdtype": "numerical"}}}

    syn = build_single_table_synthesizer("TabSyn", _FakeMeta(), epochs=5,
                                          tabsyn_params={"d_token": 16, "d_latent": 4})
    _assert(syn.d_token == 16 and syn.d_latent == 4,
            f"tabsyn_params must override the constructor defaults, got d_token={syn.d_token} d_latent={syn.d_latent}")
    # a non-TabSyn synth must not choke on an irrelevant tabsyn_params -- it's
    # a run-level config field, not something every caller filters per-synth.
    # GaussianCopulaSynthesizer validates its metadata at construction time
    # (unlike TabSyn's own duck-typed _FakeMeta above), so this needs a real
    # SDV SingleTableMetadata rather than the stub.
    from sdv.metadata import SingleTableMetadata
    real_meta = SingleTableMetadata()
    real_meta.add_column("AMOUNT", sdtype="numerical")
    gc = build_single_table_synthesizer("GaussianCopula", real_meta,
                                         tabsyn_params={"d_token": 16})
    _assert(gc is not None, "tabsyn_params must be silently ignored by non-TabSyn synthesizers")


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
