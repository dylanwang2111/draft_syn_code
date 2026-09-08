"""synth_eval.cross_table — entity-level cross-table correlation fidelity.

sdmetrics' single-table Column Pair Trends only see column pairs *inside one
table*, and its multi-table ``Intertable Trends`` only measures a **parent's**
columns against a child's — useless for a keys-only entity hub whose parent has
no attributes.  Neither answers the question that actually matters for
entity-hub / SCD data: *for a given customer, do attributes across different
tables hang together the way they do in real data?* (e.g. marital status in
PERSON vs province in CONTACT).

This module measures exactly that.  For every pair of tables that share the
entity key, each table is collapsed to **one row per entity** (its current SCD
version), the two are joined on the key, and every cross-table column pair is
scored real-vs-synth with the same primitives Column Pair Trends uses —
``ContingencySimilarity`` (categorical / mixed) and ``CorrelationSimilarity``
(numeric).  1 = the cross-table relationship is preserved.

A synthesizer that does not keep the entity key consistent across tables
(independent single-table synthesis regenerates ids per table) produces almost
no matched entities on the join; that table pair is reported as **unaligned
(n/a)** rather than scored — it is not that the correlation is bad, it is that
there is no shared entity to correlate on.  Only a model that preserves the key
across tables (HMA on the hub) can be scored here at all.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity


def _current_version(df: pd.DataFrame, entity_key: str,
                     current_flag: Optional[str], effective_col: Optional[str]) -> pd.DataFrame:
    """One row per entity: the flagged current row, else latest by effective
    date, else the last row seen.  The same rule is used for real and synth."""
    if entity_key not in df.columns:
        return df
    if df.groupby(entity_key, dropna=False).size().max() <= 1:
        return df                                   # already one row per entity
    if current_flag and current_flag in df.columns:
        flagged = df[df[current_flag].astype(str).str.upper().isin({"Y", "TRUE", "1", "T"})]
        picked = flagged.groupby(entity_key, dropna=False, as_index=False).head(1)
        got = set(picked[entity_key])
        missing = df[~df[entity_key].isin(got)].groupby(entity_key, dropna=False, as_index=False).tail(1)
        return pd.concat([picked, missing], ignore_index=True)
    if effective_col and effective_col in df.columns:
        eff = pd.to_datetime(df[effective_col], errors="coerce", format="mixed")
        order = df.assign(_eff=eff).sort_values("_eff", na_position="first")
        return order.groupby(entity_key, dropna=False, as_index=False).tail(1).drop(columns="_eff")
    return df.groupby(entity_key, dropna=False, as_index=False).tail(1)


def _similarity(dfr: pd.DataFrame, dfs: pd.DataFrame, a: str, b: str,
                a_num: bool, b_num: bool) -> Optional[float]:
    """sdmetrics pair similarity of the (a, b) joint between two frames."""
    try:
        if a_num and b_num:
            v = CorrelationSimilarity.compute(dfr[[a, b]], dfs[[a, b]])
        else:
            cont = [c for c, is_num in ((a, a_num), (b, b_num)) if is_num]
            v = ContingencySimilarity.compute(dfr[[a, b]], dfs[[a, b]],
                                              continuous_column_names=cont or None)
    except Exception:
        return None
    return None if v is None or np.isnan(v) else float(np.clip(v, 0.0, 1.0))


def _pair_score(real2: pd.DataFrame, synth2: pd.DataFrame, a: str, b: str,
                a_num: bool, b_num: bool):
    """Returns (score, real_assoc) for one cross-table pair.

    ``score`` = how well the synth reproduces the real joint (1 = identical).
    ``real_assoc`` = how correlated the pair is in the REAL data, measured as
    the drop when the pairing is shuffled away (0 = independent, 1 = fully
    determined) — same primitive, so the two numbers are on one scale.  A pair
    with real_assoc ≈ 0 carries no signal: every model scores high on it, which
    is why the report ranks the tab by this.
    """
    r = real2[[a, b]].dropna()
    s = synth2[[a, b]].dropna()
    if len(r) < 10 or len(s) < 10:
        return None, None
    score = _similarity(r, s, a, b, a_num, b_num)
    if score is None:
        return None, None
    broken = r.copy()
    broken[b] = np.random.RandomState(0).permutation(broken[b].to_numpy())
    sim_shuf = _similarity(r, broken, a, b, a_num, b_num)
    real_assoc = None if sim_shuf is None else float(np.clip(1.0 - sim_shuf, 0.0, 1.0))
    return score, real_assoc


def entity_cross_table_trends(
    real_tables: Dict[str, pd.DataFrame],
    synth_tables: Dict[str, pd.DataFrame],
    entity_key: str,
    roles_by_table: Dict[str, "object"],
    current_flags: Optional[Dict[str, str]] = None,
    effective_by_table: Optional[Dict[str, str]] = None,
    min_matched: int = 20,
    align_frac: float = 0.5,
    strong_assoc: float = 0.10,
) -> dict:
    """Entity-level cross-table correlation fidelity for one synthesizer.

    Returns ``{score, n_pairs, n_entities, pairs, unaligned, note}`` where
    ``score`` is the mean cross-table pair score (``None`` when nothing could be
    scored) and ``pairs`` is a per-pair list for the heatmap.
    """
    current_flags = current_flags or {}
    effective_by_table = effective_by_table or {}
    out = {"score": None, "score_strong": None, "n_pairs": 0, "n_strong": 0,
           "n_entities": 0, "pairs": [], "unaligned": [], "note": None}

    if not entity_key:
        out["note"] = "no entity key — cross-table correlation is not defined"
        return out
    usable = [t for t in real_tables
              if entity_key in real_tables[t].columns
              and t in synth_tables and entity_key in synth_tables[t].columns
              and len(getattr(roles_by_table.get(t), "modelable", []) or [])]
    if len(usable) < 2:
        out["note"] = "fewer than two tables share the entity key"
        return out

    # collapse every usable table to one row per entity, real and synth alike
    def collapse(tables):
        red = {}
        for t in usable:
            roles = roles_by_table[t]
            cols = [c for c in roles.modelable if c in tables[t].columns]
            cur = _current_version(tables[t], entity_key, current_flags.get(t),
                                   effective_by_table.get(t))
            keep = [entity_key] + [c for c in cols if c != entity_key]
            red[t] = cur[keep].drop_duplicates(subset=entity_key)
        return red
    real_red, synth_red = collapse(real_tables), collapse(synth_tables)

    pair_scores: List[float] = []
    matched_entities = 0
    for ta, tb in combinations(usable, 2):
        ra = roles_by_table[ta]; rb = roles_by_table[tb]
        # join the two tables on the shared entity, prefixing so columns are unique
        def join(red):
            A = red[ta].add_prefix(f"{ta}."); B = red[tb].add_prefix(f"{tb}.")
            return A.merge(B, left_on=f"{ta}.{entity_key}", right_on=f"{tb}.{entity_key}", how="inner")
        real2, synth2 = join(real_red), join(synth_red)
        # A single-table synth regenerates the entity key per table, so its rows
        # only join by coincidental id collisions — the "matched" entities are
        # unrelated rows paired at random.  Require the synth to preserve a real
        # share of the entity linkage real itself has; otherwise the join is
        # spurious and this pair is reported unaligned, not scored.
        real_matched = len(real2)
        if len(synth2) < min_matched or (real_matched and len(synth2) < align_frac * real_matched):
            out["unaligned"].append({"table1": ta, "table2": tb,
                                     "matched": int(len(synth2)), "real": int(real_matched)})
            continue
        matched_entities = max(matched_entities, len(synth2))
        num_a = set(ra.numeric); num_b = set(rb.numeric)
        acols = [c for c in ra.modelable if c != entity_key]
        bcols = [c for c in rb.modelable if c != entity_key]
        for a in acols:
            for b in bcols:
                pa, pb = f"{ta}.{a}", f"{tb}.{b}"
                if pa not in real2 or pb not in real2:
                    continue
                sc, assoc = _pair_score(real2, synth2, pa, pb, a in num_a, b in num_b)
                if sc is None:
                    continue
                pair_scores.append(sc)
                out["pairs"].append({"table1": ta, "col1": a, "table2": tb, "col2": b,
                                     "score": sc, "real_assoc": assoc})

    if not pair_scores:
        if out["unaligned"] and not out["pairs"]:
            out["note"] = ("entity keys are not shared across tables "
                           "(independent single-table synthesis) — cross-table "
                           "correlation cannot be measured")
        else:
            out["note"] = "no scoreable cross-table column pairs"
        return out
    out["score"] = float(np.mean(pair_scores))
    out["n_pairs"] = len(pair_scores)
    out["n_entities"] = int(matched_entities)
    # the headline: preservation on the pairs that actually carry cross-table
    # signal (real_assoc >= strong_assoc).  On weakly-linked data this is empty
    # and the plain mean is all there is; where real structure exists this is
    # where a multi-table model separates from independent single-table output.
    strong = [p["score"] for p in out["pairs"]
              if p.get("real_assoc") is not None and p["real_assoc"] >= strong_assoc]
    if strong:
        out["score_strong"] = float(np.mean(strong))
        out["n_strong"] = len(strong)
    return out
