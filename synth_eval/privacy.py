"""synth_eval.privacy — membership inference, DCR, exact-match, sdmetrics privacy."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import plt, _save_fig, _single_table_metadata
from .columns import ColumnRoles, _fit_mixed_encoder, _encode


def membership_inference_attack(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    k: int = 5,
    random_state: int = 0,
) -> Dict[str, float]:
    """Distance-based Membership Inference Attack.

    Idea: if the synthesizer memorised training rows, training members will sit
    *closer* to the nearest synthetic records than fresh holdout rows do.  We
    build features = distances to the k nearest synthetic records for every
    real record (members + holdout), then train a classifier to tell members
    from non-members.  AUC ~ 0.5 => attacker cannot distinguish => good privacy.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    cols = roles.modelable
    result = {"auc": float("nan"), "accuracy": float("nan"), "n_members": len(train_real),
              "n_holdout": len(holdout_real), "note": ""}
    if not cols or len(holdout_real) < 10 or len(train_real) < 10 or len(synth) < 5:
        result["note"] = "insufficient data / columns for MIA"
        return result

    try:
        enc, use_cols = _fit_mixed_encoder(train_real, roles)
    except ValueError as e:
        result["note"] = str(e)
        return result
    S = _encode(enc, synth, use_cols)
    kk = int(min(k, len(S)))
    nn = NearestNeighbors(n_neighbors=kk).fit(S)

    def feats(df):
        d, _ = nn.kneighbors(_encode(enc, df, use_cols))
        # features: distance to each of the k nearest synth records + summary
        return np.hstack([d, d.mean(axis=1, keepdims=True), d.min(axis=1, keepdims=True)])

    Xm, Xh = feats(train_real), feats(holdout_real)
    # Balance classes by subsampling the larger group.
    n = min(len(Xm), len(Xh))
    rng = np.random.default_rng(random_state)
    mi = rng.choice(len(Xm), n, replace=False)
    hi = rng.choice(len(Xh), n, replace=False)
    X = np.vstack([Xm[mi], Xh[hi]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    result["auc"] = float(roc_auc_score(yte, proba))
    result["accuracy"] = float(accuracy_score(yte, (proba >= 0.5).astype(int)))
    return result


def dcr_distributions(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    out_png: str,
    sample: int = 2000,
    random_state: int = 0,
) -> Dict[str, object]:
    """Distance to Closest Record: real->synth vs real->real baseline + plot."""
    from sklearn.neighbors import NearestNeighbors

    cols = roles.modelable
    info: Dict[str, object] = {"note": ""}
    if not cols or len(synth) < 2 or len(real) < 3:
        info["note"] = "insufficient data for DCR"
        return info

    rng = np.random.default_rng(random_state)
    real_s = real.sample(min(sample, len(real)), random_state=random_state) if len(real) > sample else real

    try:
        enc, use_cols = _fit_mixed_encoder(real, roles)
    except ValueError as e:
        info["note"] = str(e)
        return info
    R = np.nan_to_num(_encode(enc, real_s, use_cols))
    S = np.nan_to_num(_encode(enc, synth, use_cols))
    Rall = np.nan_to_num(_encode(enc, real, use_cols))

    # real -> nearest synthetic
    d_rs, _ = NearestNeighbors(n_neighbors=1).fit(S).kneighbors(R)
    d_rs = d_rs.ravel()
    # real -> nearest OTHER real (baseline): 2 neighbours, drop self (distance 0)
    nn_rr = NearestNeighbors(n_neighbors=2).fit(Rall)
    d_rr, _ = nn_rr.kneighbors(R)
    d_rr = d_rr[:, 1]  # nearest non-self

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.percentile(np.concatenate([d_rs, d_rr]), 99) + 1e-9, 40)
    ax.hist(d_rr, bins=bins, alpha=0.5, density=True, label="real->real (baseline)", color="#2ca02c")
    ax.hist(d_rs, bins=bins, alpha=0.5, density=True, label="real->synthetic (DCR)", color="#ff7f0e")
    ax.set_xlabel("distance to closest record")
    ax.set_ylabel("density")
    ax.set_title("Distance to Closest Record")
    ax.legend()
    b64 = _save_fig(fig, out_png)

    info.update(
        {
            "dcr_real_synth_median": float(np.median(d_rs)),
            "dcr_real_synth_p05": float(np.percentile(d_rs, 5)),
            "dcr_real_real_median": float(np.median(d_rr)),
            "dcr_ratio_median": float(np.median(d_rs) / (np.median(d_rr) + 1e-12)),
            "png": out_png,
            "b64": b64,
            # raw distances kept for cross-synthesizer overlay plots
            # (stripped out of JSON summaries by privacy_report)
            "distances_real_synth": d_rs,
            "distances_real_real": d_rr,
        }
    )
    return info


def nearest_real_examples(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    n: int = 5,
    scan_cap: int = 2000,
    random_state: int = 0,
    holdout: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """The concrete "can this be reverse-engineered back to a real record?"
    check: find the CLOSEST synthetic rows to any real (training) row -- worst
    case, not a random sample -- and return the pair side by side with the
    distance.

    The minimum distance over many scanned rows is an extreme-value statistic:
    scan enough rows and *some* minimum will look small even with zero
    leakage, purely from chance -- comparing it to a "typical" real-to-real
    distance is comparing a minimum to a median, not apples to apples. So the
    synthetic minimum is instead graded against a bootstrap ceiling: what
    minimum distance would a same-size sample of REAL holdout rows achieve
    against the same training rows, by chance alone? That is the same
    real-holdout-ceiling philosophy NewRowSynthesis/CategoricalCAP use above,
    adapted for a minimum instead of a mean/rate.
    """
    from sklearn.neighbors import NearestNeighbors

    cols = roles.modelable
    out: Dict[str, object] = {"examples": [], "note": ""}
    if not cols or len(synth) < 1 or len(real) < 3:
        out["note"] = "insufficient data for nearest-record lookup"
        return out

    try:
        enc, use_cols = _fit_mixed_encoder(real, roles)
    except ValueError as e:
        out["note"] = str(e)
        return out

    rng = np.random.default_rng(random_state)
    scan = synth.sample(min(scan_cap, len(synth)), random_state=random_state) \
        if len(synth) > scan_cap else synth

    Rall = np.nan_to_num(_encode(enc, real, use_cols))
    Sscan = np.nan_to_num(_encode(enc, scan, use_cols))

    nn_real = NearestNeighbors(n_neighbors=1).fit(Rall)
    d, idx = nn_real.kneighbors(Sscan)
    d, idx = d.ravel(), idx.ravel()

    # real -> real "typical spacing" -- only used to phrase distance in
    # interpretable terms on each example, never to grade PASS/FAIL
    base_sample = Rall[: min(len(Rall), 1000)]
    nn_rr = NearestNeighbors(n_neighbors=2).fit(Rall)
    d_rr, _ = nn_rr.kneighbors(base_sample)
    typical_baseline = d_rr[:, 1]

    order = np.argsort(d)[: min(n, len(d))]   # closest matches = worst case
    display_cols = [c for c in cols if c in real.columns and c in synth.columns][:8]
    scan_disp = scan[display_cols].astype(str)
    real_disp = real[display_cols].astype(str)

    examples = []
    for pos in order:
        dist = float(d[pos])
        pct = float((typical_baseline < dist).mean() * 100)
        examples.append({
            "synthetic_row": scan_disp.iloc[int(pos)].to_dict(),
            "nearest_real_row": real_disp.iloc[int(idx[pos])].to_dict(),
            "distance": round(dist, 4),
            "percentile_vs_real_baseline": round(pct, 1),
        })

    out["examples"] = examples
    out["min_distance"] = float(d.min())
    out["baseline_median"] = float(np.median(typical_baseline))

    if holdout is not None and len(holdout) >= 5:
        Hall = np.nan_to_num(_encode(enc, holdout, use_cols))
        dh, _ = nn_real.kneighbors(Hall)
        dh = dh.ravel()
        n_draw = len(Sscan)
        # bootstrap with replacement -- holdout is usually far smaller than
        # the synthetic scan, so this matches the SAMPLE SIZE of the "take N,
        # keep the smallest" procedure rather than the raw row count
        boot_mins = np.array([rng.choice(dh, size=n_draw, replace=True).min()
                               for _ in range(200)])
        ceiling = float(np.percentile(boot_mins, 5))
        out["holdout_bootstrap_min_p05"] = ceiling
        out["holdout_bootstrap_min_median"] = float(np.median(boot_mins))
        out["note"] = (
            f"closest synthetic-to-real pair (worst case, not a random sample), distance "
            f"{out['min_distance']:.4f}. Graded against a real-holdout ceiling: scanning "
            f"{n_draw} real (unseen) rows and taking their own closest match to the "
            f"training data lands at {ceiling:.4f} or below only 5% of the time by chance "
            f"alone -- a synthetic minimum AT OR ABOVE that is no worse than real, unseen "
            f"data gets from pure multiple-comparisons luck; below it is the actual signal."
        )
    else:
        out["note"] = (
            "closest synthetic-to-real pairs found (worst case, not a random sample); no "
            "holdout ceiling available (need >=5 holdout rows), so this is descriptive only "
            "and not gated as PASS/FAIL."
        )

    return out


def filter_close_records(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    resample_fn=None,
    percentile: float = 5.0,
    max_attempts: int = 5,
    random_state: int = 0,
) -> Dict[str, object]:
    """Reject-and-resample privacy filter: the active counterpart to
    nearest_real_examples -- that MEASURES how close synthetic rows sit to
    real ones, this ACTS on it.

    Any synthetic row sitting closer to its nearest real (training) row than
    real rows ever sit to EACH OTHER, below the ``percentile``-th percentile
    of real-to-real nearest-neighbor distances (5 by default, matching the
    same convention nearest_real_examples' bootstrap ceiling uses), is
    dropped. If ``resample_fn`` is given (draw N more rows from the already-
    fitted synthesizer), the quota is refilled from fresh draws, themselves
    checked the same way, for up to ``max_attempts`` rounds; if the model
    keeps producing close rows even after that, the output is short by that
    many rows rather than keeping the risky ones just to hit a row count.

    Unlike nearest_real_examples (which grades a MINIMUM over many scanned
    rows against a same-size-sample bootstrap ceiling, to correct for the
    "closest of N draws is smaller than typical" effect when judging a whole
    dataset's worst case), this filters EACH row independently against the
    real data's own nearest-neighbor spacing -- the right reference point
    when the question is "is this one row too close," not "is the dataset's
    worst case suspicious."
    """
    cols = roles.modelable
    report = {"n_input": len(synth), "n_rejected": 0, "n_resampled": 0,
              "n_output": len(synth), "threshold": None, "note": ""}
    if not cols or len(synth) < 1 or len(real) < 3:
        report["note"] = "insufficient data for close-record filtering"
        return {"data": synth, "report": report}

    from sklearn.neighbors import NearestNeighbors

    try:
        enc, use_cols = _fit_mixed_encoder(real, roles)
    except ValueError as e:
        report["note"] = str(e)
        return {"data": synth, "report": report}

    Rall = np.nan_to_num(_encode(enc, real, use_cols))
    nn_real = NearestNeighbors(n_neighbors=1).fit(Rall)

    nn_rr = NearestNeighbors(n_neighbors=2).fit(Rall)
    d_rr, _ = nn_rr.kneighbors(Rall)
    threshold = float(np.percentile(d_rr[:, 1], percentile))
    report["threshold"] = threshold

    def _reject_mask(df: pd.DataFrame) -> np.ndarray:
        X = np.nan_to_num(_encode(enc, df, use_cols))
        d, _ = nn_real.kneighbors(X)
        return d.ravel() < threshold

    current = synth.reset_index(drop=True)
    bad = _reject_mask(current)
    report["n_rejected"] = int(bad.sum())
    kept = current[~bad]

    attempts = 0
    while len(kept) < len(current) and resample_fn is not None and attempts < max_attempts:
        need = len(current) - len(kept)
        attempts += 1
        try:
            extra = resample_fn(max(need * 2, 10))
        except Exception:
            break
        if extra is None or len(extra) == 0:
            break
        extra = extra.reset_index(drop=True)
        good_extra = extra[~_reject_mask(extra)].head(need)
        report["n_resampled"] += len(good_extra)
        kept = pd.concat([kept, good_extra], ignore_index=True)

    report["n_output"] = len(kept)
    if len(kept) < len(current):
        report["note"] = (f"could not fully refill after {attempts} resample attempt(s) -- "
                           f"output is {len(current) - len(kept)} row(s) short of the request "
                           f"rather than keeping rows that failed the check")
    return {"data": kept.reset_index(drop=True), "report": report}


def _cascade_select(tables: Dict[str, pd.DataFrame], children_of: Dict[str, list],
                     pk_of: Dict[str, str], root: str, root_keys: set) -> Dict[str, "np.ndarray"]:
    """BFS the parent->child relationship graph from ``root``/``root_keys``,
    returning a boolean row-mask per reachable table marking rows linked
    (directly or transitively, through however many hops) to one of
    ``root_keys``. One walk, two uses by the caller: pass the BAD root keys to
    get a drop-mask, or the GOOD root keys against a freshly sampled batch to
    get a keep-mask -- the graph walk is identical either way.
    """
    masks: Dict[str, "np.ndarray"] = {}
    root_df = tables.get(root)
    if root_df is None or pk_of.get(root) not in root_df.columns:
        return masks
    masks[root] = root_df[pk_of[root]].isin(root_keys).to_numpy()
    frontier = [(root, root_keys)]
    while frontier:
        parent, keys = frontier.pop()
        for ct, fk, _ in children_of.get(parent, []):
            cdf = tables.get(ct)
            if cdf is None or fk not in cdf.columns:
                continue
            mask = cdf[fk].isin(keys).to_numpy()
            masks[ct] = mask | masks.get(ct, np.zeros(len(cdf), dtype=bool))
            child_pk = pk_of.get(ct)
            if child_pk and child_pk in cdf.columns and mask.any():
                frontier.append((ct, set(cdf.loc[mask, child_pk])))
    return masks


def filter_close_records_multitable(
    real_tables: Dict[str, pd.DataFrame],
    synth_tables: Dict[str, pd.DataFrame],
    roles: Dict[str, ColumnRoles],
    relationships: List[dict],
    resample_fn=None,
    percentile: float = 5.0,
    max_attempts: int = 3,
) -> Dict[str, object]:
    """Multi-table counterpart to filter_close_records, for HMA.

    HMA links rows across tables by shared keys, so a risky row can't just be
    dropped in isolation in general: removing a row from a table that HAS
    declared children has to cascade to every descendant row that references
    it, and refilling means pulling whole fresh LINKED GROUPS from a new
    sampled batch, not individual rows.

    Every table with both a ``roles`` entry and real data is checked for
    closeness, UNLESS one of its ancestors (per ``relationships``) is ALSO
    such a table -- that ancestor's cascade already covers it, checking it
    again separately would double up. This one rule handles three shapes
    without special-casing them:

      * No relationships at all: every table has no ancestor, so every table
        is its own independent check (same idea as the single-table filter,
        just sharing one fresh full-batch resample per retry round across
        every table that still needs refilling, instead of resampling per
        table).
      * A genuine declared parent/child pair between two REAL tables: only
        the parent is checked, and a rejection cascades down to the child
        (and further descendants, if any) via their foreign keys.
      * Entity-key/hub mode: the derived hub table has no ``roles`` entry
        (it isn't a real table), so its real children fall straight through
        the "ancestor also checked" test and are each treated as their own
        independent check -- this is the case that silently fell through
        earlier (an empty ``roots`` list from requiring the STRUCTURAL root
        to itself be a real table), leaving every child unfiltered even
        though each one individually has no further children to cascade to
        and doesn't need one.

    A row being close in a leaf/child table alone, without its parent also
    being close, is not covered by this pass -- a deliberately scoped v1, not
    an oversight.

    ``resample_fn``, if given, draws a fresh FULL multi-table batch (e.g.
    ``lambda: hma.sample(scale=scale)``) -- HMA has no "give me N more root
    rows only" API, so each retry resamples everything and keeps just what's
    needed, discarding the rest. ``max_attempts`` defaults lower than the
    single-table filter's because of that extra cost per retry.
    """
    from sklearn.neighbors import NearestNeighbors

    report: Dict[str, object] = {"tables": {}, "note": ""}
    current = dict(synth_tables)

    children_of: Dict[str, list] = {}
    pk_of: Dict[str, str] = {}
    parent_of: Dict[str, str] = {}
    for r in relationships:
        children_of.setdefault(r["parent_table_name"], []).append(
            (r["child_table_name"], r["child_foreign_key"], r["parent_primary_key"]))
        pk_of[r["parent_table_name"]] = r["parent_primary_key"]
        parent_of[r["child_table_name"]] = r["parent_table_name"]

    def _has_ancestor_with_roles(t: str) -> bool:
        seen, cur = set(), parent_of.get(t)
        while cur and cur not in seen:
            seen.add(cur)
            if cur in roles and cur in real_tables:
                return True
            cur = parent_of.get(cur)
        return False

    effective_roots = [t for t in synth_tables
                        if t in roles and t in real_tables and not _has_ancestor_with_roles(t)]
    if not effective_roots:
        report["note"] = "no table with both roles and real data found to check"
        return {"data": synth_tables, "report": report}

    for root in effective_roots:
        root_roles = roles[root]
        real_root, synth_root = real_tables[root], current[root]
        try:
            enc, use_cols = _fit_mixed_encoder(real_root, root_roles)
        except ValueError as e:
            report["tables"][root] = {"note": str(e)}
            continue

        Rall = np.nan_to_num(_encode(enc, real_root, use_cols))
        nn_real = NearestNeighbors(n_neighbors=1).fit(Rall)
        nn_rr = NearestNeighbors(n_neighbors=2).fit(Rall)
        d_rr, _ = nn_rr.kneighbors(Rall)
        threshold = float(np.percentile(d_rr[:, 1], percentile))

        def _mask(df: pd.DataFrame, _enc=enc, _cols=use_cols, _nn=nn_real, _th=threshold):
            X = np.nan_to_num(_encode(_enc, df, _cols))
            d, _ = _nn.kneighbors(X)
            return d.ravel() < _th

        n_input = len(synth_root)
        bad_mask = _mask(synth_root)
        n_bad = int(bad_mask.sum())

        # only cascade if this root actually HAS declared children with a
        # usable primary key -- e.g. an entity-hub child with no children of
        # its own needs none of that, a plain boolean mask is enough
        pk_col = pk_of.get(root)
        has_children = bool(children_of.get(root)) and pk_col and pk_col in synth_root.columns
        cascaded_removed: Dict[str, int] = {}

        if has_children:
            bad_keys = set(synth_root.loc[bad_mask, pk_col])
            drop_masks = _cascade_select(current, children_of, pk_of, root, bad_keys)
            for t, mask in drop_masks.items():
                if mask.any():
                    cascaded_removed[t] = int(mask.sum())
                    current[t] = current[t][~mask].reset_index(drop=True)
        else:
            current[root] = current[root][~bad_mask].reset_index(drop=True)

        n_resampled = 0
        attempts = 0
        while len(current[root]) < n_input and resample_fn is not None and attempts < max_attempts:
            attempts += 1
            try:
                fresh = resample_fn()
            except Exception:
                break
            if root not in fresh or len(fresh[root]) == 0:
                break
            need = n_input - len(current[root])
            fresh_root = fresh[root]
            fresh_bad = _mask(fresh_root)

            if has_children:
                good_keys = set(fresh_root.loc[~fresh_bad, pk_col])
                good_keys = set(list(good_keys)[:need])
                if not good_keys:
                    continue
                keep_masks = _cascade_select(fresh, children_of, pk_of, root, good_keys)
                for t, mask in keep_masks.items():
                    if not mask.any():
                        continue
                    piece = fresh[t][mask]
                    current[t] = pd.concat([current.get(t, piece.iloc[0:0]), piece], ignore_index=True)
                    if t == root:
                        n_resampled += len(piece)
            else:
                good = fresh_root[~fresh_bad].head(need)
                if len(good):
                    n_resampled += len(good)
                    current[root] = pd.concat([current[root], good], ignore_index=True)

        report["tables"][root] = {
            "n_input": n_input, "n_rejected": n_bad, "n_resampled": n_resampled,
            "n_output": len(current[root]), "cascaded_removed": cascaded_removed,
            "threshold": threshold,
            "note": "" if len(current[root]) >= n_input else
                    f"could not fully refill after {attempts} resample attempt(s) -- "
                    f"output is {n_input - len(current[root])} row(s) short",
        }

    return {"data": current, "report": report}


def exact_match_rate(real: pd.DataFrame, synth: pd.DataFrame, roles: ColumnRoles) -> Dict[str, float]:
    """Fraction of synthetic rows that exactly match a real row (modelable cols)."""
    cols = [c for c in roles.modelable if c in real.columns and c in synth.columns]
    if not cols or len(synth) == 0:
        return {"exact_match_rate": 0.0, "n_exact_matches": 0, "note": "no comparable columns"}
    real_keys = set(map(tuple, real[cols].astype(str).fillna("<NA>").itertuples(index=False, name=None)))
    synth_rows = list(map(tuple, synth[cols].astype(str).fillna("<NA>").itertuples(index=False, name=None)))
    matches = sum(1 for row in synth_rows if row in real_keys)
    return {
        "exact_match_rate": float(matches / len(synth_rows)),
        "n_exact_matches": int(matches),
        "note": "",
    }


def sdmetrics_privacy(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    metadata=None,
    table_name: str = "",
    holdout: Optional[pd.DataFrame] = None,
    cap_sensitive: Optional[str] = None,
) -> Dict[str, object]:
    """Run sdmetrics single-table privacy metrics that apply generically.

    ``real`` is the training split; ``holdout`` (if given) is the pre-fit
    validation split, which lets us run sdmetrics' DCROverfittingProtection --
    the official parallel to our custom MIA/DCR memorisation checks.
    """
    out: Dict[str, object] = {}
    # DCR metrics need a single-table metadata dict WITH a top-level 'columns'
    # key -- that is metadata.to_dict()['tables'][table], not the object's
    # own .to_dict() (which nests differently).  Resolve both forms.
    meta_dict = None
    try:
        if metadata is not None and hasattr(metadata, "to_dict"):
            full = metadata.to_dict()
            if isinstance(full, dict) and "tables" in full and table_name in full["tables"]:
                meta_dict = full["tables"][table_name]
            elif isinstance(full, dict) and "columns" in full:
                meta_dict = full
        if meta_dict is None:
            single_meta = _single_table_metadata(metadata, table_name)
            md = single_meta.to_dict() if hasattr(single_meta, "to_dict") else single_meta
            meta_dict = md.get("tables", {}).get(table_name, md) if isinstance(md, dict) else md
    except Exception:
        meta_dict = None

    # NewRowSynthesis: fraction of synthetic rows that are genuinely new.
    # Restrict to the *modelable* columns: the id/date/name/audit columns were
    # refilled by independent bootstrap, so they make every row trivially "novel"
    # AND make the per-row scan far slower on wide tables (NewRowSynthesis is
    # ~O(n_synth * n_real * n_cols) -> tens of seconds on a 36-col table).  Also
    # cap the number of synthetic rows scanned so runtime stays bounded.
    try:
        from sdmetrics.single_table import NewRowSynthesis

        nrs_cols = [c for c in roles.modelable if c in real.columns and c in synth.columns
                    and (not meta_dict or c in (meta_dict.get("columns", {}) if isinstance(meta_dict, dict) else {}))]
        if nrs_cols and isinstance(meta_dict, dict) and meta_dict.get("columns"):
            nrs_meta = {"columns": {c: meta_dict["columns"][c] for c in nrs_cols}}
            nrs_real, nrs_synth = real[nrs_cols], synth[nrs_cols]
        else:                                   # fall back to the full frame
            nrs_meta, nrs_real, nrs_synth = meta_dict, real, synth
        cap = int(min(len(nrs_synth), 1000)) or None
        score = NewRowSynthesis.compute(
            real_data=nrs_real, synthetic_data=nrs_synth, metadata=nrs_meta,
            numerical_match_tolerance=0.01, synthetic_sample_size=cap,
        )
        out["NewRowSynthesis"] = float(score)
        # Baseline: what does a REAL holdout score against the training rows on
        # the same columns?  On a low-entropy projection (a few code columns)
        # even real rows duplicate each other, so the achievable ceiling is far
        # below 1 — the synthesizer should be judged against this, not 1.0.
        if holdout is not None and len(holdout):
            try:
                h = holdout[nrs_cols] if list(nrs_real.columns) != list(real.columns) else holdout
                hcap = int(min(len(h), 1000)) or None
                base = NewRowSynthesis.compute(
                    real_data=nrs_real, synthetic_data=h, metadata=nrs_meta,
                    numerical_match_tolerance=0.01, synthetic_sample_size=hcap,
                )
                out["NewRowSynthesis_baseline"] = float(base)
            except Exception:  # pragma: no cover - baseline is best-effort context
                pass
    except Exception as e:  # pragma: no cover
        out["NewRowSynthesis"] = None
        out["NewRowSynthesis_error"] = str(e)[:200]

    # CategoricalCAP: only meaningful with >=1 key field and 1 sensitive field.
    if len(roles.categorical) >= 2:
        try:
            from sdmetrics.single_table import CategoricalCAP, CategoricalGeneralizedCAP

            cands = [c for c in roles.categorical
                     if c in real.columns and c in synth.columns
                     and real[c].nunique(dropna=True) >= 2]
            if not cands:
                raise ValueError("no categorical column with >=2 distinct values")
            if cap_sensitive and cap_sensitive in cands:
                sensitive, picked = cap_sensitive, "user"
            else:
                # Auto-pick the most BALANCED candidate (smallest majority-class
                # share).  With a heavily skewed sensitive field the attacker wins
                # by guessing the majority value everywhere — that is population
                # knowledge, not individual disclosure, and it fails every
                # synthesizer identically.  A balanced field makes the attack
                # (and the score) actually about the synthetic data.
                sensitive = min(cands, key=lambda c: float(
                    real[c].value_counts(normalize=True, dropna=True).iloc[0]))
                picked = ("auto" if not cap_sensitive
                          else f"auto ('{cap_sensitive}' not usable)")
            key = [c for c in roles.categorical if c != sensitive][:3]

            def _attack(attacker: pd.DataFrame):
                v = float(CategoricalCAP.compute(
                    real_data=real, synthetic_data=attacker,
                    key_fields=key, sensitive_fields=[sensitive]))
                var = "exact"
                if np.isnan(v):
                    # Plain CAP only scores real rows whose *exact* key combination
                    # occurs in the attacker's data; with high-cardinality code
                    # columns as keys that can be zero rows -> NaN. Fall back to the
                    # generalized attacker, which matches the closest key (hamming
                    # distance) instead, so the attack is always scoreable.
                    v = float(CategoricalGeneralizedCAP.compute(
                        real_data=real, synthetic_data=attacker,
                        key_fields=key, sensitive_fields=[sensitive]))
                    var = "generalized"
                return v, var

            cap, variant = _attack(synth)
            out["CategoricalCAP"] = None if np.isnan(cap) else cap
            out["CategoricalCAP_fields"] = {"key": key, "sensitive": sensitive,
                                            "variant": variant, "picked": picked}
            # Baseline: run the SAME attack armed with a real holdout instead of
            # the synthetic data.  If the sensitive field is inferable from the
            # real data's own structure, the holdout scores just as low — that is
            # the achievable ceiling the synthesizer should be judged against.
            if out["CategoricalCAP"] is not None and holdout is not None and len(holdout):
                try:
                    base, _ = _attack(holdout)
                    if not np.isnan(base):
                        out["CategoricalCAP_baseline"] = float(base)
                except Exception:  # pragma: no cover - baseline is best-effort
                    pass
        except Exception as e:  # pragma: no cover
            out["CategoricalCAP"] = None
            out["CategoricalCAP_error"] = str(e)[:200]
    return out


def privacy_report(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    table_name: str,
    reports_dir: str,
    metadata=None,
    cap_sensitive: Optional[str] = None,
) -> Dict[str, object]:
    """Compact privacy module for one table: three metrics + verdicts.

    A small, non-overlapping set, one per distinct attack:

      * Membership Inference (MIA) — custom trained-attacker AUC: can a real
                                     record be identified as a training member?
      * NewRowSynthesis (sdmetrics) — are synthetic rows novel (not copies)?
      * CategoricalCAP (sdmetrics)  — can a sensitive categorical field be inferred?

    Plus one concrete example-based check, nearest_record_examples: the
    closest synthetic row to any real row, shown side by side with the
    distance — the literal "can you reverse-engineer this back to a real
    person" test, for demoing rather than just asserting the score above.

    MIA and sdmetrics' DCROverfittingProtection test the same membership-
    inference threat; we report the trained-attacker AUC framing here.
    """
    mia = membership_inference_attack(train_real, holdout_real, synth, roles)
    sdm = sdmetrics_privacy(train_real, synth, roles, metadata, table_name,
                            holdout=holdout_real, cap_sensitive=cap_sensitive)
    nearest = nearest_real_examples(train_real, synth, roles, holdout=holdout_real)

    verdicts = {}
    # MIA AUC close to 0.5 == attacker cannot tell members from non-members.
    auc = mia.get("auc")
    if auc is None or (isinstance(auc, float) and np.isnan(auc)):
        verdicts["membership_inference"] = ("SKIP", mia.get("note", "MIA not computed"))
    elif abs(auc - 0.5) <= 0.10:
        verdicts["membership_inference"] = ("PASS", f"attacker AUC={auc:.3f} (~0.5 => members indistinguishable)")
    elif abs(auc - 0.5) <= 0.20:
        verdicts["membership_inference"] = ("WARN", f"attacker AUC={auc:.3f} (some membership signal)")
    else:
        verdicts["membership_inference"] = ("FAIL", f"attacker AUC={auc:.3f} (strong membership signal)")

    nrs = sdm.get("NewRowSynthesis")
    if nrs is not None:
        base = sdm.get("NewRowSynthesis_baseline")
        if base is not None:
            # Judge against what real data achieves on the same columns: on a
            # low-entropy projection even a real holdout duplicates training
            # rows, so an absolute bar would fail every synthesizer for a
            # property of the table.  gap = how far below the real ceiling.
            gap = base - nrs
            verdicts["new_row_synthesis"] = (
                "PASS" if gap <= 0.05 else "WARN" if gap <= 0.20 else "FAIL",
                f"NewRowSynthesis={nrs:.3f} vs {base:.3f} for a real holdout on the same "
                f"columns — {'matches the achievable ceiling' if gap <= 0.05 else f'{gap:.2f} below it'}"
                + ("" if base >= 0.9 else
                   " (low ceiling: the evaluated columns hold few distinct combinations, "
                   "so duplicates are expected even between real rows)"),
            )
        else:
            verdicts["new_row_synthesis"] = (
                "PASS" if nrs >= 0.9 else "WARN" if nrs >= 0.7 else "FAIL",
                f"NewRowSynthesis={nrs:.3f} (fraction of synthetic rows that are not copies of real rows)",
            )

    cap = sdm.get("CategoricalCAP")
    if cap is not None and not (isinstance(cap, float) and np.isnan(cap)):
        f = sdm.get("CategoricalCAP_fields") or {}
        note = " · nearest-key attacker (no real key combo occurs verbatim in the synthetic data)" \
            if f.get("variant") == "generalized" else ""
        sens = f" (sensitive: {f['sensitive']})" if f.get("sensitive") else ""
        base = sdm.get("CategoricalCAP_baseline")
        if base is not None:
            # Judge against what a REAL holdout scores under the same attack: if
            # the sensitive field is inferable from the real data's own structure
            # (correlations, imbalance), even real rows "leak" it — an absolute
            # bar would fail every synthesizer for a property of the table.
            #
            # Judged on TWO lenses, taking whichever is more forgiving for PASS
            # and requiring BOTH to agree for FAIL:
            #   absolute gap    -- the original, baseline-agnostic bar. A tiny
            #                      gap (e.g. 0.03) is fine no matter what.
            #   relative loss   -- gap / headroom (headroom = 1 - base), i.e.
            #                      what fraction of the real-data ceiling's
            #                      remaining room the synthesizer gave up;
            #                      equivalently "the attack succeeds X% more
            #                      often than it already does against real
            #                      data". Needed because a flat absolute gap
            #                      means very different things at a 0.55
            #                      baseline (attacker success barely moves) vs
            #                      a 0.95 baseline (same gap, much bigger bite
            #                      out of a much smaller remaining margin).
            # Relative-only would flag a harmless 0.03 gap at a 0.95 baseline
            # (tiny headroom inflates the percentage) -- the absolute lens
            # gives that the benefit of the doubt; conversely absolute-only
            # would miss a real, large relative jump at a high baseline that
            # still has a "moderate-looking" raw gap. PASS needs only one
            # lens to look fine; FAIL needs both to look bad.
            gap = base - cap
            headroom = 1 - base
            rel_loss = (gap / headroom) if headroom > 1e-6 else (float("inf") if gap > 1e-6 else 0.0)
            passes = gap <= 0.05 or rel_loss <= 0.25
            fails = gap > 0.20 and rel_loss > 1.0
            status = "PASS" if passes else "FAIL" if fails else "WARN"
            if status == "PASS":
                detail = "matches the real-data ceiling"
            elif rel_loss == float("inf"):
                detail = "the real-data baseline leaks nothing here, but the synthetic data does"
            else:
                detail = f"{gap:.2f} below it (the attack succeeds {rel_loss * 100:.0f}% more often than it already does against real data)"
            verdicts["categorical_cap"] = (
                status,
                f"CategoricalCAP={cap:.3f} vs {base:.3f} for a real holdout under the "
                f"same attack{sens} — " + detail
                + ("" if base >= 0.5 else
                   " (low ceiling: the sensitive field is largely guessable from the real "
                   "data itself — population statistics, not individual disclosure)")
                + note,
            )
        else:
            verdicts["categorical_cap"] = (
                "PASS" if cap >= 0.5 else "WARN" if cap >= 0.3 else "FAIL",
                f"CategoricalCAP={cap:.3f} (1.0 => a sensitive field cannot be "
                f"inferred from the key fields){sens}{note}",
            )
    elif "CategoricalCAP" in sdm:
        # attempted but not computable — a SKIP, never a FAIL: NaN carries no
        # evidence of leakage (nan >= 0.5 is False, which used to fall to FAIL)
        why = sdm.get("CategoricalCAP_error", "no scoreable rows")
        verdicts["categorical_cap"] = (
            "SKIP", f"CategoricalCAP not computable ({why}) — no evidence either way")
    else:
        # not even attempted: the attack needs >=1 categorical key field plus a
        # categorical sensitive field — say so instead of silently omitting the row
        verdicts["categorical_cap"] = (
            "SKIP", f"needs ≥2 categorical columns (1 key + 1 sensitive); this table "
                    f"has {len(roles.categorical)} — attack not applicable")

    examples = nearest.get("examples") or []
    ceiling = nearest.get("holdout_bootstrap_min_p05")
    if examples and ceiling is not None:
        min_dist = nearest["min_distance"]
        if min_dist >= ceiling:
            status, why = "PASS", "no closer than real, unseen data gets from pure chance"
        else:
            # how far below the ceiling, as a fraction of the ceiling itself
            ratio = min_dist / (ceiling + 1e-9)
            status = "WARN" if ratio >= 0.5 else "FAIL"
            why = "closer to a real record than even chance alone produces from real data"
        verdicts["nearest_record"] = (
            status,
            f"closest synthetic row's distance ({min_dist:.4f}) vs the real-holdout 5th-"
            f"percentile ceiling for a same-size sample ({ceiling:.4f}) — {why}",
        )
    elif examples:
        pct = examples[0]["percentile_vs_real_baseline"]
        verdicts["nearest_record"] = (
            "SKIP",
            f"closest synthetic row at the {pct:.0f}th percentile of typical real-to-real "
            f"spacing — no real-holdout ceiling available to grade this against (need "
            f">=5 holdout rows)",
        )
    else:
        verdicts["nearest_record"] = ("SKIP", nearest.get("note") or "not computable")

    return {
        "table": table_name,
        "membership_inference": mia,
        "sdmetrics": sdm,
        "nearest_record_examples": nearest,
        "verdicts": {k: {"status": s, "detail": d} for k, (s, d) in verdicts.items()},
    }


