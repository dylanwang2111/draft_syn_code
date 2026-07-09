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
    try:
        from sdmetrics.single_table import NewRowSynthesis

        score = NewRowSynthesis.compute(
            real_data=real, synthetic_data=synth, metadata=meta_dict,
            numerical_match_tolerance=0.01,
        )
        out["NewRowSynthesis"] = float(score)
    except Exception as e:  # pragma: no cover
        out["NewRowSynthesis"] = None
        out["NewRowSynthesis_error"] = str(e)[:200]

    # CategoricalCAP: only meaningful with >=1 key field and 1 sensitive field.
    if len(roles.categorical) >= 2:
        try:
            from sdmetrics.single_table import CategoricalCAP

            sensitive = roles.categorical[-1]
            key = [c for c in roles.categorical if c != sensitive][:3]
            cap = CategoricalCAP.compute(
                real_data=real, synthetic_data=synth,
                key_fields=key, sensitive_fields=[sensitive],
            )
            out["CategoricalCAP"] = float(cap)
            out["CategoricalCAP_fields"] = {"key": key, "sensitive": sensitive}
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
) -> Dict[str, object]:
    """Compact privacy module for one table: three metrics + verdicts.

    A small, non-overlapping set, one per distinct attack:

      * Membership Inference (MIA) — custom trained-attacker AUC: can a real
                                     record be identified as a training member?
      * NewRowSynthesis (sdmetrics) — are synthetic rows novel (not copies)?
      * CategoricalCAP (sdmetrics)  — can a sensitive categorical field be inferred?

    MIA and sdmetrics' DCROverfittingProtection test the same membership-
    inference threat; we report the trained-attacker AUC framing here.
    """
    mia = membership_inference_attack(train_real, holdout_real, synth, roles)
    sdm = sdmetrics_privacy(train_real, synth, roles, metadata, table_name, holdout=holdout_real)

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
        verdicts["new_row_synthesis"] = (
            "PASS" if nrs >= 0.9 else "WARN" if nrs >= 0.7 else "FAIL",
            f"NewRowSynthesis={nrs:.3f} (fraction of synthetic rows that are not copies of real rows)",
        )

    cap = sdm.get("CategoricalCAP")
    if cap is not None:
        verdicts["categorical_cap"] = (
            "PASS" if cap >= 0.5 else "WARN" if cap >= 0.3 else "FAIL",
            f"CategoricalCAP={cap:.3f} (1.0 => a sensitive field cannot be "
            f"inferred from the key fields)",
        )

    return {
        "table": table_name,
        "membership_inference": mia,
        "sdmetrics": sdm,
        "verdicts": {k: {"status": s, "detail": d} for k, (s, d) in verdicts.items()},
    }


