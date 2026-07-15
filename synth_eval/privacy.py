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

            sensitive = roles.categorical[-1]
            key = [c for c in roles.categorical if c != sensitive][:3]
            cap = float(CategoricalCAP.compute(
                real_data=real, synthetic_data=synth,
                key_fields=key, sensitive_fields=[sensitive],
            ))
            variant = "exact"
            if np.isnan(cap):
                # Plain CAP only scores real rows whose *exact* key combination
                # occurs in the synthetic data; with high-cardinality code
                # columns as keys that can be zero rows -> NaN. Fall back to the
                # generalized attacker, which matches the closest synthetic key
                # (hamming distance) instead, so the attack is always scoreable.
                cap = float(CategoricalGeneralizedCAP.compute(
                    real_data=real, synthetic_data=synth,
                    key_fields=key, sensitive_fields=[sensitive],
                ))
                variant = "generalized"
            out["CategoricalCAP"] = None if np.isnan(cap) else cap
            out["CategoricalCAP_fields"] = {"key": key, "sensitive": sensitive,
                                            "variant": variant}
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
        variant = (sdm.get("CategoricalCAP_fields") or {}).get("variant", "exact")
        note = " · nearest-key attacker (no real key combo occurs verbatim in the synthetic data)" \
            if variant == "generalized" else ""
        verdicts["categorical_cap"] = (
            "PASS" if cap >= 0.5 else "WARN" if cap >= 0.3 else "FAIL",
            f"CategoricalCAP={cap:.3f} (1.0 => a sensitive field cannot be "
            f"inferred from the key fields){note}",
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

    return {
        "table": table_name,
        "membership_inference": mia,
        "sdmetrics": sdm,
        "verdicts": {k: {"status": s, "detail": d} for k, (s, d) in verdicts.items()},
    }


