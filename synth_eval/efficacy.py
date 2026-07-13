"""synth_eval.efficacy — target selection and TSTR ML-efficacy metrics."""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .columns import ColumnRoles, _fit_mixed_encoder, _encode


def auto_select_target(df: pd.DataFrame, roles: ColumnRoles) -> Optional[Tuple[str, str]]:
    """Pick a modelling target: (column, task) where task in {classification, regression}.

    Prefers a categorical column with 2-20 classes (classification); otherwise
    falls back to a numeric column with reasonable variance (regression).
    """
    for col in roles.categorical:
        nun = df[col].nunique(dropna=True)
        if 2 <= nun <= 20:
            return col, "classification"
    # regression fallback: numeric column with the most variance
    best, best_var = None, -1.0
    for col in roles.numeric:
        v = pd.to_numeric(df[col], errors="coerce")
        if v.notna().sum() < 20:
            continue
        var = float(v.var())
        if var > best_var:
            best, best_var = col, var
    if best is not None:
        return best, "regression"
    return None


def _build_model(task: str, kind: str, random_state: int = 0):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task == "classification":
        if kind == "rf":
            return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        return LogisticRegression(max_iter=1000)
    else:
        if kind == "rf":
            return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        return LinearRegression()


def _prep_xy(df: pd.DataFrame, target: str, feature_roles: ColumnRoles):
    feats = [c for c in feature_roles.modelable if c != target and c in df.columns]
    return df[feats].copy(), df[target].copy(), feats


def _score(task, y_true, y_pred, y_proba=None):
    from sklearn.metrics import (
        accuracy_score, f1_score, r2_score, roc_auc_score, mean_squared_error,
    )

    if task == "classification":
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted")),
        }
        try:
            if y_proba is not None:
                classes = np.unique(y_true)
                if len(classes) == 2:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            out["roc_auc"] = float("nan")
        return out
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {"rmse": rmse, "r2": float(r2_score(y_true, y_pred))}


def ml_efficacy_tstr(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth: pd.DataFrame,
    roles: ColumnRoles,
    target: str,
    task: str,
    table_name: str = "",
    random_state: int = 0,
) -> pd.DataFrame:
    """TSTR: train on real vs synthetic, always test on the SAME real holdout.

    ``synth`` may be a single dataframe (labelled 'synthetic') or a dict of
    ``{synthesizer_name: dataframe}`` to benchmark several synthesizers in one
    tidy comparison table (one row per model x train-source).
    """
    from sklearn.pipeline import Pipeline

    feature_roles = ColumnRoles(
        numeric=[c for c in roles.numeric if c != target],
        categorical=[c for c in roles.categorical if c != target],
    )
    rows = []

    # Test set is always the real holdout.
    Xte, yte, feats = _prep_xy(holdout_real, target, feature_roles)
    if not feats:
        return pd.DataFrame([{"table": table_name, "note": "no usable feature columns"}])

    # Align target dtype for classification (stringify to avoid mixed types).
    def _yfix(y):
        return y.astype(str) if task == "classification" else pd.to_numeric(y, errors="coerce")

    yte = _yfix(yte)

    if isinstance(synth, dict):
        sources = {"real": train_real, **synth}
    else:
        sources = {"real": train_real, "synthetic": synth}
    for kind in ("rf", "baseline"):
        for src_name, src_df in sources.items():
            if target not in src_df.columns:
                continue
            Xtr, ytr, _ = _prep_xy(src_df, target, feature_roles)
            ytr = _yfix(ytr)
            # Drop rows with missing target.
            m = ytr.notna()
            Xtr, ytr = Xtr[m.values], ytr[m.values]
            if len(ytr) < 10 or (task == "classification" and ytr.nunique() < 2):
                continue
            enc, use_cols = _fit_mixed_encoder(train_real, feature_roles)
            model = _build_model(task, kind, random_state)
            pipe = Pipeline([("enc", enc), ("model", model)])
            try:
                pipe.fit(Xtr[use_cols], ytr)
                yp = pipe.predict(Xte[use_cols])
                proba = None
                if task == "classification" and hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(Xte[use_cols])
                    except Exception:
                        proba = None
                mask = yte.notna()
                sc = _score(task, yte[mask.values], np.asarray(yp)[mask.values],
                            None if proba is None else proba[mask.values])
            except Exception as e:  # pragma: no cover
                sc = {"error": str(e)[:200]}
            rows.append({
                "table": table_name, "target": target, "task": task,
                "model": "RandomForest" if kind == "rf" else "baseline",
                "train_on": src_name, **sc,
            })
    df = pd.DataFrame(rows)

    # Add real-vs-synthetic efficacy gap per model x synthetic source.
    gap_rows = []
    metric = "accuracy" if task == "classification" else "r2"
    if not df.empty and "model" in df.columns and metric in df.columns:
        synth_names = [k for k in sources if k != "real"]
        for model in df["model"].unique():
            sub = df[df["model"] == model]
            r = sub[sub["train_on"] == "real"][metric]
            for sname in synth_names:
                s = sub[sub["train_on"] == sname][metric]
                if len(r) and len(s) and pd.notna(r.iloc[0]) and pd.notna(s.iloc[0]):
                    gap_rows.append({
                        "table": table_name, "target": target, "task": task,
                        "model": model, "train_on": f"gap(real-{sname})",
                        metric: float(r.iloc[0] - s.iloc[0]),
                    })
    if gap_rows:
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)
    return df


def sdmetrics_ml_efficacy(
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame,
    synth,
    roles: ColumnRoles,
    target: str,
    task: str,
    table_name: str = "",
    max_train_rows: int = 20000,
) -> pd.DataFrame:
    """Condensed TSTR using **sdmetrics** ML-efficacy metrics.

    Metric selection (kept deliberately small):
        binary target      -> BinaryDecisionTreeClassifier (F1)
        multiclass target  -> MulticlassDecisionTreeClassifier (macro F1)
        numeric target     -> LinearRegression (R^2)

    For classification we also report accuracy / precision (macro) / recall
    (macro), which sdmetrics does not expose, computed from a single sklearn
    DecisionTree fit (rows prefixed ``DecisionTree ·``).  These are distinct
    lenses on the same predictions; F1 is left to the sdmetrics metric above so
    it is not duplicated.

    Each metric's model is trained on (a) the real training split (TRTR
    reference) and (b) each synthesizer's data (TSTR), and always evaluated
    on the SAME real holdout via ``Metric.compute(test_data, train_data,
    target=...)``.  Only modelable columns (id/name/date columns dropped) are
    passed in.  Returns a tidy frame with ``gap(real-<synth>)`` rows appended.
    """
    from sdmetrics import single_table as st

    if isinstance(synth, dict):
        sources = {"real": train_real, **synth}
    else:
        sources = {"real": train_real, "synthetic": synth}

    nun = int(train_real[target].nunique(dropna=True))
    if task == "classification" and nun == 2:
        # one sdmetrics F1 (decision tree) so binary and multiclass are
        # symmetric; the second binary logistic model just duplicated F1.
        metric_classes = {
            "BinaryDecisionTreeClassifier (F1)": st.BinaryDecisionTreeClassifier,
        }
    elif task == "classification":
        metric_classes = {
            "MulticlassDecisionTreeClassifier (macro F1)": st.MulticlassDecisionTreeClassifier,
        }
    else:
        metric_classes = {"LinearRegression (R2)": st.LinearRegression}

    cols = [c for c in roles.modelable if c in holdout_real.columns]
    if target not in cols:
        cols.append(target)
    test = holdout_real[cols].dropna(subset=[target]).copy()

    cat_cols = [c for c in cols if c in roles.categorical]

    NA = "__nan__"

    def _align_categories(tr, te):
        """Make ``tr``/``te`` safe for sdmetrics' one-hot (handle_unknown='error').

        Two things break the encoder and are fixed here on the categorical
        feature columns:
          * a **missing value** in the holdout that the synthetic training data
            never produced -> "Found unknown categories [nan]".  We encode NaN as
            its own explicit ``__nan__`` category on *both* frames so it's never
            unknown.
          * any other **test-only category** absent from training -> mapped to the
            training column's most frequent value.
        Returns ``(tr, te, n_columns_changed)`` — the aligned *train* frame is
        returned too so the metric is fit on the same encoding.
        """
        tr = tr.copy()
        te = te.copy()
        n_aligned = 0
        for c in cat_cols:
            if c == target or c not in tr.columns or c not in te.columns:
                continue
            # missing -> explicit category, as strings, on both sides
            tr[c] = tr[c].astype(object).where(tr[c].notna(), NA).astype(str)
            te[c] = te[c].astype(object).where(te[c].notna(), NA).astype(str)
            known = set(tr[c].unique())
            mask = ~te[c].isin(known)
            if mask.any():
                te.loc[mask, c] = tr[c].mode().iloc[0]
                n_aligned += 1
        return tr, te, n_aligned

    # Extra classification metrics sdmetrics does not expose (accuracy /
    # precision / recall), from ONE DecisionTree fit so they are mutually
    # consistent.  F1 is intentionally omitted here because the sdmetrics
    # *DecisionTreeClassifier (F1)* metric already reports it -- keeping our own
    # F1 too would just duplicate that.  Uses the shared mixed encoder (one-hot
    # handle_unknown='ignore'), matching sdmetrics' decision-tree family.
    EXTRA_CLS = ["accuracy", "precision (macro)", "recall (macro)"]

    def _extra_scores(tr, te):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        feat_roles = ColumnRoles(
            numeric=[c for c in roles.numeric if c != target and c in tr.columns and c in te.columns],
            categorical=[c for c in roles.categorical if c != target and c in tr.columns and c in te.columns],
        )
        if not feat_roles.modelable:
            return {}
        enc, use = _fit_mixed_encoder(tr, feat_roles)
        Xtr = np.nan_to_num(_encode(enc, tr, use))
        Xte = np.nan_to_num(_encode(enc, te, use))
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(Xtr, tr[target].astype(str))
        pred = clf.predict(Xte)
        yt = te[target].astype(str)
        return {
            "accuracy": float(accuracy_score(yt, pred)),
            "precision (macro)": float(precision_score(yt, pred, average="macro", zero_division=0)),
            "recall (macro)": float(recall_score(yt, pred, average="macro", zero_division=0)),
        }

    rows = []
    for src, df in sources.items():
        if target not in df.columns:
            continue
        tr = df[[c for c in cols if c in df.columns]].dropna(subset=[target]).copy()
        if len(tr) > max_train_rows:
            tr = tr.sample(max_train_rows, random_state=0)
        if len(tr) < 10 or len(test) < 5:
            continue
        # drop test rows whose TARGET class was never seen in training
        # (an unseen label cannot be predicted and would crash scoring)
        train_labels = set(tr[target].dropna().astype(str).unique())
        test_src = test[test[target].astype(str).isin(train_labels)].copy()
        tr, test_src, n_aligned = _align_categories(tr, test_src)
        base_note = ""
        if len(test_src) < len(test):
            base_note = f"dropped {len(test)-len(test_src)} holdout rows with unseen target class"
        if n_aligned:
            base_note = (base_note + "; " if base_note else "") + \
                f"aligned {n_aligned} feature col(s) with unseen/missing categories to train"
        enough = len(test_src) >= 5

        def _add(metric, score, note):
            rows.append({"table": table_name, "target": target, "task": task,
                         "metric": metric, "train_on": src, "score": score, "note": note})

        # sdmetrics metric(s) — the "official" score(s).  sdmetrics' internal
        # tree pipeline mean-imputes and warns on all-NaN feature columns
        # (harmless — that column just contributes nothing); silence that noise.
        for mname, M in metric_classes.items():
            try:
                if not enough:
                    raise ValueError("too few holdout rows share the training classes")
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Skipping features without any observed values")
                    score_ = float(M.compute(test_data=test_src, train_data=tr, target=target))
                _add(mname, score_, base_note)
            except Exception as e:  # pragma: no cover - defensive
                _add(mname, float("nan"), (base_note + "; " if base_note else "") + str(e)[:140])

        # extra sklearn classification metrics (accuracy / precision / recall / F1)
        if task == "classification":
            try:
                if not enough:
                    raise ValueError("too few holdout rows share the training classes")
                extra = _extra_scores(tr, test_src)
                for m in EXTRA_CLS:
                    _add(f"DecisionTree · {m}", extra.get(m, float("nan")), base_note)
            except Exception as e:  # pragma: no cover - defensive
                for m in EXTRA_CLS:
                    _add(f"DecisionTree · {m}", float("nan"),
                         (base_note + "; " if base_note else "") + str(e)[:140])
    out = pd.DataFrame(rows)

    # gap(real - synth) per metric x synthetic source
    gaps = []
    if not out.empty:
        for mname in out["metric"].unique():
            sub = out[out["metric"] == mname]
            r = sub[sub["train_on"] == "real"]["score"]
            for sname in [k for k in sources if k != "real"]:
                s = sub[sub["train_on"] == sname]["score"]
                if len(r) and len(s) and pd.notna(r.iloc[0]) and pd.notna(s.iloc[0]):
                    gaps.append({
                        "table": table_name, "target": target, "task": task,
                        "metric": mname, "train_on": f"gap(real-{sname})",
                        "score": float(r.iloc[0] - s.iloc[0]), "note": "",
                    })
    if gaps:
        out = pd.concat([out, pd.DataFrame(gaps)], ignore_index=True)
    return out


