"""synth_eval.efficacy — target selection and TSTR ML-efficacy metrics."""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .columns import ColumnRoles, _fit_mixed_encoder, _encode, group_diversity_reduction

#: minimum macro-F1 / R^2 lift a target must clear over its noise floor
#: before its efficacy ratio is trusted as signal rather than noise-over-noise.
#: Deliberately low -- this only screens out targets with essentially NO real
#: relationship to the other columns, not weak ones.
_MIN_SIGNAL_LIFT = 0.05
#: label-permutation repeats used to estimate the classification noise floor
#: (see _predictive_signal) -- a flexible tree can overfit pure noise well
#: past what a majority-class score suggests, so that alone isn't a safe
#: baseline; a few shuffled-label refits of the SAME tree measure how much
#: apparent score this exact model/sample-size can manufacture from nothing.
_SHUFFLE_REPEATS = 5
#: how close a feature's group_diversity_reduction with the TARGET must sit
#: to that feature's OWN ceiling (1 - 1/cardinality) before it's treated as a
#: quasi-identifier for the signal check (see _quasi_identifier_group_col) --
#: same numeric bar as best_refill_group_column's min_association, but
#: measured as a RATIO of the feature's own ceiling rather than the raw
#: score, since a low-cardinality feature (e.g. 2 categories, ceiling 0.5)
#: can never reach a fixed raw threshold like 0.5 even at perfect
#: determinism -- the ratio scales correctly regardless of cardinality.
_QUASI_ID_TOLERANCE = 0.5


def _quasi_identifier_group_col(
    df: pd.DataFrame, target_col: str, feature_roles: ColumnRoles,
) -> Optional[str]:
    """A categorical feature whose group_diversity_reduction with
    ``target_col`` sits within ``_QUASI_ID_TOLERANCE`` of that feature's own
    ceiling is a quasi-identifier for it -- e.g. an SCD-versioned table's own
    entity key, where every OTHER static attribute is basically fixed per
    entity (an occupation code near-determines its own category, skill
    level, etc.). A plain random row split lets a model "predict" the target
    by memorizing that feature's value instead of learning anything general,
    since multiple rows sharing that value routinely land on both sides of
    the split. Returns the single BEST such feature (highest ratio to its
    own ceiling), or ``None`` if nothing qualifies -- the normal case for a
    table where rows genuinely are independent entities.
    """
    best_col, best_ratio = None, _QUASI_ID_TOLERANCE
    for c in feature_roles.categorical:
        if c not in df.columns:
            continue
        card = df[c].nunique(dropna=True)
        if card <= 1 or card >= len(df):   # not a real grouping candidate
            continue
        ceiling = 1.0 - 1.0 / card
        if ceiling <= 0:
            continue
        ratio = group_diversity_reduction(df, c, target_col) / ceiling
        if ratio >= best_ratio:
            best_col, best_ratio = c, ratio
    return best_col


class InsufficientHoldoutError(ValueError):
    """Raised by sdmetrics_ml_efficacy when the REAL baseline's own holdout
    split doesn't share enough target classes with its own training split to
    score reliably (small table, many-class target -- e.g. 131 rows across
    21 codes leaves an 80/20 holdout with only a handful of rows per class,
    easily missing several entirely). Per-metric failures already fall back
    to a NaN row with an explanatory note (see the try/except around each
    metric below) -- this is different: if even the REAL baseline can't be
    scored, comparing synthesizers against it is meaningless, so the caller
    should skip the WHOLE target for this table (folding it into the same
    efficacy_skipped list auto_select_target's own guards use) rather than
    publish a table of NaN rows that reads as a cascade of failures."""


def _predictive_signal(
    df: pd.DataFrame, target_col: str, feature_roles: ColumnRoles, task: str,
) -> Optional[Tuple[float, float]]:
    """(real_score, noise_floor_score) from a quick train/test split within
    ``df``, scored with a shallow decision tree -- or ``None`` if there isn't
    enough data to judge either way.

    This is a cheap screen, not the final TSTR metric (see
    :func:`sdmetrics_ml_efficacy`): just enough model to tell "some other
    column predicts this" from "nothing does", before committing a full
    synthesizer comparison -- or showing a ratio -- to a target that's really
    just independent noise.

    On a table where multiple rows represent the same underlying entity
    (an SCD-versioned dimension/reference table -- the normal case for this
    schema), a plain random row split lets a quasi-identifier feature (e.g.
    the entity's own versioning key, which near-determines every OTHER
    static attribute) "predict" the target by memorizing a value it's
    literally already seen for that same entity in training, not by
    learning anything general. If :func:`_quasi_identifier_group_col` finds
    such a feature, the split groups by IT instead (holding out whole
    entities, never seen in training at all) so the signal check only
    credits genuine generalization. Verified on OCCUPATION.csv: a random
    split showed OCCUPATION_CATEGORY_CD "predictable" (macro-F1 0.244 vs a
    0.087 noise floor, comfortably clearing the signal gate) purely via
    OCCUPATION_TP_CD memorization; an entity-aware split drops that to
    0.109 (lift ~0.02, below the gate) -- the honest answer.
    """
    from sklearn.metrics import f1_score, r2_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    y = df[target_col]
    m = y.notna()
    if m.sum() < 20:
        return None
    sub, y = df[m], y[m]

    group_col = _quasi_identifier_group_col(sub, target_col, feature_roles) \
        if task == "classification" else None
    try:
        if group_col:
            groups = sub[group_col]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
            tr_pos, te_pos = next(gss.split(sub, groups=groups))
            tr_idx, te_idx = sub.index[tr_pos], sub.index[te_pos]
        else:
            stratify = y.astype(str) if task == "classification" and y.nunique() > 1 else None
            tr_idx, te_idx = train_test_split(sub.index, test_size=0.3, random_state=0,
                                              stratify=stratify)
    except (ValueError, StopIteration):
        return None  # e.g. a class with a single member, or too few groups to split -- can't judge safely
    try:
        enc, use_cols = _fit_mixed_encoder(sub.loc[tr_idx], feature_roles)
    except ValueError:
        return None  # no usable feature columns
    Xtr = np.nan_to_num(_encode(enc, sub.loc[tr_idx], use_cols))
    Xte = np.nan_to_num(_encode(enc, sub.loc[te_idx], use_cols))
    ytr, yte = y.loc[tr_idx], y.loc[te_idx]
    if task == "classification":
        if ytr.nunique() < 2:
            return None
        ytr_s, yte_s = ytr.astype(str), yte.astype(str)
        clf = DecisionTreeClassifier(max_depth=6, random_state=0)
        clf.fit(Xtr, ytr_s)
        real = float(f1_score(yte_s, clf.predict(Xte), average="macro", zero_division=0))
        # noise floor: same tree, labels permuted -- what a majority-class
        # check alone would miss (see _MIN_SIGNAL_LIFT / _SHUFFLE_REPEATS)
        floor_scores = []
        ytr_arr = ytr_s.to_numpy()
        for i in range(_SHUFFLE_REPEATS):
            shuffled = ytr_arr.copy()
            np.random.default_rng(1000 + i).shuffle(shuffled)
            sh_clf = DecisionTreeClassifier(max_depth=6, random_state=0).fit(Xtr, shuffled)
            floor_scores.append(f1_score(yte_s, sh_clf.predict(Xte),
                                         average="macro", zero_division=0))
        base = float(np.mean(floor_scores))
    else:
        reg = DecisionTreeRegressor(max_depth=6, random_state=0)
        reg.fit(Xtr, ytr)
        real = float(r2_score(yte, reg.predict(Xte)))
        # a mean-predictor's R^2 is 0 by definition -- no fit needed, and more
        # stable than a shuffled-label tree floor (which is erratic for R^2)
        base = 0.0
    return real, base


def target_signal_note(
    df: pd.DataFrame, target_col: str, roles: ColumnRoles, task: str,
    min_lift: float = _MIN_SIGNAL_LIFT,
) -> Optional[str]:
    """Caution string when ``target_col`` doesn't clear its noise floor by
    ``min_lift`` -- i.e. its real-data score is close to what a model with no
    real relationship to predict would score anyway (chance-level guessing
    for regression, or label-permuted overfitting for classification -- see
    :func:`_predictive_signal`), so a synthetic-vs-real efficacy ratio for it
    is mostly noise divided by noise, not a fidelity signal. Returns ``None``
    when there's real signal (or too little data to tell either way).
    """
    feature_roles = ColumnRoles(
        numeric=[c for c in roles.numeric if c != target_col],
        categorical=[c for c in roles.categorical if c != target_col],
    )
    sig = _predictive_signal(df, target_col, feature_roles, task)
    if sig is None:
        return None
    real, base = sig
    if (real - base) >= min_lift:
        return None
    metric = "macro F1" if task == "classification" else "R²"
    return (f"weak real-data signal for '{target_col}' ({metric} {real:.3f} real vs "
            f"{base:.3f} noise floor) — efficacy ratios for this target may reflect "
            f"noise more than fidelity")


def auto_select_target(
    df: pd.DataFrame, roles: ColumnRoles, min_rows: int = 30,
) -> Optional[Tuple[str, str]]:
    """Pick a modelling target: (column, task) where task in {classification, regression}.

    Prefers a categorical column with 2-20 classes (classification); otherwise
    falls back to a numeric column with reasonable variance (regression).

    Returns ``None`` for tables too small or too thin to model meaningfully.
    A dimension/lookup table (an id column plus a name/desc column, both
    skipped by classify_columns, and maybe one small categorical left) can
    otherwise auto-pick that one remaining categorical as a target with
    nothing left to predict it FROM, or hand TSTR/TRTR a holdout of a
    handful of rows where the score is mostly noise. Guards: the table needs
    ``min_rows`` rows (a holdout split that small isn't a meaningful
    comparison), a candidate target needs at least one OTHER modelable column
    left over to use as a feature, and -- since neither of those catches a
    target that's simply *independent* of everything else in the table --
    :func:`_predictive_signal` must clear a real lift over a naive baseline
    (skipped, not blocked, on ties/too-little-data, so this never turns a
    previously-picked target into "no target").
    """
    if len(df) < min_rows:
        return None

    def _has_features(target_col: str) -> bool:
        return any(c != target_col for c in roles.modelable)

    def _feature_roles(target_col: str) -> ColumnRoles:
        return ColumnRoles(
            numeric=[c for c in roles.numeric if c != target_col],
            categorical=[c for c in roles.categorical if c != target_col],
        )

    def _has_signal(target_col: str, task: str) -> bool:
        sig = _predictive_signal(df, target_col, _feature_roles(target_col), task)
        return sig is None or (sig[0] - sig[1]) >= _MIN_SIGNAL_LIFT

    for col in roles.categorical:
        nun = df[col].nunique(dropna=True)
        if 2 <= nun <= 20 and _has_features(col) and _has_signal(col, "classification"):
            return col, "classification"
    # regression fallback: numeric columns ranked by variance, highest first;
    # first one with real signal wins (skip ones nothing predicts)
    candidates = []
    for col in roles.numeric:
        if not _has_features(col):
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if v.notna().sum() < 20:
            continue
        candidates.append((col, float(v.var())))
    for col, _ in sorted(candidates, key=lambda x: -x[1]):
        if _has_signal(col, "regression"):
            return col, "regression"
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
        if task == "classification":
            # sdmetrics' Binary*/Multiclass*Classifier only remap labels to
            # boolean when the target dtype is 'object' -- a numeric-coded
            # binary target (e.g. PREF_LANG_TP_CD = 703793/703794) skips that
            # remap and falls through to sklearn's f1_score with the default
            # pos_label=1, which crashes since neither class IS 1.
            tr[target] = tr[target].astype(str)
            test_src[target] = test_src[target].astype(str)
        base_note = ""
        if len(test_src) < len(test):
            base_note = f"dropped {len(test)-len(test_src)} holdout rows with unseen target class"
        if n_aligned:
            base_note = (base_note + "; " if base_note else "") + \
                f"aligned {n_aligned} feature col(s) with unseen/missing categories to train"
        enough = len(test_src) >= 5
        if src == "real" and not enough:
            # "real" is always the first source (dict insertion order, see
            # `sources` above) -- if even the REAL baseline's own train/
            # holdout split can't be scored, no synthesizer comparison
            # against it means anything either; bail out before producing
            # ANY rows (real or synthetic) instead of a table full of NaNs.
            raise InsufficientHoldoutError(
                f"{len(test)} holdout rows have a value for '{target}', but only "
                f"{len(test_src)} of those share a class the real training split also has"
                + (f" ({nun} classes total)" if task == "classification" else "")
                + " -- too few to score reliably")

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


