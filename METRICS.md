# Metrics Documentation

All metrics used by `synthetic_evaluation.ipynb` / `synth_eval.py`, grouped by the three
evaluation axes. Metrics come from **sdmetrics** wherever one exists; the two custom
metrics (membership inference, DCR) cover privacy properties sdmetrics does not measure
directly. Every score below is reported **per synthesizer per table** and rolled up into
the leaderboard.

---

## Column handling (applies to every metric)

Columns are auto-classified before any metric runs (`synth_eval.classify_columns`),
using this schema's naming conventions first, then SDV metadata, then dtype/cardinality:

| Rule | Columns | Role |
|---|---|---|
| `*_TP_CD`, `*_TP_CODE`, `*_CD`, `*_CODE` | type codes (e.g. `MARITAL_ST_TP_CD`, `GENDER_TP_CODE`, `PERSON_ORG_CODE`) | **categorical** (skipped if > 50 distinct values) |
| `*_IND` | Y/N flags (e.g. `SOLICIT_IND`, `ALERT_IND`) | **categorical** |
| `*_ID`, `*_TX_ID` | identifiers (`CONT_ID`, `IDP_WAREHOUSE_ID`, …) | **skipped** |
| `*_DT`, `*_DATE` | dates — often Excel-mangled (`43:45.2`, `00:00.0`) | **skipped** |
| `*_NAME`, `*_DESC`, `*_USER` + name-like tokens | free text / names / audit users | **skipped** |
| remaining numeric dtype | e.g. `CHILDREN_CT` | **numeric** |
| remaining object dtype, ≤ 50 distinct | | **categorical** |

Missing values: numeric → median impute, categorical → mode impute (only inside the
custom privacy/ML encoders; sdmetrics handles NaNs itself).

> ⚠️ Data caveat: scientific-notation IDs (`2.68856E+17`) and time-like date values
> (`43:45.2`) indicate the CSVs passed through Excel. IDs/dates are skipped from all
> metrics, so results are unaffected, but re-export from source if you need those fields
> synthesized faithfully.

---

## 1. Fidelity (sdmetrics `QualityReport`)

| Metric | Source | What it measures | Range / target |
|---|---|---|---|
| **Column Shapes** | sdmetrics: `KSComplement` (numeric/datetime), `TVComplement` (categorical/boolean) per column | Marginal distribution similarity: 1 − Kolmogorov–Smirnov statistic, or 1 − total-variation distance of category frequencies | 0–1, higher better; ≥ 0.9 good |
| **Column Pair Trends** | sdmetrics: `CorrelationSimilarity` (numeric pairs), `ContingencySimilarity` (categorical pairs / mixed) | Whether pairwise relationships (correlations, contingency tables) are preserved | 0–1, higher better |
| **Overall Quality Score** | sdmetrics `QualityReport.get_score()` | Mean of the two properties above | 0–1, higher better |

Visuals: grouped-bar comparison across synthesizers; per-column shape-score heatmap
(columns × synthesizers) showing exactly which columns each model reproduces poorly;
per-column real-vs-synthetic distribution plots; correlation heatmaps; SDV
`get_column_plot` / `get_column_pair_plot` where available.

---

## 2. Privacy

| Metric | Source | What it measures | Range / target |
|---|---|---|---|
| **NewRowSynthesis** | sdmetrics (`sdmetrics.single_table.NewRowSynthesis`) | Fraction of synthetic rows that are *not* copies of real rows (numeric tolerance 1%) | 0–1; **PASS ≥ 0.9**, WARN ≥ 0.7 |
| **CategoricalCAP** | sdmetrics (`sdmetrics.single_table.CategoricalCAP`) | Correct Attribution Probability attack: risk that an attacker knowing key categorical fields infers a sensitive categorical field. Score is privacy protection (1 = safe). Run when ≥ 2 categorical columns exist | 0–1, higher better |
| **Membership Inference Attack (MIA)** | custom (`synth_eval.membership_inference_attack`) | Hold out 25% of real rows *before* fitting; features = distances to the k=5 nearest synthetic records; a RandomForest attacker tries to distinguish training members from holdouts. Reports attack **AUC** and accuracy | AUC ≈ 0.5 = no leakage. **PASS \|AUC−0.5\| ≤ 0.10**, WARN ≤ 0.20, else FAIL |
| **DCR (Distance to Closest Record)** | custom (`synth_eval.dcr_distributions`) | Nearest-neighbour distance real→synthetic vs a real→real baseline, in the mixed-encoded feature space. Ratio = median(real→synth) / median(real→real) | ratio ≥ 1 means synthetic rows sit no closer than real rows do to each other. **PASS ≥ 0.9**, WARN ≥ 0.5 |
| **DCROverfittingProtection** | sdmetrics (`sdmetrics.single_table.DCROverfittingProtection`) | Official sdmetrics parallel to MIA/DCR: compares whether synthetic rows sit closer to the **training** split than to the pre-fit **validation holdout**. 1.0 = no closer to training than to unseen data (no memorization); lower = overfitting/leakage. Reuses the same 25% holdout | 0–1, higher better. **PASS ≥ 0.9**, WARN ≥ 0.5. Also folded into the leaderboard privacy score |
| **DCRBaselineProtection** | sdmetrics (`sdmetrics.single_table.DCRBaselineProtection`) | Synthetic-vs-real DCR compared against a random-data baseline. Best-effort: skipped (with a note) when Excel-mangled numeric IDs overflow its random sampler | 0–1, higher better. **PASS ≥ 0.9**, WARN ≥ 0.5 |
| **Exact-match rate** | custom | % of synthetic rows identical to a real row on all modelable columns | **PASS ≤ 0.1%**, WARN ≤ 1% |

Visuals: 3-panel dashboard (MIA AUC with 0.5 safe band, DCR ratio with ideal-≥1 line,
exact-match %); per-table DCR density overlays for all synthesizers vs baseline.

---

## 3. ML Efficacy / Utility (sdmetrics ML-efficacy metrics, TSTR protocol)

A condensed set — one tree model plus one linear reference, chosen by target type.
The target column per table is auto-selected (categorical with 2–20 classes →
classification; else the highest-variance numeric → regression) and can be overridden
via the `TARGETS` dict in the notebook.

| Target type | sdmetrics metrics used | Score |
|---|---|---|
| binary categorical | `BinaryDecisionTreeClassifier`, `BinaryLogisticRegression` | F1 |
| multiclass (3–20 classes) | `MulticlassDecisionTreeClassifier` | macro F1 |
| numeric | `LinearRegression` | R² |

**Additional classification metrics** (sdmetrics only reports F1). For every
classification target we also report **accuracy**, **precision (macro)**,
**recall (macro)** and **F1 (macro)** — rows prefixed `DecisionTree ·`. All four
come from a *single* scikit-learn `DecisionTreeClassifier` fit (matching
sdmetrics' decision-tree family) on the shared mixed encoder
(median-impute + scale numeric; mode-impute + one-hot `handle_unknown='ignore'`
categorical), so they are mutually consistent and directly comparable across
training sources.

Robustness: before scoring, holdout rows whose **target class** never appeared
in the training split are dropped (an unseen label can't be predicted), and any
**feature** category present in the holdout but absent from training is mapped to
the training column's mode. Both adjustments are recorded in the row's `note`
and surfaced in the dashboard instead of a blank cell.

Protocol (**TSTR — Train on Synthetic, Test on Real**): each metric's model is trained
twice — (a) on the real training split (TRTR reference) and (b) on each synthesizer's
data — and always evaluated on the **same real holdout** via
`Metric.compute(test_data=real_holdout, train_data=…, target=…)`.

Reported: score per training source, plus **`gap(real-<synth>)` = TRTR − TSTR** per
metric (≈ 0 means the synthetic data is as useful as real data for ML). Saved to
`reports/ml_efficacy_tstr.csv` with a grouped-bar comparison figure.

---

## 4. Leaderboard aggregation (`reports/leaderboard.csv`)

One 0–1 score per synthesizer per axis, averaged over tables:

| Dimension | Formula |
|---|---|
| **fidelity** | mean QualityReport overall score |
| **privacy** | mean of [ 1 − 2·\|MIA AUC − 0.5\| , min(1, DCR ratio) , NewRowSynthesis , DCROverfittingProtection , 1 − exact-match rate ] |
| **utility_tstr** | mean over (table × metric) of clip(TSTR score / TRTR score, 0, 1) |
| **overall** | mean of the three dimensions |

Rendered as an annotated RdYlGn heatmap; all raw numbers land in `reports/summary.json`.
