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

## 1. Fidelity (sdmetrics `QualityReport` + referential integrity)

Fidelity has two halves: the **column statistics** below (how each table looks on its
own) and **referential integrity** (how the tables relate — FK validity, participation
and cardinality, §4). Both feed the single fidelity number on the leaderboard.

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
| **NewRowSynthesis** | sdmetrics (`sdmetrics.single_table.NewRowSynthesis`) | Fraction of synthetic rows that are *not* copies of real rows (numeric tolerance 1%), on the **modelable columns only**. Verdict is judged against the **real-holdout baseline** on the same columns — on a low-entropy projection (a few code columns) even real rows duplicate each other, so the achievable ceiling can be far below 1 | 0–1; **PASS = within 0.05 of the real-holdout ceiling**, WARN within 0.20 |
| **CategoricalCAP** | sdmetrics (`sdmetrics.single_table.CategoricalCAP`) | Correct Attribution Probability attack: risk that an attacker knowing key categorical fields infers a sensitive categorical field. Score is privacy protection (1 = safe). Run when ≥ 2 categorical columns exist | 0–1, higher better |
| **Membership Inference Attack (MIA)** | custom (`synth_eval.membership_inference_attack`) | Hold out 25% of real rows *before* fitting; features = distances to the k=5 nearest synthetic records; a RandomForest **attacker** tries to distinguish training members from holdouts. Reports attack **AUC** | AUC ≈ 0.5 = no leakage. **PASS \|AUC−0.5\| ≤ 0.10**, WARN ≤ 0.20, else FAIL |

> The privacy set is deliberately kept to **three** metrics — one per distinct
> attack: **MIA** (membership inference), **NewRowSynthesis** (copying),
> **CategoricalCAP** (attribute disclosure). MIA and sdmetrics'
> `DCROverfittingProtection` test the same membership-inference threat; we report
> the trained-attacker AUC framing. The other custom helpers (`dcr_distributions`,
> `exact_match_rate`) remain in `synth_eval.privacy` for ad-hoc use but are not
> reported.

Visuals: 3-panel dashboard — NewRowSynthesis (ideal 1), MIA attacker AUC (ideal
0.5, safe band), CategoricalCAP (ideal 1).

---

## 3. ML Efficacy / Utility (sdmetrics ML-efficacy metrics, TSTR protocol)

A condensed set — one tree model plus one linear reference, chosen by target type.
The target column per table is auto-selected (categorical with 2–20 classes →
classification; else the highest-variance numeric → regression) and can be overridden
via the `TARGETS` dict in the notebook.

| Target type | sdmetrics metric (headline) | Score |
|---|---|---|
| binary categorical | `BinaryDecisionTreeClassifier` | F1 |
| multiclass (3–20 classes) | `MulticlassDecisionTreeClassifier` | macro F1 |
| numeric | `LinearRegression` | R² |

One sdmetrics tree model per target type — binary and multiclass are symmetric
(the previous second binary logistic model just duplicated F1 and was removed).

**Additional classification metrics** (sdmetrics only reports F1). For every
classification target we also report **accuracy**, **precision (macro)** and
**recall (macro)** — rows prefixed `DecisionTree ·` — from a *single* scikit-learn
`DecisionTreeClassifier` fit on the shared mixed encoder (median-impute + scale
numeric; mode-impute + one-hot `handle_unknown='ignore'` categorical). These are
distinct lenses on the same predictions; **F1 is intentionally not repeated
here** — it is the sdmetrics headline metric above.

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
| **column_fidelity** | mean QualityReport overall score — itself the mean of Column Shapes and Column Pair Trends |
| **referential_integrity** | sdmetrics `CardinalityShapeSimilarity` — see below. `NaN` when no relationships are defined |
| **fidelity** | (2 × column_fidelity + referential_integrity) / 3, or column_fidelity alone when there are no relationships |
| **privacy** | mean of the three protection scores [ 1 − 2·\|MIA AUC − 0.5\| , NewRowSynthesis , CategoricalCAP ] |
| **utility_tstr** | mean over (table × metric) of clip(TSTR score / TRTR score, 0, 1) |
| **overall** | mean of fidelity, privacy and utility_tstr |

> **utility is a mean of ratios, not a ratio of means.** Each (table × metric) *panel*
> contributes one ratio and they are averaged with equal weight, so
> `mean(synth) / mean(real)` will **not** reproduce the score — a panel with small scores
> counts as much as a panel with large ones, and `clip(…, 0, 1)` caps a panel where the
> synthesizer beat the real baseline. The dashboard prints every panel ratio (the "÷ real"
> column) plus the per-table means, so the headline can be added up by hand.
> Also note utility is *relative*: against a weak baseline (real-trained model scoring 0.33)
> a high ratio only says the synthetic data is as weak as the real data, not that the model is good.

`QualityReport` is single-table only: it never looks across a foreign key, so a
synthesizer can score a perfect 1.0 on it while getting the cross-table structure wrong.
**referential_integrity** (`synth_eval.compare.structure_scores`) closes that gap. The
score is **`CardinalityShapeSimilarity` alone** — the distribution of child rows per
parent (including parents with none). It is the only one of the three quantities below
that is a *distribution-similarity* measure on the same footing as Column Shapes, which
is why it is the only one averaged into fidelity.

| Quantity | Formula | Role |
|---|---|---|
| **cardinality shape** | sdmetrics `CardinalityShapeSimilarity` | **the score.** Averaged into fidelity |
| **FK validity** | share of synthetic child rows whose FK hits a parent key (forward coverage) | **pass/fail gate, not scored.** A *constraint*, not a similarity — averaging it in would let a model buy its way out of orphan rows with good marginals. It is also **1.0 by construction whenever the parent is a derived entity hub** (the hub is built from the keys the synthesizer emitted), so it is reported as `n/a` there rather than as a free 1.0 |
| **participation** | 1 − \|synth parent coverage − real parent coverage\|, where parent coverage = share of parents with ≥ 1 child row | **diagnostic, not scored.** Parent coverage is `P(count > 0)` — a single point on the CDF of the very distribution cardinality shape already measures in full, so scoring both would double-count. Still worth reading: it should **match real**, and is *not* supposed to be 1 |

The same numbers are surfaced per report tab in the dashboard (`results["summary"]`,
built by `compute_summary`), so the Fidelity / Utility / Privacy tabs each show the
headline score and the components behind it.

Rendered as an annotated RdYlGn heatmap; all raw numbers land in `reports/summary.json`.
