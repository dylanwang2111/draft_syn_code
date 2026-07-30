# Metrics Documentation

All metrics used by `notebook/synthetic_evaluation.ipynb` / `synth_eval/`, grouped by the three
evaluation axes. Metrics come from **sdmetrics** wherever one exists; the custom privacy
metrics (membership inference, nearest-record) cover privacy properties sdmetrics does not
measure directly, and the reject-and-resample filter acts on what they find rather than just
reporting it. Every score below is reported **per synthesizer per table** and rolled up into
the leaderboard.

---

## Column handling (applies to every metric)

Every column is classified numeric / categorical / skipped before any metric or
synthesizer runs, and PII columns (names, emails, phones, addresses) are detected
and faked separately from that classification. Both moved to their own doc, since
they're a data-preparation concern shared by every metric below, not a scoring
concern themselves: **[`docs/DATA_HANDLING.md`](DATA_HANDLING.md)**.

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
| **CategoricalCAP** | sdmetrics (`sdmetrics.single_table.CategoricalCAP`) | Correct Attribution Probability attack: risk that an attacker knowing key categorical fields infers a sensitive categorical field. Score is privacy protection (1 = safe). Run when ≥ 2 categorical columns exist. The sensitive column is **auto-picked as the most balanced categorical** (a heavily skewed field is guessable from population statistics alone, which fails every synthesizer identically) and is **user-selectable per table** in the run parameters. Verdict is judged against the **real-holdout baseline**: the same attack armed with real holdout rows instead of synthetic ones — if the field is inferable from the real data's own structure, that ceiling is low for everyone | 0–1; verdict uses two lenses and takes whichever is more forgiving for PASS, both for FAIL: **absolute gap** (`base - cap`, baseline-agnostic) and **relative loss** (`gap / (1 - base)`, i.e. "the attack succeeds X% more often than on real data" — needed because a flat gap means very different things at a 0.55 vs. a 0.95 baseline). **PASS if gap ≤0.05 OR relative loss ≤25%**, **FAIL only if gap >0.20 AND relative loss >100%**, else WARN |
| **Membership Inference Attack (MIA)** | custom (`synth_eval.membership_inference_attack`) | Hold out 25% of real rows *before* fitting; features = distances to the k=5 nearest synthetic records; a RandomForest **attacker** tries to distinguish training members from holdouts. Reports attack **AUC** | AUC ≈ 0.5 = no leakage. **PASS \|AUC−0.5\| ≤ 0.10**, WARN ≤ 0.20, else FAIL |
| **Nearest-record check** | custom (`synth_eval.nearest_real_examples`) | The literal "can a synthetic row be traced back to a real one" check: find the CLOSEST synthetic row to any real (training) row across the whole table (worst case, not a random sample), and grade its distance against a **bootstrap ceiling** built from the real holdout doing the exact same "scan N rows, take the smallest distance" procedure against the training data. A raw minimum is an extreme-value statistic — scan enough rows and *some* minimum will look small even with zero leakage, so comparing it to a *typical* real-to-real distance would be comparing a minimum to a median. The ceiling fixes that by comparing minimum-to-minimum, same-size sample, computed on the real holdout instead | distance in the shared mixed-encoder space; **PASS if the synthetic minimum ≥ the holdout's 5th-percentile bootstrap minimum**, WARN if it's within half that ceiling, else FAIL |

> The privacy set is deliberately kept to **four** checks — one per distinct
> attack: **MIA** (membership inference), **NewRowSynthesis** (copying),
> **CategoricalCAP** (attribute disclosure), **nearest-record** (worst-case
> reconstruction). MIA and sdmetrics' `DCROverfittingProtection` test the same
> membership-inference threat; we report the trained-attacker AUC framing. All
> four feed the leaderboard's numeric `privacy` score (§4) as an equal-weight
> mean; the nearest-record term is `clip(closest synthetic-to-real distance ÷
> the real-holdout bootstrap ceiling, 0, 1)`, the same ratio the PASS/WARN/FAIL
> verdict already uses, so the term and the verdict agree by construction. The
> actual matched row pair is still attached to the report (§ visuals below) so
> the finding is inspectable, not just a number to trust. The other custom
> helpers (`dcr_distributions`, `exact_match_rate`) remain in
> `synth_eval.privacy` for ad-hoc use but are not reported.

Visuals: 3-panel dashboard — NewRowSynthesis (ideal 1), MIA attacker AUC (ideal
0.5, safe band), CategoricalCAP (ideal 1) — plus a dedicated nearest-record panel
(per synthesizer / table) showing the closest synthetic row next to its real match
side by side, identical cells highlighted, so the check is demonstrable, not just
a number.

### Reject-and-resample filter (`filter_close_records` / `filter_close_records_multitable`)

An **active mitigation**, not a metric: after generation, every synthetic row is
checked against the same nearest-record test above, and any row sitting closer to
a real (training) row than real rows ever sit to *each other* (below the 5th
percentile of real-to-real nearest-neighbor distances) is dropped and replaced
with a fresh draw from the same fitted model, for up to a few retry rounds. If the
model can't produce enough clean replacements, the output comes back short rather
than keeping a risky row just to hit a row count.

Runs automatically for every synthesizer. For single-table models
(GaussianCopula/CTGAN/TVAE/CopulaGAN) it's a straightforward per-table filter. For
HMA, rows are linked across tables by shared keys, so removing one has to cascade
to every descendant row that references it, and refilling means pulling whole
fresh **linked groups** out of a new sampled batch, not individual rows. Every
table with both real data and a `roles` entry is checked, unless an ancestor table
is *also* such a table (that ancestor's cascade already covers it) — one rule that
handles a plain declared parent/child pair, no relationships at all (every table
filtered independently, same idea as the single-table filter but sharing one
resample per retry round), and entity-key/hub mode (the derived hub table has no
real data of its own, so its real children are each checked independently instead
of silently skipped, which is what an earlier version of this filter got wrong).

Reported per synthesizer per table as `close_filter` in the results: rows in, rows
rejected, rows successfully resampled, the distance threshold used, and (for HMA)
how many descendant rows were cascaded away per table.

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
| **privacy** | mean of the four protection scores [ 1 − 2·\|MIA AUC − 0.5\| , NewRowSynthesis , CategoricalCAP , clip(nearest-record distance ÷ bootstrap ceiling, 0, 1) ] |
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
