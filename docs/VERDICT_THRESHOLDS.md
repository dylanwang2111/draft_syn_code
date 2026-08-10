# Verdict Thresholds (PASS / WARN / FAIL)

Quick reference for exactly when each score lands in the yellow WARN band. Every
threshold below is pulled straight from the source lines that compute it, so if the
code changes, re-check this doc against it rather than trusting it blind.

Three different kinds of "yellow" exist in the dashboard and they should not be
confused:

* **Scored verdicts** ("§1" below): a `PASS` / `WARN` / `FAIL` / `SKIP` string
  attached to a metric result, driving the `.pill` badges in the report tables.
* **Business-view dimension verdicts** ("§2" below): `Ready` / `Review` / `Not
  ready` badges on the three headline dimensions (Realism = fidelity, Safety =
  privacy, Usefulness = utility) and the overall per-synthesizer recommendation
  in the exec/business view.
* **UI-only delta coloring** ("§3" below): informational color-coding on numbers
  that never produce a PASS/FAIL verdict of their own (they annotate a scored
  metric or show a worked example). Yellow here means "worth a look," not "gated."

---

## 1. Scored verdicts

### Privacy (`synth_eval/privacy.py`)

| Metric | PASS | **WARN** | FAIL | SKIP |
|---|---|---|---|---|
| **Membership Inference (MIA) attacker AUC** | `\|AUC − 0.5\| ≤ 0.10` | `0.10 < \|AUC − 0.5\| ≤ 0.20` | `\|AUC − 0.5\| > 0.20` | AUC not computable |
| **NewRowSynthesis** *(real-holdout baseline available)* | gap ≤ 0.05 | `0.05 < gap ≤ 0.20`, where `gap = baseline − NewRowSynthesis` | gap > 0.20 | n/a |
| **NewRowSynthesis** *(no baseline, fallback)* | NRS ≥ 0.9 | `0.7 ≤ NRS < 0.9` | NRS < 0.7 | n/a |
| **CategoricalCAP** *(real-holdout baseline available)* | `gap ≤ 0.05` **OR** `relative loss ≤ 25%` | neither PASS nor FAIL condition met (see below) | `gap > 0.20` **AND** `relative loss > 100%` | not computable / not applicable |
| **CategoricalCAP** *(no baseline, fallback)* | CAP ≥ 0.4 | `0.3 ≤ CAP < 0.4` | CAP < 0.3 | not computable / not applicable |
| **Nearest-record distance** *(≥5 holdout rows, ceiling available)* | `min_dist ≥ ceiling` | `ratio ≥ 0.5`, where `ratio = min_dist / ceiling` | `ratio < 0.5` | fewer than 5 holdout rows |

**CategoricalCAP's WARN band, spelled out:** `gap = baseline − CAP`,
`headroom = 1 − baseline`, `relative loss = gap / headroom` (the attack succeeding
X% more often than it already does against real data). WARN is everything the two
lenses disagree on: e.g. a real-world example from this codebase, `CAP = 0.626`
vs. a `0.710` real-holdout baseline. `gap = 0.084` (fails the ≤0.05 PASS bar),
`relative loss = 0.084 / 0.290 ≈ 29%` (fails the ≤25% PASS bar too, but nowhere
near the FAIL bar of >100%) → WARN. The attacker's absolute success rate
(`1 − CAP ≈ 37%`) looking "under 50%" is not what's being judged; it's how much
*worse* that is than the ~29% an attacker already gets from the real data.

**Why WARN exists as a middle band at all (both metrics):** PASS only needs one
lens to look fine, FAIL needs both lenses to look bad. That leaves a deliberate
gap where the lenses disagree, e.g. a small absolute gap at a very high baseline
(where headroom is tiny and the same gap reads as a large relative loss) or vice
versa. That gap is graded WARN rather than forced either way.

### Referential integrity (`backend/dashboard_core.py`)

| Check | PASS | **WARN** | FAIL |
|---|---|---|---|
| **Pre-run model validation**, FK coverage (`_validate_relationships`) | coverage ≥ 0.99 | `0.90 ≤ coverage < 0.99` | coverage < 0.90 |
| **Pre-run model validation**, child FK entirely null | — | always WARN | — |
| **Post-run per-relationship** FK coverage (`_referential_integrity`) | coverage ≥ 0.99 | `0.90 ≤ coverage < 0.99` | coverage < 0.90 |

### Structure / leaderboard FK gate (`synth_eval/compare.py`)

| Check | PASS | **WARN** | FAIL | n/a |
|---|---|---|---|---|
| **FK validity gate** (diagnostic, not scored into fidelity) | fk_validity ≥ 0.999 | `0.99 ≤ fk_validity < 0.999` | fk_validity < 0.99 | FK is by construction (derived parent) or not computable |

### ML-efficacy / utility (`synth_eval/efficacy.py`)

**No WARN band exists here.** F1, accuracy, precision, recall (and the R² /
gap-vs-TRTR numbers around them) are reported as raw scores and deltas, not
gated into PASS/WARN/FAIL — there is no yellow threshold to hit because there is
no verdict system on this axis at all. `auto_select_target` / `_predictive_signal`
can skip a target entirely (logged as `skipped (...)`) but that's a "did not
run" decision made before scoring, not a WARN verdict on a score.

---

## 2. Business-view dimension verdicts (`web/app.js`)

The exec/business view collapses each report tab into one 0-1 score per
synthesizer and grades it `Ready` (green) / `Review` (**yellow**) / `Not ready`
(red) with `scoreVerdict(score, good, ok)`:

```
score >= good  → Ready
ok <= score < good → Review   (yellow)
score < ok     → Not ready
```

| Dimension | Backs onto | good | ok | **Review / yellow band** |
|---|---|---|---|---|
| **Realism** (`sec-quality`, fidelity) | `summary[synth].fidelity.score` | 0.8 | 0.6 | `0.6 ≤ score < 0.8` |
| Column shapes (`sec-shapes`) | `summary[synth].fidelity.column_shapes` | 0.8 | 0.6 | `0.6 ≤ score < 0.8` |
| Column pair trends (`sec-pairs`) | `summary[synth].fidelity.column_pair_trends` | 0.8 | 0.6 | `0.6 ≤ score < 0.8` |
| Referential integrity (`sec-ri`) | `summary[synth].fidelity.structure` | 0.8 | 0.6 | `0.6 ≤ score < 0.8` |
| **Usefulness** (`sec-utility`) | `summary[synth].utility.score` | 0.85 | 0.7 | `0.7 ≤ score < 0.85` |
| **Safety** (`sec-privacy`) | *(gate, not a score threshold, see below)* | n/a | n/a | n/a |

**Safety is a gate, not a threshold** (`safetyVerdict`): it takes the *worst* of
every privacy metric's own §1 verdict, across every table, for that
synthesizer. A single `FAIL` anywhere makes Safety `Not ready`; a `WARN`
anywhere (with no `FAIL`) makes it `Review`; all-`PASS`/`SKIP` makes it `Ready`.
It doesn't re-threshold the averaged privacy score itself.

**Overall per-synthesizer verdict** (`recommendation()`, used for the top
recommendation banner and each generator's summary card):
`worstVerdict([Realism, Safety, Usefulness])`, i.e. the worst of the three
dimension verdicts above. So the overall badge shows **Review (yellow)**
whenever *any one* of Realism/Safety/Usefulness lands in its own yellow band
(or Safety hits a WARN-level privacy check), even if the other two are `Ready`.
It only shows **Not ready** if at least one dimension is `Not ready`
(equivalently: Realism/Usefulness below their `ok` floor, or Safety has a
`FAIL`-level privacy check).

---

## 3. UI-only delta coloring (informational, not a gate)

These live in `web/app.js` and color a number without ever setting a PASS/FAIL
status on it.

| Where | ok / green | **warn / yellow** | bad / red |
|---|---|---|---|
| **Parent-coverage delta badge** (how far a synthesizer's reverse FK coverage sits from the real ratio) | `\|delta\| ≤ 5 pts` | `5 pts < \|delta\| ≤ 15 pts` | `\|delta\| > 15 pts` |
| **Nearest-record drill-down example**, percentile vs. real-to-real baseline spacing | percentile ≥ 50 | `20 ≤ percentile < 50` | percentile < 20 |

The second row colors the single worked example shown in the drill-down panel
(closest synthetic row vs. its real match). It is related to, but not identical
to, the aggregate **Nearest-record distance** verdict in §1, which grades the
true minimum distance across the whole table against the bootstrap ceiling, not
a percentile display of one example.

---

## Source lines, for re-checking after a code change

* `synth_eval/privacy.py:724-858`: MIA, NewRowSynthesis, CategoricalCAP, nearest-record
* `backend/dashboard_core.py:505,558,561`: pre-/post-run referential integrity
* `synth_eval/compare.py:525`: FK validity gate
* `web/app.js:1485-1518`: business-view dimension verdicts and gate logic
* `web/app.js:2123-2133`: per-tab good/ok thresholds (`TAB_DIM`)
* `web/app.js:1842`: parent-coverage delta badge
* `web/app.js:2057`: nearest-record drill-down percentile color
