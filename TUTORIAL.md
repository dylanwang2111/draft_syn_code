# Synth/Lab — how to use it

A five-minute tour. Metric definitions live in [METRICS.md](METRICS.md).

## 0. Start it

```bash
source /home/dylannguyen/trent_master/synthetic/.venv/bin/activate
uvicorn server:app --reload      # --reload picks up code edits without a restart
```
Run it **from the project root**, or *Load Sample* won't find `sdg/seed/`. Open
<http://localhost:8000>. Each **browser tab is its own session** — you can run a synthesis in one
tab and keep editing data in another without them touching each other.

## 1. Load data

**Upload** one or more CSVs (one file = one table, filename = table name), or **Load Sample**
for the bundled seed data. The profiler reads the tables and may show an **advisor banner** with
a suggested strategy — it's a suggestion, not a step; you can always override it below.

## 2. Schema tab — check the types

Every column is auto-typed. Fix any wrong call with the dropdown (**red = your override**), and
set a **PRIMARY KEY** per table if there is one.

> `id`, `datetime` and `unknown` columns are excluded from the privacy and ML metrics, and are
> refilled from the real data's marginals rather than modelled. Getting the sdtypes right here is
> the single highest-leverage thing you can do.

## 3. Data Model tab — how the tables relate

Skip this if your tables are independent. Otherwise:

- **Drag a port → another port** to draw a relationship. A sidebar opens: pick the two columns
  and the cardinality (1:∗, ∗:1, 1:1), then **Save**. Nothing is committed until you save.
- **Click an arrow** to edit it; **Delete/Backspace** removes the selected one.
- **Entity-key hub** — pick a key shared by several tables (e.g. `CONT_ID`), tick the tables it
  should link, and **Generate hub**. This derives a parent table of distinct keys so the child
  tables hang off it. This is what lets HMA preserve referential integrity by construction.
- **Validate** structurally checks the model (key uniqueness, FK coverage) before you spend
  minutes fitting. **Auto-layout** re-tidies the canvas; new cards never spawn on top of old ones.

## 4. Configure the run (left panel)

| Step | What to set |
|---|---|
| **1 · Structure & keys** | Mirrors the Data Model. Set the SCD timeline columns here if your data is slowly-changing (effective / end / current). |
| **2 · Constraints** | Rules the synthetic data must satisfy *by construction*: `low ≤ high`, `low ≤ mid ≤ high`, columns co-vary, multiple of N. |
| **3 · Synthesizers** | **HMA** is multi-table (the only one that learns cross-table structure). GaussianCopula is fast and single-table. CTGAN / TVAE / CopulaGAN are neural — slower, and the **epochs** slider only applies to them. Pick two or more to compare. |
| **4 · Run parameters** | **Scale** = synthetic rows ÷ real rows. **Holdout** = real rows held back for the privacy and utility tests (never shown to the synthesizer). **Target** per table for the ML-efficacy test (auto is usually fine). |

Hit **▶ Synthesize**. The progress bar and console track the run; **■ Cancel** stops everything.
Fitting HMA on a key with many distinct values is the slow part — expect minutes, not seconds.

## 5. Read the report

Three dimensions, each scored 0–1, each with its own tab. Every score card prints the **formula
with your actual numbers in it**, so the headline can always be reconstructed by hand.

- **Leaderboard** — `overall` = mean of the three dimensions below.
- **Fidelity** = `(2 × column statistics + referential integrity) / 3`.
  - *Column Shapes* — do the individual columns look right?
  - *Column Pair Trends* — do the relationships *between* columns survive? (usually the weak one)
  - *Referential Integrity* — FK validity (no orphans), participation (parents with ≥1 child —
    should **match real**, not be maximised), cardinality shape.
- **Utility** — mean of one ratio `clip(synth ÷ real, 0, 1)` per (table × metric) panel. It's a
  **mean of ratios, not a ratio of means** — the `÷ real` column shows each term.
- **Privacy** — MIA protection (attacker AUC, ideal **0.5**), NewRowSynthesis, CategoricalCAP.

Download any synthetic table as CSV from the **Synthetic Data** panel on the left.

## Reading the scores honestly

- **Utility is relative.** If the real-trained baseline is weak (say 0.33 macro-F1), a utility of
  0.9 only says the synthetic data is *as weak as the real data* on that target — not that the
  model is good. Check the raw `real` column, not just the ratio.
- **A perfect QualityReport is not a faithful database.** It's single-table and never looks across
  a foreign key — that's exactly why referential integrity is folded into fidelity.
- **Parent coverage should not be 100%.** Match the real value; a synthesizer that gives *every*
  parent a child has invented structure that isn't there.
