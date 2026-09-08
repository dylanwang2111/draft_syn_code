# Column & PII Handling

How columns are classified before any metric or synthesizer touches them, and how
personally-identifying columns are kept out of the synthetic output. Split out from
`METRICS.md` since this is a data-preparation concern, not a scoring one — every
metric and every synthesizer builds on the same classification and PII plan.

---

## Column classification (`synth_eval.classify_columns`)

Every column in every table is assigned exactly one role before fitting or scoring:
**numeric**, **categorical**, or **skipped**. Priority order, first match wins:

1. **Schema suffix conventions** (checked before anything else, including SDV's own
   metadata):

   | Suffix | Example | Role |
   |---|---|---|
   | `_id`, `_dt`, `_date`, `_name`, `_desc`, `_user` | `CONT_ID`, `EFFECTIVE_DT`, `CONTACT_NAME` | **skipped** |
   | `_tp_cd`, `_tp_code`, `_cd`, `_code`, `_ind` | `MARITAL_ST_TP_CD`, `SOLICIT_IND` | **categorical**, unless it has more than `max_categorical_card` (default 50) distinct values, in which case it's id-like and **skipped** |

2. **SDV metadata sdtype**, if the column wasn't already caught by a suffix:
   `id`/`datetime`/`unknown` → skipped, `numerical` → numeric, `boolean` →
   categorical, `categorical` → categorical *unless* it looks like a name/id (see
   below) or exceeds the cardinality cap, in which case skipped. SDV often labels
   free-text name/id columns "categorical"; one-hot encoding those adds no signal
   while breaking TSTR on any holdout category never seen in training.

3. **Name/id heuristic** (`_looks_like_id_or_name`), when there's no usable
   metadata: word-boundary match against a token list (`id`, `guid`, `uuid`, `key`,
   `name`, `fname`, `lname`, `surname`, `email`, `phone`, `address`, `uri`, `url`),
   plus a start/end-with check for `id`/`guid`/`uuid`.

4. **pandas dtype + cardinality**, the final fallback: numeric dtype → numeric,
   *unless* it's a low-cardinality integer code (≤10 distinct values, all within
   -1..10) in which case it's treated as categorical; datetime dtype → skipped;
   anything else → categorical if ≤50 distinct values, else skipped (likely free
   text).

`ColumnRoles.modelable` = numeric + categorical: the columns that actually feed
distributions, correlations, the MIA/DCR/nearest-record distance encoders, and
ML-efficacy models. Skipped columns (mostly id/date/name/audit columns) are
excluded from every metric and are refilled from real rows after the modelable
columns are synthesized (`_refill`, see below) — see PII handling further down
for why that refill doesn't just leak real strings back out.

Missing values inside the shared mixed encoder (used by MIA, DCR, nearest-record,
and the reject-and-resample filter): numeric → median impute, categorical → mode
impute. sdmetrics' own metrics handle NaNs themselves.

> ⚠️ Data caveat: scientific-notation IDs (`2.68856E+17`) and time-like date values
> (`43:45.2`) indicate the CSVs passed through Excel. IDs/dates are skipped from
> every metric already, so results are unaffected, but re-export from source if you
> need those fields synthesized faithfully.

---

## Refilling skipped columns (`backend.dashboard_core._refill`)

Skipped columns aren't modeled by the synthesizer at all (see "why not" in PII
handling below), they're filled in afterward from real rows. Why not model them
directly: they're free text or near-unique per row, so there's no repeating
pattern for a synthesizer to generalize from — a neural model asked to generate
something close to unique per row tends to memorize and regurgitate real training
values rather than invent new ones, which would be *worse* for privacy, not
better, and there'd be no reliable way to tell a memorized real value apart from
a genuinely invented one afterward. Refilling from a real record we explicitly
control (and can run through the PII plan below) is the privacy-safe path.

Two layers, in order:

1. **Joint row sampling.** All of a table's skipped columns for one output row
   are drawn from the *same* sampled real row, not resampled independently
   column-by-column. This keeps whatever real correlation exists *between*
   skipped columns intact — e.g. a `NAME` and its matching `DESC` — which
   independent per-column bootstrapping would otherwise reduce to near
   coincidence (verified on real data: >90% mismatched pairs under the old
   per-column version, <1% once sampled jointly).

2. **Conditioning on a modeled column, when one is genuinely tied to it.** Joint
   row sampling alone still can't fix a skipped column's relationship to a
   *modeled* one — e.g. a lookup table's `NAME` against its own `OCCUPATION_TP_CD`
   — because the modeled column's synthetic value came from the synthesizer, not
   from whatever real row got sampled for the skipped columns; a random real row
   has no reason to share it. So before sampling, each skipped column is checked
   against every modeled categorical column in the same table via
   `synth_eval.group_diversity_reduction` (how much does grouping real rows by
   the candidate column narrow down the skipped column's values, relative to its
   overall diversity — 0 = no association, approaching 1 = the candidate nearly
   determines it) and `synth_eval.best_refill_group_column` (the strongest
   candidate, provided it clears a minimum-association threshold, default 0.5).
   This is measured directly from the real data, not assumed from column names,
   so it generalizes to a schema that's never been seen before rather than only
   the ones whose naming happens to hint at the relationship.

   Skipped columns sharing the same best-matching modeled column are grouped and
   sampled together: for each output row, a real row is drawn from the subset of
   real rows that share that row's *already-synthesized* value of the modeled
   column, instead of from the whole table. Skipped columns with no modeled
   column clearing the threshold fall back to the plain joint whole-table sample
   (layer 1).

   **Privacy floor:** conditioning only fires if the matching real group has at
   least `min_group_size` rows (default 10); a rarer code shared by only a
   couple of real people would otherwise make the refilled row too easy to trace
   back to one of them, so it falls back to the whole-table sample instead. The
   same floor covers a synthesized code value that doesn't exist anywhere in the
   real data at all (an empty group is, trivially, below the floor).

   Intentional v1 scope boundary: a skipped column conditions on at most one
   modeled column (the single strongest match), not a joint match on several at
   once — joint conditioning would shrink the matching group much faster, which
   is worse for both data availability and privacy, for a case not yet shown to
   matter in practice.

3. **Merging a real ordering relationship BETWEEN two skipped columns, even
   across two different conditioning groups.** Layer 2 picks each skipped
   column's best-matching modeled column independently — so an
   effective-date/end-date pair (or any other pair with a genuine real
   invariant `low <= high` between *themselves*, not a shared tie to some
   other field) can end up conditioned on two DIFFERENT modeled columns and
   get sampled from two DIFFERENT real rows, breaking that invariant even
   though layer 1's whole-row sampling exists specifically to prevent this.
   Flagged directly from a manual walkthrough of HMA-synthesized data against
   production (an effective date later than its own row's end date).

   Detected the same way as layer 2 — measured directly from the real data,
   never guessed from column names (`synth_eval.detect_ordered_date_pairs`:
   does `low <= high` hold for virtually every real row with both present?
   — robust to an end-date column being mostly null by design, e.g. most
   entities are still on their first, still-open version). Any detected pair
   split across two clusters is merged into one before sampling
   (`backend.dashboard_core._merge_ordered_date_clusters`), preferring to
   keep whichever cluster is conditioned on a modeled column over an
   unconditioned one. A pair is never assumed disjoint from another — three
   date columns that all pairwise order (`created <= effective <= end`) all
   end up in one merged group, not just two of them.

---

## SCD timeline repair (`synth_eval.scd`)

A synthesizer generates each version row of an SCD-2 table independently, so
one entity's effective/end windows can overlap, leave gaps, or — the failure
mode this section is about — leave more than one row looking "current" at
once. `synth_eval.repair_scd_timeline` fixes this per entity: sort by
effective date, tile the windows (each version's end = the next version's
start), leave only the last version open.

**Manual config, per table.** The UI's Schema tab lets a run specify an
effective-column / end-column / current-flag-column selection *per table*
(`cfg["scd"] = {table: {effective, end, current}}`, mirroring the per-table
PII-policy config). This used to be one name triple applied identically to
every table regardless of that table's own naming — which meant a table
versioned under different column names than whichever one was configured
was silently skipped, and there was no way to cover two independently-named
SCD tables in the same upload at once. Left blank, a table falls through to
automatic detection below; the manual selection exists only as an override
for when auto-detection picks the wrong pair or misses one — it always wins
over the automatic pass for whichever `(table, entity_key)` it covers.
Each row's placeholder text shows what auto-detection actually found for
that table (`POST /api/scd_preview`, which just calls
`detect_scd_window_pair` directly against the entity key currently chosen in
the Data Model tab) — so a correct auto pick is visible without opening the
dropdown, and leaving it on "(auto)" is a genuine no-op, since it's the same
detector that runs for real at synthesis time.

**Automatic, per table (`detect_scd_window_pair`, `find_mirror_pair`).**
Every entity-keyed table not covered by a manual override gets an automatic
pass. This is what closed the original gap: confirmed live, a run where
PERSONNAME's own `(START_DT, END_DT)` was never configured left a relinked,
hub-fabricated entity with two rows both holding an open (`9999-12-31`) end
date — two "current" versions of the same entity at once.

1. `detect_scd_window_pair(real, entity_key, cols)` finds the table's own
   version-boundary pair, straight off its real data, never guessed from
   column names. Two conditions, both required: (a) `entity_key` has genuine
   multi-row (versioned) entities in the real data — nothing to repair
   otherwise; (b) among `detect_ordered_date_pairs`' candidates, the "high"
   column must carry an "open" signal for a meaningful share of rows —
   either of two conventions, checked independently since a source system
   can use either: a dominant, repeated sentinel value (a true end-date
   column lands on it for many rows, unlike a per-row audit timestamp like
   created-at/updated-at, which is ~always distinct per row and has no such
   spike — confirmed on real data: `END_DT`/`IDP_END_DATE` both cluster 93%
   of rows on one value, `LAST_UPDATE_DT`'s most common value covers 0.1%),
   or NULL meaning "still open" instead of a sentinel — equally common in
   practice, and checked as its own signal because dropping NULLs first (as
   the sentinel check does) would make a NULL-convention column look like a
   sparse, unrelated column with no dominant value among what's left.
2. `find_mirror_pair(real, low, high, cols)` checks for another column pair
   that duplicates the chosen one value-for-value — a common ETL pattern
   (e.g. a data-warehouse load-audit pair mirroring a business start/end
   pair exactly). Repairing only the primary pair would trade one
   inconsistency (two open rows) for another (the repaired pair and its
   untouched mirror now disagreeing on the same row), so the mirror gets
   tiled identically.

Entity groups too small to have any real versioning, or where no candidate
column shows a genuine open-sentinel spike, are left alone rather than
forcing a flag column to mean something it measurably doesn't — a first
version of this checked `USE_STANDARD_IND`-style "current name" flags for
exactly-one-`Y`-per-entity, but real PERSONNAME data shows 88/114 multi-row
entities already carry more than one `Y` even before synthesis, so treating
that as a violation to correct would just be inventing a cleanliness the
source system itself doesn't have. The date-window tiling (no two rows open
at once) is the well-defined, generalizable invariant; a flag column's
business meaning is not assumed.

**Duration fidelity (`scd_duration_fidelity`).** Repair above only
guarantees *structural* correctness — a tiled, non-overlapping timeline,
one open row per entity. It says nothing about whether the *spacing*
between one entity's own successive versions looks real, because that
spacing comes from wherever the effective dates came from — independently
synthesizer-generated per row, then in many pipelines regrouped into
entities post-hoc by relinking — and repair has no visibility into either.
Confirmed directly: simulating relinking on real PERSONNAME data and then
running the existing, structurally-correct repair still leaves post-repair
durations (`end − effective`, closed rows only) diverging from real ones at
KS stat 0.215 (p=0.008) — correctness and this kind of distributional
fidelity are different questions, and a table can have the former without
the latter.

`scd_duration_fidelity(real, synth, entity_key, effective_col, end_col)`
measures this directly: a two-sample KS test between real and synthetic
closed-row durations, for whichever pair `detect_scd_window_pair` found (or
whatever was manually configured) — reported per table per synthesizer
(`results["scd_duration_fidelity"]`, and the "SCD timeline duration
fidelity" table in the report's Referential Integrity section, shown as
`1 − KS stat` so higher still reads as closer to real). `None` when either
side has fewer than `min_closed` (default 10) closed rows — too little to
trust a distribution comparison. This is a *measurement*, not a fix: it
exists to make an already-real gap visible rather than to close it.

---

## PII handling (`synth_eval.pii`, `apply_pii_plan`)

The skipped id/date/name/audit columns above aren't discarded — they're refilled
from real rows as described above (keeps fitting fast on wide tables too). Because
the *values* are genuinely real, a name or email column refilled this way would
put real strings into the "synthetic" output as-is. `synth_eval.pii` exists
specifically to catch that.

### Detection (`detect_pii`)

Three passes:

1. **Column-name tokens**, checked in order, first hit wins: `EMAIL`/`E_MAIL` →
   email; `PHONE`/`TELEPHONE`/`_TEL`/`TEL_`/`FAX`/`MOBILE`/`CELL` → phone;
   `POSTAL`/`ZIP` → postal; `ADDR_LINE`/`ADDRESS_LINE`/`STREET`/`_ADDR`/`ADDR_` →
   street; `NAME`/`_NM`/`NM_`/`SURNAME`/`GIVEN`/`_USER`/`USER_` → name. Deliberately
   **no bare FIRST/LAST/MIDDLE tokens** — `LAST_VERIFIED_TRANSIT` isn't a name;
   real name columns virtually always carry `NAME` or `NM` somewhere. A suffix
   match against `_CD`/`_CODE`/`_ID`/`_IND`/`_DT`/`_DATE`/`_TP`/`_TYPE`/`_CT`/
   `_QTY`/`_AMT`/`_PCT` overrides everything — `PREFIX_NAME_TP_CD` is a type code,
   not a name, whatever "NAME" appears in it.

2. **Content confirmation, `name` only** (`_confirms_person_name`). The `NAME`
   token still matches columns that hold something else entirely — an
   `OCCUPATION_NAME` column holding job titles ("Registered Nurse", "Financial
   Analyst"), or a `LAST_UPDATE_USER` column holding a batch-job tag
   (`SYS_BATCH`), not a person. A sample of the column's values is checked
   against Faker's own first/last-name corpus (the first or last whitespace
   token must match a known first or last name); if fewer than 15% do, the
   `name` classification is dropped and the column falls through to the
   value-shape pass below. Checking value *content* against a name corpus,
   rather than hardcoding more schema-specific exceptions onto the token list,
   is what makes this generalize to any schema's naming conventions instead of
   just this one's.

3. **Value-shape regexes**, only for object-dtype columns *outside* the modelable
   set that passes 1–2 didn't already catch, so a low-cardinality code column
   can never be misread as a postal code just because a handful of its values
   happen to look like one. Checked against a sample of up to 60 non-null values,
   ≥60% must match: email, postal code (Canadian `A1A 1A1` or US 5/9-digit ZIP),
   street address, then phone last (loose digit pattern, requires a median of ≥7
   digits per value so it doesn't fire on short codes).

### Policy per column, chosen in the UI (default: `fake`)

| Policy | Effect |
|---|---|
| **`fake`** (default) | Replace every value with a **Faker-generated** value that never existed in the source (`en_CA` locale), preserving the column's original missing rate. `name` further specializes by column name: `FIRST`/`GIVEN`/`MIDDLE` → first names only, `LAST`/`SURNAME` → last names only, else a full name. Deterministic for a given (kind, row count, seed, column name), so reruns with the same seed reproduce the same fake values (see the `random_state` wiring in `synth_eval.suite`). **Entity-consistent** on a table with a shared entity key (e.g. `CONT_ID` on an SCD-versioned table with several history rows per real customer): every row sharing that key gets the SAME faked value and the SAME missing/not-missing status, not an independent re-roll per row. Without this, a synthesizer that correctly reused one CONT_ID across several dated rows would still show that "same customer" with a different name on every row — confirmed against real seed data (517/517 multi-row contacts got a different fake name per row before this; 0/517 after). Falls back to a fully independent per-row draw on a table with no declared entity key. |
| **`shuffle`** | Leave the bootstrap refill as-is — i.e. **real values, shuffled across rows**. Flagged loudly in the run log (`⚠ kept REAL values (shuffled) in …`) since this is the one path that puts real PII into the output; it exists for columns where the shuffled-real value is actually needed (e.g. a downstream join key that must stay a real, valid-looking value) and the risk is accepted deliberately. |
| **`drop`** | Remove the column from the synthetic output entirely. |

Faked/dropped columns are deliberately *not* faithful to the real marginals (that's
the point) — they're excluded from the evaluation metadata so QualityReport neither
crashes on a dropped column nor scores fidelity that was intentionally destroyed
for privacy. Shuffled columns keep the real marginal, so they stay scoreable.

### In the UI

The Schema tab flags every auto-detected PII column inline (a red **PII** badge
next to the column name, with the detected kind in the tooltip) so a reviewer can
see what will be faked without switching tabs. The Data tab's privacy panel lets
each column's policy be overridden per table before a run.

### What this does and doesn't cover

Detection and faking neutralizes **direct identifiers** — names, emails, phones,
addresses. It does not by itself address **quasi-identifiers**: ordinary modelable
columns (status codes, categorical attributes, numeric fields) whose *combination*
can still be unique enough to re-identify someone even with every direct identifier
correctly faked. That risk is what the privacy metrics (§2 in `METRICS.md`,
especially the nearest-record check) and the reject-and-resample filter are for —
faking the obvious identifier doesn't neutralize risk from the quasi-identifiers
sitting next to it.
