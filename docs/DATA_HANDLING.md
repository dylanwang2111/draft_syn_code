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
excluded from every metric and are refilled independently by bootstrap sampling
after the modelable columns are synthesized — see PII handling below for why that
refill doesn't just leak real strings back out.

Missing values inside the shared mixed encoder (used by MIA, DCR, nearest-record,
and the reject-and-resample filter): numeric → median impute, categorical → mode
impute. sdmetrics' own metrics handle NaNs themselves.

> ⚠️ Data caveat: scientific-notation IDs (`2.68856E+17`) and time-like date values
> (`43:45.2`) indicate the CSVs passed through Excel. IDs/dates are skipped from
> every metric already, so results are unaffected, but re-export from source if you
> need those fields synthesized faithfully.

---

## PII handling (`synth_eval.pii`, `apply_pii_plan`)

The skipped id/date/name/audit columns above aren't discarded — they're **refilled
by bootstrap sampling** from the real column after the modelable columns are
synthesized (keeps fitting fast on wide tables). Bootstrapping means the *values*
are genuinely real, just shuffled across rows, so a name or email column refilled
this way would put real strings into the "synthetic" output. `synth_eval.pii`
exists specifically to catch that.

### Detection (`detect_pii`)

Two passes, name first:

1. **Column-name tokens**, checked in order, first hit wins: `EMAIL`/`E_MAIL` →
   email; `PHONE`/`TELEPHONE`/`_TEL`/`TEL_`/`FAX`/`MOBILE`/`CELL` → phone;
   `POSTAL`/`ZIP` → postal; `ADDR_LINE`/`ADDRESS_LINE`/`STREET`/`_ADDR`/`ADDR_` →
   street; `NAME`/`_NM`/`NM_`/`SURNAME`/`GIVEN`/`_USER`/`USER_` → name. Deliberately
   **no bare FIRST/LAST/MIDDLE tokens** — `LAST_VERIFIED_TRANSIT` isn't a name;
   real name columns virtually always carry `NAME` or `NM` somewhere. A suffix
   match against `_CD`/`_CODE`/`_ID`/`_IND`/`_DT`/`_DATE`/`_TP`/`_TYPE`/`_CT`/
   `_QTY`/`_AMT`/`_PCT` overrides everything — `PREFIX_NAME_TP_CD` is a type code,
   not a name, whatever "NAME" appears in it.

2. **Value-shape regexes**, only for object-dtype columns *outside* the modelable
   set that the name pass didn't already catch, so a low-cardinality code column
   can never be misread as a postal code just because a handful of its values
   happen to look like one. Checked against a sample of up to 60 non-null values,
   ≥60% must match: email, postal code (Canadian `A1A 1A1` or US 5/9-digit ZIP),
   street address, then phone last (loose digit pattern, requires a median of ≥7
   digits per value so it doesn't fire on short codes).

### Policy per column, chosen in the UI (default: `fake`)

| Policy | Effect |
|---|---|
| **`fake`** (default) | Replace every value with a **Faker-generated** value that never existed in the source (`en_CA` locale), preserving the column's original missing rate. `name` further specializes by column name: `FIRST`/`GIVEN`/`MIDDLE` → first names only, `LAST`/`SURNAME` → last names only, else a full name. Deterministic for a given (kind, row count, seed, column name), so reruns with the same seed reproduce the same fake values (see the `random_state` wiring in `synth_eval.suite`). |
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
