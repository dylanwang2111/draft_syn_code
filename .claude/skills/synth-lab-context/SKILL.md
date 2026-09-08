---
name: synth-lab-context
description: Mission and standing design principles for the Synth/Lab synthetic-data dashboard. Load before proposing new detection, classification, or generation logic (column-type rules, PII detection, entity-key/relationship discovery, refill/fill-column logic), or when reasoning about whether a fix generalizes across arbitrary uploaded schemas versus only the schemas already tested against.
---

# Synth/Lab: mission and standing design principles

## What this application is

A FastAPI + vanilla-JS dashboard where a user uploads ANY seed tabular data
(one or more CSVs, arbitrary schema, arbitrary business domain) and gets back:

1. Well-synthesized data (via HMA / GaussianCopula / CTGAN / TVAE / CopulaGAN)
2. A report scoring that synthetic data on fidelity, ML-efficacy (utility), and privacy

The schema is **never known in advance**. Every detection/classification/generation
rule in this codebase has to work on a table it has never seen, not just the demo
seed data (PERSON/CONTACT/PERSONNAME/OCCUPATION) or the one production MDM schema
it's currently being validated against. A rule that only works because it happens
to match one schema's column-naming convention is a bug waiting to surface on the
next upload, not a finished feature.

## Standing design principles

### 1. Generalize, don't hardcode to one schema

Naming-convention suffix rules (`_TP_CD`, `_NAME`, `_DESC`, `_USER`, ...), PII
detection, entity-key/relationship discovery, and column-type overrides all lean
on the fact that real-world warehouse schemas commonly follow these conventions.
That's a good *default*, but the moment a rule is only correct because "that's
how this one schema happens to name things," it needs a fallback that actually
measures the data instead of trusting column names to be honest.

Concretely: when a rule needs to decide something schema-specific (e.g. "does this
filled-in column relate to a synthesizer-modeled column, and which one"), prefer
*measuring the real data* over encoding a fixed list of column-name pairs.
Measurement generalizes to a table nobody has looked at yet; a hardcoded pair list
only ever covers tables someone happened to eyeball first.

### 2. Weigh scalability AND value before proposing a fix

Before committing to build something, size both dimensions explicitly: does it
hold up as the tool grows (more tables, more rows, more concurrent uploads,
schemas nobody has seen), and is the actual value (how much of the reported
problem it fixes, how many tables/columns it touches) worth the added complexity.
Say both out loud rather than defaulting to the most thorough possible fix.

### 3. Privacy-first column handling, not a shortcut

Free-text / near-unique columns (names, descriptions, ids, audit users) are
deliberately excluded from direct synthesizer modeling and instead filled in from
real records after the fact (`_refill` in `backend/dashboard_core.py`), then run
through PII detection so anything sensitive gets faked (`apply_pii_plan`). This is
not a shortcut around "real" synthesis, it's the safer design: a neural model
trained on a modest number of rows, asked to generate near-unique free text, tends
to memorize and regurgitate real training values rather than invent new ones, and
there'd be no reliable way to catch that after the fact. Refilling from a
known-real record we explicitly control (and can fake if PII) is the privacy-safe
path; letting the synthesizer "learn" a name/description column is not.

## Resolved design thread: conditional refill (shipped 2026-08-06)

Production screenshots (PERSON/CONTACT/PERSONNAME/X_CDOCCUPATIONTP tables) showed
that filled-in columns (`NAME`, `P_LAST_NAME`, ...) could end up mismatched against
the synthesizer-generated code they logically belong with (`OCCUPATION_TP_CD`,
`SOURCE_IDENT_TP_CD`), because `_refill` drew a real row independently of what the
synthesizer generated for that row's modeled columns. `X_CDOCCUPATIONTP.NAME` ×
`OCCUPATION_TP_CD` and `PERSONNAME.P_LAST_NAME` × `SOURCE_IDENT_TP_CD` both showed
up red in the Column Pair Trends report as a result. The prior `_refill` fix
(joint-row sampling) fixed correlation *between* filled-in columns (e.g. `NAME` ↔
`DESCRIPTION`), but not correlation between a filled-in column and a modeled one.

**Fixed, per principle 1 above (generalize, don't hardcode):** rather than
hardcoding `X_CDOCCUPATIONTP.NAME → OCCUPATION_TP_CD` as a special case, each
filled-in column's association with every modeled column in its table is now
*measured directly from the real data* (`synth_eval.group_diversity_reduction`:
how much grouping by a candidate column narrows the filled-in column's diversity,
0 = no association, approaching 1 = near-total). `synth_eval.best_refill_group_column`
picks whichever candidate clears a minimum-association threshold (default 0.5);
`backend.dashboard_core._refill` then samples that filled-in column from real rows
sharing the *already-synthesized* value of the matched modeled column, instead of
from the whole table, with a `min_group_size` (default 10) privacy floor — a code
shared by too few real rows falls back to the unconditional whole-table sample, and
so does a code the real data never saw at all. Fill columns with no modeled column
clearing the threshold fall back the same way. Verified against the real
`OCCUPATION.csv` seed data: `OCCUPATION_TP_CD` → `OCCUPATION_NAME` association
measured at 0.952 (near its ceiling for 21 distinct codes), and once the group-size
floor was set at/below the real per-code row count, name/code mismatch dropped from
93% (unconditional) to 1% (conditional) in direct simulation.

Full writeup: `docs/DATA_HANDLING.md` → "Refilling skipped columns". Regression
coverage: `synth_eval/regression_suite.py` (`group_diversity_reduction`,
`best_refill_group_column`, and `_refill`'s conditioning/privacy-floor/
unseen-code-fallback behavior — 8 checks, part of the 30/30 suite).

**Intentional v1 scope boundary, still true:** a filled-in column conditions on at
most one modeled column (the single strongest match), not a joint match on several
at once — joint conditioning would shrink the matching group much faster, worse for
both data availability and privacy, for a case not yet shown to matter in practice.
