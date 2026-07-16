# Synth/Lab — live-demo script (spoken)

~10 minutes + Q&A. Pure script — everything below is spoken, in order, while
driving the app. [Square brackets] = quiet notes to self, not spoken.
The report beats are **framing only** — who wins and what each number says,
you read off the screen; that time is budgeted in.

**Setup beforehand:** Tab B has a finished HMA + GaussianCopula run sitting on
the Report; Tab A is fresh for the live part.

---

## Intro (45 s)

Why we built this: when a team needs realistic data — a dev environment, a
vendor PoC, a model to train — that's weeks of access requests and approvals,
and every copy of real customer data is one more place it can leak from.
Synthetic data attacks both: data that behaves like the real thing, with no
real person's record in it.

And it's not random data — random values have nothing a model can learn
from. Our models *learn* the real data's distributions, correlations, and
table relationships, then generate brand-new rows — and a report measures
how well the output behaves like the real thing, including whether it leaks.
Generating is easy; the report is the trust.

This is Synth/Lab — a capability we want to embed in bmo.ai. Let me show it.
[Say each sentence ONCE. No rephrasing.]

---

## Loading data & schema (30 s)

One click loads three related tables — contact, person, person-name — tied
by a customer ID. With real data, drag and drop your CSVs.

Column types are auto-detected. One fix to show: this transit-number column
looks numeric but it's a branch code — a category — so I override it. Red
means my override. [Don't narrate distinct/missing counts — the screen shows
them.]

---

## The data model (45 s)

The tool already detected how the tables connect — a relationship is one
drag, column to column, one-to-many or one-to-one. [Keep the auto-detected
ones; show ONE drag at most. Do not delete and rebuild.]

The trickier case: all three tables share the customer ID, but it's unique
in *none* of them. One click builds the entity hub — a table of unique IDs —
and connects all three to it; that keeps keys consistent across tables.
[Click while talking; don't read the button path out loud.]

Validate sanity-checks the whole model in a second.

---

## The synthesizers (30 s), then run

You can pick several generators and let the report rank them. Today, two:
GaussianCopula — statistical, fast, learns each column's distribution and
how columns move together — versus HMA, Hierarchical Modeling Algorithm —
multi-table, preserves cross-table links by construction. Two neural options
exist too — CTGAN and TVAE — slower, for complex data. [That's the whole
tour — expansions live in Q&A.]

[Hit Synthesize.] Live progress, a real log, a Cancel that works. The
multi-table model needs minutes, so here's the run I did earlier.

---

## The report (4.5 min — framing below, numbers off the screen)

### Leaderboard (30 s)

One card per generator, ranked. Three scores, zero to one: fidelity — looks
like the real data; utility — a model trained on it performs like one
trained on real; privacy — an attacker can't get real people back out.
Overall is the average.

[Say who won and read the numbers — one pass, no re-explaining the
dimensions.]

### Fidelity (2 min, four screens — one framing line each, numbers off the screen)

**[Fidelity tab — score cards + per-table breakdown]**
Fidelity is column statistics plus referential integrity — I'll define each
on its own tab. Each card prints the exact arithmetic — nothing is a black
box. Below, one row per synthesizer per table. [Read the overall column: who
wins which table.]

**[Column Shapes tab — one heatmap per table]**
Column shapes: one column at a time, does its distribution match real — same
spread of ages, same mix of codes? One cell per column: green matches, red
doesn't — you see exactly which columns each generator got right. [Best and
worst cell, name the column — hover shows the score. One example, move on.]

**[Column Pair Trends tab — one heatmap per synthesizer]**
Column pair trends: do columns still move *together* — older customers still
married more often — for every pair of columns? This is where generators
usually fail: every column right on its own, the relationships between them
fall apart. Blank cells: the real data had nothing measurable there. [One
kept pair, one lost pair. Don't read the grid.]

**[Referential Integrity tab — coverage table + cardinality table]**
Referential integrity: the same question *across* tables — does every child
row point at a customer that exists, and does each customer have a realistic
number of rows in each table? This is new — the previous tool had nothing
here. FK coverage under one hundred percent is broken keys. Parent coverage:
the gray *real* row is the target, and the badge shows how far each
synthesizer landed from it — one hundred percent when real isn't, is wrong.
And the score, cardinality shape: the right *number* of child rows per
customer, distribution-wise. [Winner off the cards, then Utility.]

### Utility (45 s)

One question: can you train a model on this data? The same model is trained
twice — on real, on synthetic — and both are tested on the same held-back
real data. Each metric becomes a synthetic-over-real ratio; the headline is
their average — point-nine means synthetic training gets you ninety percent
of real.

One caveat, printed on the card: it's relative — a weak real baseline is
easy to match.

[Read the per-table rows and the winner off the screen.]

### Privacy (45 s)

Privacy is attacked, not assumed. A slice of real data is held back before
training, then three attacks. Membership inference: can an attacker tell who
was in the training data? Ideal is a coin flip. A copy check: are synthetic
rows copies of real ones? And attribute inference: knowing a few fields, can
you guess a sensitive one? The last two are judged against what real data
itself scores — a fair ceiling, not an absolute bar.

If the data leaks, this page says so *before* anyone ships it.

[Read the pass/warn/fail verdicts off the screen.]

---

## More customization (45 s, back in the live tab)

Each run is tunable. Constraints — business rules the output must satisfy,
like effective-date before end-date. PII — names, emails, phones are
auto-detected and replaced with fakes that never existed in the source; or
keep or drop, per column. Run parameters — output size, holdout share, the
ML target. And the deliverable: one CSV per generator per table,
downloadable right here.

---

## Close (30 s)

This is a PoC: the goal is to show what's possible and get ideas
flowing. What lands in bmo.ai is the broader team's call.

The ask: your feedback now, next step we'll test it out with real production data.

---

## If something goes wrong mid-demo

- **Run looks stuck around 28 %** — it isn't; that's the multi-table model's
  silent learning phase. Say: "this is the slow model doing the actual
  multi-table learning" and switch to Tab B.
- **Anything else breaks in Tab A** — Tab B has the finished run; the whole
  report section lives there anyway.
- **A metric question you don't remember** — hover the ⓘ next to it and read
  it out; the explanations are written to be spoken.
- **No network** — charts render as static images automatically; the demo
  works fully offline.

## Q&A back-pocket

- **"Is synthetic data automatically non-sensitive?"** No — that's why the
  privacy report exists. The *report* is the gate, not the word "synthetic".
- **"What's the difference between the generators?"** Statistical
  (GaussianCopula: fast, learns distributions + correlations), hierarchical
  (HMA — Hierarchical Modeling Algorithm: multi-table, keeps cross-table
  links), neural (CTGAN — Conditional Tabular GAN; TVAE — Tabular Variational
  AutoEncoder: slower, better on complex patterns). Different data favours
  different ones — that's why we run several and rank them.
- **"What about IDs, dates, names?"** Never modelled. Names/emails/phones are
  replaced with generated fake values by default — they never existed in the
  source. Dates are resampled from the real timeline rather than modelled.
- **"What is statistic similarity / [deep metric question]?"** Shape compares
  the whole child-per-parent distribution; statistic compares its mean. These
  come from the sdmetrics framework — take the details offline rather than
  derail.
- **"How long does a run take?"** Statistical models: seconds. Multi-table on
  a few thousand rows: minutes. Neural models are the slow option and are
  labelled as such.
- **"What leaves the machine?"** Nothing. Data lives in memory per session;
  the profiler's recommendation uses structure only.
- **"How does this get into bmo.ai?"** It's already a REST API plus one
  embeddable web module — a new tab and an auth pass-through. And that's a
  decision for the broader team after today.
