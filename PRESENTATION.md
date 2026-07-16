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
vendor PoC, a model to train — that's weeks of access requests, masking, and
approvals, and every copy of real customer data is one more place it can leak
from. Synthetic data goes after both at once: data that behaves like the real
thing, with no real person's record in it.

One misconception up front: synthetic doesn't mean random — random strings
and numbers have nothing a model can learn from. We generate from models that
*learned* the real data's distributions, correlations, and table
relationships — and we ship a report that measures how well the output
behaves like the real thing, including whether it leaks. The report is the
trust.

This is Synth/Lab — a capability we want to embed in bmo.ai. Let me show it.
[Say each sentence ONCE. No rephrasing.]

---

## Loading data & schema (30 s)

One click loads three related tables — contact, person, person-name — tied
together by a customer ID. With real data, this is a drag-and-drop of your
CSVs.

Column types are auto-detected. One fix to show: this transit-number column
looks numeric but it's a branch code — a category — so I override it. Red
means my override. [Don't narrate distinct/missing counts — the screen shows
them.]

---

## The data model (45 s)

The tool already detected how the tables connect — a relationship is just a
drag from column to column, one-to-many or one-to-one. [Keep the
auto-detected ones; show ONE drag at most. Do not delete and rebuild.]

The trickier case: all three tables share the customer ID, but it isn't
unique in *any* of them. One click builds the entity hub — a table of unique
IDs — and connects all three to it. That's what keeps keys consistent across
tables. [Click while talking; don't read the button path out loud.]

Validate sanity-checks the whole model in a second, before any compute is
spent.

---

## The synthesizers (30 s), then run

You can pick several generators and let the report rank them. Today, two:
GaussianCopula — statistical, learns each column's distribution and how
columns move together, fast — versus HMA, the Hierarchical Modeling
Algorithm — the multi-table one, preserves cross-table links by
construction. There are two neural options too — CTGAN and TVAE — slower,
for complex data. [That's the whole tour — expansions live in Q&A.]

[Hit Synthesize.] Live progress, a real log, a Cancel that works. The
multi-table model needs a few minutes, so I ran this exact configuration
before we started — switching to that tab.

---

## The report (4.5 min — framing below, numbers off the screen)

### Leaderboard (30 s)

One card per generator, ranked. Three scores, zero to one: fidelity — looks
like the real data; utility — a model trained on it performs like one
trained on real; privacy — an attacker can't get real people back out.
Overall is the average.

[Say who won and read the numbers — one pass, no re-explaining the
dimensions.]

### Fidelity (1.5 min)

Fidelity is column statistics plus referential integrity — and the card
prints the exact arithmetic, so nothing here is a black box.

Column shapes: how close each column's distribution is to the real one — one
cell per column, green good, red bad. Column pair trends: the same for
*pairs* — are the correlations kept? That's where generators usually fail:
columns look right one at a time and the relationships fall apart.

Referential integrity is the new part — the previous tool had none. One row
per parent-child relationship, three things. Foreign-key coverage: every
child row must point at a parent that exists — under one hundred percent
means broken keys. Parent coverage: in the real data only about a third of
customers have a row in each child table, and the synthetic value should
*match* that — a generator that gives everyone a child row looks perfect and
is wrong. And the score, cardinality shape similarity: does each parent have
the right *number* of children, distribution-wise?

[Walk the table: who wins and why, off the screen.]

### Utility (45 s)

One question: can you do machine learning on this data? We train the same
model twice — once on real, once on synthetic — and test both on the same
held-back real data. Each metric gets a synthetic-over-real ratio; the
headline is their average. Zero-point-nine means training on synthetic gets
you ninety percent of what real would.

One caveat, printed on the card: it's relative — if the real baseline is
weak, matching it is easy.

[Read the per-table rows and the winner off the screen.]

### Privacy (45 s)

Privacy is attacked, not assumed. A slice of real data is held back before
training, and then three attacks. Membership inference: can an attacker tell
which records were used for training? Ideal is a coin flip. A copy check: are
synthetic rows just copies of real ones? — judged against what real data
itself scores. And attribute inference: knowing a few fields about someone,
can you guess a sensitive one? — also judged against that real-data ceiling.

If the data leaks, this page says so *before* anyone ships it.

[Read the pass/warn/fail verdicts off the screen.]

---

## More customization (45 s, back in the live tab)

Each run is tunable. Constraints — business rules the output must satisfy by
construction, like effective-date before end-date. PII handling — names,
emails, phones are detected automatically and replaced with generated fakes
that never existed in the source; per column you can keep or drop instead.
Run parameters — output size, holdout share, which column the ML test
predicts. And the deliverable: one CSV per generator per table, downloadable
right here.

---

## Close (30 s)

Everything ran on our own machine — the data never left it, and the stack is
open source.

This is a proof of concept: the goal is to show what's possible and get ideas
flowing. What lands in bmo.ai is the broader team's call.

The ask: your feedback now, and a pilot candidate — one team, one dev/test
dataset — to try this on real needs.

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
