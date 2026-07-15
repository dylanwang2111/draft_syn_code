# Synth/Lab — live-demo script (spoken)

~10 minutes + Q&A. Pure script — everything below is spoken, in order, while
driving the app. [Square brackets] = quiet notes to self, not spoken.

**Setup beforehand:** Tab B has a finished HMA + GaussianCopula run sitting on
the Report; Tab A is fresh for the live part.

---

## Intro (1–1.5 min)

When a team needs realistic data — a dev environment, a vendor PoC, a model to
train — that's usually weeks of work: access requests, masking, approvals, getting a pad.
"Just use the real data" is the cheapest sentence in analytics and the most
expensive one in practice in terms of approval, and privacy.

First, what synthetic data is *not*: it's not random data. Anyone can write a
Python function that fills a string column with random characters and a
percentage column with random numbers from zero to a hundred. That has no utility
 — no patterns, no relationships, nothing a model
can learn from.

What we do instead has two halves. One: we use statistical methods that
*learn* the real data's patterns — the distributions, the correlations, how
tables relate to each other — and generate brand-new rows from what they
learned. 
and just as important: we provide an evaluation framework that
measures how well the synthetic data behaves like the real thing. Because
generating is easy — *trusting* the output is the hard part. The report is
what gives the end user confidence the data is usable for their needs: model
training and testing, dev and test environments, sharing — with privacy
checked.

This is Synth/Lab — a capability we want to embed in bmo.ai. Let me just show
it.

---

## Loading data & schema (1 min — move fast)

One click and we've loaded three related tables — contact, person, and
person-name — tied together by a customer ID. With real data this is just a
drag-and-drop of your CSVs.

Column types are detected automatically — the date columns here came in as
real timestamps and were recognized as datetime on their own. If it gets one
wrong, you fix it with a dropdown — red means it's your override. For example,
this transit-number column looks numeric, but it's really a branch code — a
category — so I'll override it.

---

## The data model (1.5 min)

When you upload multiple tables, chances are they're connected — and here's
where you tell the tool how. You draw the connections column to column: drag, pick one-to-many or one-to-one, save.

There's also a trickier case these three tables happen to have: they all share
the same customer ID, but it isn't unique in *any* of them — one customer has
many rows in each table. One click generates a table of unique IDs — we call
it the entity hub — and connects all three tables to it. That's what lets the
multi-table synthesizer keep the keys consistent across tables.

Before spending any compute, Validate sanity-checks the whole model in a
second.

---

## The synthesizers (1 min), then run

You can pick several generators at once — because different data favours
different models, and the report will tell us which one won. A quick tour of
what's on the menu:

GaussianCopula — named after the copula, a statistical tool that links
individual column distributions into one joint model. It learns each column's
distribution and how columns move together, then samples new rows.

HMA — Hierarchical Modeling Algorithm — the multi-table one. It also learns
how parent and child tables relate, so it's the only one that preserves the
links between tables by construction.

CTGAN — Conditional Tabular Generative Adversarial Network — two neural
networks trained against each other: one generates fake rows, the other tries
to tell fake from real, until it can't. Better on complex, imbalanced data,
but much slower.

TVAE — Tabular Variational AutoEncoder — another neural approach: it
compresses rows into a compact representation and generates new rows from it.
Also slower.

Today I'll run HMA and GaussianCopula — the structural one versus the fast
one. [Hit Synthesize.] You get live progress, a real log, and a Cancel that
actually works. The multi-table model needs a few minutes — so I ran this exact configuration before we started.
Let me switch to that tab.

---

## The report (4 min — the payoff)

### The leaderboard (45 s)

Top-level assessment first: one card per generator, ranked. Three dimensions,
each scored zero to one.

Fidelity — how well the synthetic data *looks like* the real data. Utility —
how *usable* it is: does a model trained on it give similar results to one
trained on real data? And privacy — can an attacker get anything about real
people back out?

Overall is just the average of the three. That's the quick read — now let's
drill into what the scores actually mean.

### Fidelity (1.5 min — quick on shapes, slow on referential integrity)

Fidelity is column statistics plus referential integrity — and the card
prints the exact arithmetic with the real numbers in it, so nothing here is a
black box.

Column shapes first: this compares the distribution
of each column between synthetic and real — how close does each column look?
One cell per column, for every table: green good, red bad. So per table, you
can see exactly which columns each generator got right.

Column pair trends is the same idea for *pairs* of columns — is the
correlation between two columns kept in the synthetic data? Dark or blank
cells just mean the real data had no measurable relationship there to compare.
This is where generators usually fail: columns look right one at a time, and
the relationships between them fall apart.

Now referential integrity — and I want to spend a moment here, because the
previous tool didn't have this at all; it's new for our group. This table
shows one row per parent-child relationship, and three things about each.

Foreign-key coverage is table stakes: every child row must point at a parent
that exists. Anything under one hundred percent means broken keys.

The interesting one is parent coverage. In the real data, only about a third
of customers have a row in each child table. It's *not supposed to be* one
hundred percent — the synthetic value should *match the real value*, and the
badge shows exactly that: "equals real", or how many points off it is. A
generator that gives every customer a child row looks perfect on coverage and
is actually wrong.

And the score for this section is cardinality shape similarity — one number,
closer to one the better, answering: does each parent have the right *number*
of children, distribution-wise? [If asked about "statistic similarity": it
compares the mean instead of the whole distribution; both come from the
sdmetrics framework — take details offline.]

### Utility (45 s)

Utility answers one question: is this data usable for machine learning? We
train the same model twice — once on real data, once on the synthetic — and
test both on the same held-back real data.

The table shows the results side by side, per table: F1, accuracy, precision,
recall — the real-trained score, the synthetic-trained score, and a "divided
by real" column for each. For each metric we take that ratio,
synthetic-trained over real-trained, and the headline number is the average of
those ratios. Around zero-point-nine means: training on synthetic gets you
ninety-plus percent of what training on real would.

One honest caveat, printed right on the card: the ratio is relative to the
real baseline — if the real model is weak, matching it is easy.

### Privacy (45 s)

Privacy is attacked, not assumed. Before training, a slice of real data is
held back — the generators never see it — and then we attack the output three
ways.

A membership-inference attack: can an attacker tell which real records were
used for training? Ideal is a coin flip. A copy check: are synthetic rows just
copies of real rows? — judged against what real data itself scores, so tables
with few distinct values aren't unfairly failed. And an attribute-inference
attack: knowing a few fields about someone, can you guess a sensitive one?

The point is: if the data leaks, this page says so *before* anyone ships it.

---

## More customization (1 min, back in the live tab)

Beyond the defaults, each run is tunable.

Constraints are business rules the output must satisfy by construction — like
effective-date must be before end-date.

PII handling: name, email, and phone-like columns are detected automatically
and, by default, replaced with generated fake values that never existed in the
source. Per column, you can also choose to keep or drop them instead.

Run parameters: output size — half the rows, double the rows — how much real
data to hold back for the evaluation, and which column the machine-learning
test predicts.

And the deliverable: one CSV per generator per table, downloadable right here.

---

## Close (30 s)

Everything you saw ran on our own machine — the data never left it, and the
stack underneath is open source.

To set expectations: this is a proof of concept. The goal today is to show
what's possible and get ideas flowing. What lands in bmo.ai, and in what form,
is the broader team's call — keep some of it, drop some, extend some.

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
