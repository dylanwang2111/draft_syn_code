# Synth/Lab — live-demo script (spoken)

10 minutes + Q&A. Everything below is spoken; [square brackets] = notes to
self, not spoken.

**Read this first — the 18-minute rehearsal was 2× this script, not a longer
script. Three rules fix it:**
1. **Say each sentence ONCE.** If it comes out clumsy, keep going. Never
   restate an idea in different words — that alone doubled the last run.
2. **Never explain a table cell by cell.** One sentence per screen, then the
   winner. The audience can read.
3. **Nothing gets explained twice in two places.** If it's defined on its own
   tab, don't preview it on the tab before.

**Checkpoints — if you're past these, skip ahead:**
`2:00` leaving the data model · `3:00` on the report · `7:30` leaving privacy
· `8:30` customization [the thing you forgot last run — it's 30 s, don't drop it]

**Setup beforehand:** relationships + hub already built, two synthesizers
already ticked, Tab B has the finished run on the Report.

---

## Intro (30 s)

Why we built this: developers and analysts constantly need real data — to
train a model when there isn't enough real data, to stress-test an app, to
stand up a dev environment. Getting the real thing is weeks of access requests
and approvals, and every copy of it is one more place it can leak from.
Synthetic data solves both.

Not random data — our synthesizers learn the real data's distributions,
correlations and table relationships, then generate brand-new rows with no
real person's record in them. Generating is easy, so we also ship a report
that proves three things: fidelity — it behaves like real data; utility — a
model trained on it works; privacy — nothing links back to a real person.
That report is what lets someone build on it.

This is Synth/Lab — we want to embed it in bmo.ai. Let me show it.

---

## Load + schema (30 s)

One click loads three related tables — contact, person, person-name — tied by
a customer ID. With real data, you drag and drop your CSVs.

Types are auto-detected. One fix: this transit-number column looks numeric,
but it's a branch code — a category — so I override it. Red means my
override. [Nothing about SCD, distinct, or missing. Move.]

---

## The data model (40 s)

The tool detected how the tables connect — a relationship is one drag, column
to column, and you pick one-to-many or one-to-one. [Point at an existing
arrow. DO NOT drag, DO NOT open the dialog, DO NOT save — you did the full
flow last run and it cost a minute. The sentence covers it.]

The trickier case: all three share the customer ID, but it's unique in none
of them. One click builds the entity hub — a table of unique IDs — and links
all three to it. That's what keeps keys consistent across tables, and it's
what referential integrity in the report measures. [Click hub, keep talking.]

Validate sanity-checks the model in a second.

---

## Synthesizers (25 s), then run

Pick several generators, let the report rank them. Today two: GaussianCopula
— statistical, fast — versus HMA, Hierarchical Modeling Algorithm —
multi-table, preserves cross-table links by construction. CTGAN and TVAE are
the neural options, slower, for complex data. [Two only. Do not add CTGAN to
the run. Do not explain how a GAN works — that's Q&A.]

[Synthesize.] Live progress, a real log, a working Cancel. HMA takes minutes,
so here's the run from earlier.

---

## The report (4 min)

### Leaderboard (30 s)

One card per generator, ranked. Three scores, zero to one — and the tension
between them is the whole point. Fidelity: does it look like the real data.
Utility: does a model trained on it behave like one trained on real. Privacy:
can an attacker get real people back out?

Any one of them alone is easy to game. That random data I mentioned? Perfect
privacy score — there's nothing in it to leak. It just fails the other two
completely. A generator has to win all three at once, and that's what this
page ranks. Overall is the average. [Name the winner. Do not re-explain these
three anywhere else in the demo.]

### Fidelity (1.5 min — three screens, one sentence each)

**[Fidelity tab]** Fidelity is column statistics plus referential integrity.
The card prints the exact arithmetic — nothing is a black box. [Winner. Move.]

**[Column Shapes]** Column shapes: one column at a time, does its
distribution match real — same spread of ages, same mix of codes? Green
matches, red doesn't. [Point at one red cell if there is one. Otherwise:
"all green — every column's distribution matches." Move.]

**[Column Pair Trends]** Pair trends: do columns still move *together* —
older customers still married more often? This is where generators usually
fail: every column right on its own, the relationships between them fall
apart. Green kept the trend, red lost it, blank means the real data had
nothing measurable there. [ONE red cell, name the pair, winner, move. Don't
tour the grid and don't report the winning margin — nobody needs 0.01.]

**[Referential Integrity]** The across-tables question, and it's new — the
previous tool had nothing here. Three numbers, only one is the score.

**FK coverage** — the gate: every child row points at a customer that exists.
It says n/a because our hub is built from each synthesizer's own keys, so it
can't fail — no credit for that.

**Parent coverage** — the diagnostic: the gray real row is the target.
Matching real is the goal, not hitting one hundred percent.

**Cardinality shape** — the score, and the only one that feeds fidelity: does
each customer have the right *number* of rows, across the whole distribution?

[HMA wins by construction — the only one fitted with the relationships. Each
line ONCE. Winner, move.]

[End of Fidelity — the one place you're allowed to acknowledge depth:
"There's more behind every one of these — happy to go deeper in Q&A." Say it
here only, never per tab.]

### Utility (45 s)

Can you train a model on this data? The same model is trained twice — on
real, on synthetic — and both are tested on the same held-back real data.
Each metric becomes a synthetic-over-real ratio, and the headline is their
average: point-nine means synthetic training gets you ninety percent of real.
One caveat printed on the card: it's relative — a weak real baseline is easy
to match. [Read ONE ratio and the winner. Do not walk the columns.]

### Privacy (45 s)

Privacy is attacked, not assumed. A slice of real data is held back before
training, then three attacks — one line each, pointing as you name them:

**MIA protection** — membership inference: can an attacker tell who was in
the training data? Ideal is a coin flip.
**NewRowSynthesis** — the copy check: are synthetic rows copies of real ones?
**CategoricalCAP** — attribute inference: knowing a few fields about someone,
can you guess a sensitive one?

The last two are judged against what real data itself scores — a fair
ceiling, not an absolute bar. If the data leaks, this page says so before
anyone ships it. [Name, one line, point, next. This is where the last run
tripled. Winner, then move.]

---

## Customization (30 s — one breath, no demo)

Each run is tunable: constraints like effective-date before end-date; PII —
names, emails, phones auto-detected and replaced with fakes that never
existed in the source; run parameters — output size, holdout, the ML target,
the privacy attack column. And the deliverable: one CSV per generator per
table, right here. [Gesture at the panel. Do not open the sub-panels — offer
them in Q&A if asked.]

---

## Close (25 s)

This is a PoC — the goal is to show what's possible and get ideas flowing.
What lands in bmo.ai is the broader team's call.

The next step your feedback now. Next step, we test it on real production data
we're requesting access to — and potentially unstructured data after that.

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
- **"What are the actual use cases / who uses this?"** Developers and analysts.
  Analysts: train an ML model when there isn't enough real data — synthesize
  more, validate fidelity and utility, then train on it. Developers:
  stress-testing and test data without touching production. And a bigger one to
  keep in your back pocket — cloud migration: teams moving on-prem data to the
  cloud who want to test the move without exposing real customer data. HR and
  other teams are going through exactly that now.
- **"What's the difference between the generators?"** Statistical
  (GaussianCopula: fast, learns distributions + correlations), hierarchical
  (HMA — Hierarchical Modeling Algorithm: multi-table, keeps cross-table
  links), neural (CTGAN — Conditional Tabular GAN, two networks trained
  against each other until fake is indistinguishable; TVAE — Tabular
  Variational AutoEncoder: slower, better on complex patterns).
- **"These look like slowly-changing-dimension tables?"** They are — each
  customer has multiple versions with effective/end dates. The key stays
  constant but isn't unique in any table, which is exactly why the entity hub
  exists.
- **"What about IDs, dates, names?"** Never modelled. Names/emails/phones are
  replaced with generated fake values by default — they never existed in the
  source. Dates are resampled from the real timeline rather than modelled.
- **"What is statistic similarity / [deep metric question]?"** Shape compares
  the whole child-per-parent distribution; statistic compares its mean. These
  come from the sdmetrics framework — take the details offline rather than
  derail.
- **"Why is FK coverage n/a instead of 100%?"** Because the hub is derived
  from the keys each synthesizer emitted — every child key is in it by
  construction. It's a free 1.0 for everyone, so we report it as a gate, not
  a score. Link two real tables directly and it becomes a real pass/fail.
- **"Why isn't parent coverage part of the score?"** Parent coverage is
  "share of customers with at least one row" — a single point on the
  distribution cardinality shape already measures in full. Scoring both would
  count the same thing twice. It stays as a diagnostic because it should
  *match* real, not be maximised.
- **"How long does a run take?"** Statistical models: seconds. Multi-table on
  a few thousand rows: minutes. Neural models are the slow option and are
  labelled as such.
- **"What leaves the machine?"** Nothing. Data lives in memory per session;
  the profiler's recommendation uses structure only.
- **"How does this get into bmo.ai?"** It's already a REST API plus one
  embeddable web module — a new tab and an auth pass-through. And that's a
  decision for the broader team after today.
