# Synth/Lab — live-demo presentation script

Demo-led: the app carries the talk. ~10 minutes + Q&A.
Each beat has **DO** (your clicks) and **SAY** (spoken lines). Square brackets = notes to self, not spoken.

---

## Before the audience arrives — setup checklist

1. Start the server from the project root (`uvicorn server:app`), open **two
   browser tabs** on the app. Tabs are isolated sessions — this is the safety net.
2. **Tab B (backup):** Load Sample → entity key `CONT_ID` over all three tables →
   HMA + GaussianCopula → Synthesize → let it finish. Leave it sitting on the Report.
3. **Tab A (live):** fresh and empty — this is the one you present in.
4. Zoom ~110 %, pick the theme that suits the room lighting, unfold the report
   nav, close dev tools.
5. [Bad network? Charts silently fall back from interactive to static images —
   nothing to do, don't mention it.]

---

## Intro (30 s) — before touching the app

> When a team here needs realistic data — a dev environment, a vendor PoC, a
> model to train — that's usually weeks: access requests, masking, approvals or even getting a pad.
> "Just use the real data" is the cheapest sentence in analytics and the most
> expensive one in practice.
>
> So instead of slides, let me just show you the alternative. This is Synth/Lab —
> a capability we want to embed in **bmo.ai**. The idea: upload your tables,
> a few clicks, and you leave with synthetic data that behaves like the real
> thing — **plus a report that proves it**.

---

## Beat 1 — Load data & schema (2 min)

**DO:** In Tab A, click **Load Sample**. You land on the Schema tab. Scroll the
CONTACT table slowly. Change one sdtype dropdown so they see the red override,
then change it back. Point at the advisor banner on the left; hover its ⓘ and
pause two seconds.

**SAY:**

> One click and three related tables are in — contact, person, person-name,
> 1,600 rows each, tied together by a customer ID. With real data this is a
> drag-and-drop of CSVs.
>
> The schema is detected automatically: every column gets a type, distinct
> counts, missing rates. If the detector gets one wrong, it's a dropdown — red
> means "you overrode it".
>
> Two details that matter in our environment. IDs, dates and free-text names are
> **excluded from statistical modelling** automatically — they're refilled
> structurally afterwards, so no model ever memorises a customer ID. And this
> advisor has already profiled the dataset and recommended a strategy — using
> **structure only**, so it's safe even on data nobody is allowed to eyeball.
> The full reasoning is one hover away. [hover the ⓘ]

---

## Beat 2 — Data Model canvas (2 min)

**DO:** Open the **Data Model** tab. Drag one column port onto a port in another
table → the relationship sidebar opens → point at the cardinality picker →
**Save**. Click the arrow to select it, press **Delete** to remove it. Then:
pick `CONT_ID` in the hub bar, tick the three tables, **Generate hub** — the hub
card parks itself left of its children. Click **Validate**.

**SAY:**

> Real banking data is never one table, so relationships are first-class. I drag
> from column to column — like Power BI — pick one-to-many or one-to-one, save.
> Don't like it? Select the arrow, hit Delete.
>
> But the common case is simpler: these tables share a customer ID. One click
> builds an **entity hub** over it — a derived parent table of distinct
> customers — and now a multi-table synthesizer can preserve **referential
> integrity**: every synthetic row still points at a valid synthetic customer.
> Single-table tools break exactly this.
>
> And before spending any compute, **Validate** sanity-checks the whole model —
> key uniqueness, coverage — in a second. [point at the green PASS rows]

---

## Beat 3 — Configure & run (1.5 min)

**DO:** In the left panel, open **Synthesizers** and click the **HMA** and
**GaussianCopula** chips. Open **Run parameters**: nudge the scale slider, point
at the holdout slider. Hit **▶ Synthesize**. Let the progress bar and console
run for ~15 seconds. Point at **■ Cancel**.

**SAY:**

> I'm picking two generators on purpose — a hierarchical one that learns the
> cross-table structure, and a fast copula model. This isn't "generate an
> answer"; it's **run a competition**.
>
> Scale is a slider — half the rows, double the rows. The holdout is the
> important one: a slice of real data the generators **never see**, kept aside
> to attack the output later.
>
> [run starts] Live progress, a real log, and a Cancel that actually stops
> everything. The hierarchical model needs a few minutes on this data — and
> since every browser tab is its own isolated session, I ran this exact
> configuration before we started. Let me switch to that tab rather than make
> you watch a progress bar.

**DO:** Switch to **Tab B**, already sitting on the finished Report.

---

## Beat 4 — The report (3 min — the payoff)

### 4a. Leaderboard (30 s)

**DO:** Show the podium. Point at the BEST tag and the three meters on a card.

**SAY:**

> One card per generator, ranked. Three dimensions: **fidelity** — does it look
> like the real data; **utility** — can you train a model on it; **privacy** —
> can an attacker get anything back out. The overall score is the mean of the
> three — and I can prove every digit on this screen.

### 4b. Fidelity (45 s)

**DO:** Click **Fidelity** in the right nav. Point at the formula line printed
on a score card. Then click **Column Pair Trends** and hover a heatmap cell.

**SAY:**

> Here's what I mean by "prove". The fidelity card doesn't just say 0.88 — it
> prints the arithmetic: two parts column statistics, one part referential
> integrity, with the actual numbers substituted in. Every number on this screen
> can be reconstructed by hand. In a regulated environment that's not a nicety —
> it's the difference between a demo and something risk will sign off on.
>
> Drill-down is one click. This heatmap is every **pair** of columns — whether
> the relationships *between* fields survived, not just each field on its own.
> That's where most generators quietly fail. [hover a red cell]

### 4c. Referential integrity (45 s)

**DO:** Click **Referential Integrity**. Point at a parent-coverage badge
(`= real` / `↑ pts`), then at the cardinality table.

**SAY:**

> The cross-table check. Forward coverage says no synthetic row points at a
> customer that doesn't exist — no orphans. The subtle one is this badge: in
> the real data only about a third of customers have a row in each child table.
> A naive generator gives **every** customer one — which looks perfect on a
> coverage metric and is actually wrong. We score **matching reality**, not
> maximising a number. That's the honesty built into this report.

### 4d. Utility & privacy (60 s)

**DO:** Click **Utility**; point at a `÷ real` column, then the roll-up table.
Click **Privacy**; point at the MIA line on a score card.

**SAY:**

> Utility is the acid test: train the same model once on real data, once on
> each generator's output, test both on the **same real holdout**. This column
> is the ratio, panel by panel — the headline is literally the mean of the
> numbers you can see. Around 0.9 means a model trained on the synthetic data
> performs ninety-plus percent as well as one trained on real.
>
> Privacy is attacked, not asserted. A **membership-inference attack** tries to
> tell training rows from unseen rows — ideal is a coin flip, 0.5, and the raw
> attacker score is shown right here. Plus a copy-detection check and an
> attribute-inference check. If the synthetic data leaks, this page says so
> **before** anyone ships it.

### 4e. Download (15 s)

**DO:** Open the **Synthetic Data** panel on the left; click one ↓ download.

**SAY:**

> And the deliverable itself: one CSV per generator per table, at whatever scale
> you asked for. That's the whole loop — upload to evidence-backed synthetic
> data in a handful of clicks.

---

## Close (30 s)

> Everything you just watched ran on this machine — the data never left it,
> sessions are isolated per user, and the stack underneath is open source, so
> there's no per-row vendor cost. Landing it in bmo.ai is a new tab and an auth
> pass-through — integration work, not invention.
>
> The ask: a pilot — one team, one dev/test dataset, inside bmo.ai. If the
> report says the data is faithful and private for their use case, we've turned
> a weeks-long data request into a five-minute self-serve, with the evidence
> attached.

---

## If something goes wrong mid-demo

- **Run looks stuck around 28 %** — it isn't; that's the hierarchical model's
  silent learning phase and the log prints "still working — elapsed…". Say:
  *"this is the slow model doing the actual multi-table learning"* and switch
  to Tab B.
- **Anything else breaks in Tab A** — Tab B has the finished run; the whole
  report section of the demo lives there anyway.
- **A metric question you don't remember** — hover the ⓘ next to it and read
  it out; the explanations are written to be spoken.
- **No network** — charts render as static images automatically; the demo
  works fully offline.

## Q&A back-pocket

- **"Is synthetic data automatically non-sensitive?"** No — that's why the
  privacy report exists. Policy should treat the *report* as the gate, not the
  word "synthetic".
- **"What about IDs, dates, free-text names?"** Never modelled — excluded by
  design and refilled structurally. Faithful timelines need real date columns
  (the profiler flags Excel-mangled ones).
- **"How long does a run take?"** Copula models: seconds to a minute.
  Hierarchical multi-table on a few thousand rows: minutes. GANs are the slow
  option and are labelled as such.
- **"Slowly-changing-dimension history tables?"** Detected and handled —
  version patterns are modelled and timelines repaired so effective/end dates
  don't overlap per entity.
- **"What leaves the machine?"** Nothing. Data lives in memory per session; the
  profiler's recommendation uses structure only, so it's safe even on data we
  can't visually inspect.
- **"How does this get into bmo.ai?"** It's already a REST API plus one
  embeddable web module — a new tab and an auth pass-through. Batch pipelines
  can skip the UI and call the API directly.
