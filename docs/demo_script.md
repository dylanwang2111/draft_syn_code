# Synth/Lab Demo Script (~5 minutes)

## Context

- **Wednesday sessions (Cody @ 10:30, Ryan Guffy @ 12:30) are practice runs.** Low stakes, show-and-tell. Use them to rehearse pacing, not to get every word right.
- **The Kristen presentation is the one that matters.** Same content, same script.
- **Framing:** Customer 360 use case, presented as a standalone product initiative (DII product epic). Don't tie it to XArch, that engagement is treated as done after the Kristen presentation.
- **Audience is non-technical.** Avoid jargon:
  - Say "real data" or "production data," not "PII data."
  - Say "distribution" or "patterns," not "column shape."
  - Keep the UI on the high-level view; only click "Show details" if asked.

## Time budget

| Segment | Target time |
|---|---|
| Intro / use case | ~15 sec |
| 1. Ingest real data | ~45 sec |
| 2. Synthesize | ~30 sec |
| 3. Assess quality (fidelity / utility / privacy) | ~2.5 min |
| Buffer / questions | remainder |

Total: ~5 minutes.

---

## Script

### Intro (~15 sec)

> "This is a tool we built for generating synthetic data for our Customer 360 use case. I want to walk you through the journey: how we go from real production data to synthetic data, and how we confirm that synthetic data is actually good quality before anyone uses it."

### Step 1: Ingest real data (~45 sec)

- Point to / upload the real production data.
- "The tool automatically detects the schema and the relationships between tables. You can customize settings like data types if you need to, but most of the time the defaults are right."
- **Relationship customization (~20-30 sec, keep it simple):**
  > "Sometimes the relationship between tables is a bit more complex than a straight line. One way we handle that is by letting you create a custom relationship, for example a central table that connects a few others."
  - Do a quick drag-and-drop demo connecting a column to a column.
  - Don't explain *why* it's a hub table or what problem it solves technically.
  - Close with: "Now the tool understands there's a relationship between these tables."

### Step 2: Synthesize (~30 sec)

- Click generate/run.
- "This creates a synthetic version of the data. It's built to preserve the patterns and structure of the real data without containing any actual customer records."

### Step 3: Assess quality (~2.5 min)

**Explain the three pillars (~45-60 sec total, ~15-20 sec each):**

- **Fidelity:** does the synthetic data statistically look like the real data (same distributions, same patterns)?
- **Utility:** is the synthetic data actually useful, e.g. can you train a machine learning model on it and get similar results as if you'd used the real data?
- **Privacy:** does the synthetic data avoid leaking or resembling any real individual's actual record?

**Score framing (say this once, applies to all three):**
> "Each of these is scored from 0 to 1. Closer to 0 is bad, closer to 1 is best. So a score of 0.86 means the synthetic data is about 86% of the way to the ideal outcome."

**One example per pillar (~20 sec each):**

- **Fidelity:** Show the heatmap, pick one numeric column. "This shows how closely the synthetic data's distribution, its average, median, spread, matches the real data for this field."
- **Privacy:** Show the New Row Synthesis metric. "This checks whether any synthetic record is an exact copy of a real one. It shouldn't be."
- **Utility:** Show the overall ML efficacy score only. "This represents how well a machine learning model trained on the synthetic data performs compared to one trained on the real data. I'll keep this high-level, there's a more detailed report underneath if you want to dig in."

**If asked for more detail on any metric:**
> "There's a 'Show details' view that gets into the technical calculation, but in the interest of time let's keep this high-level. Happy to set up a follow-up to go deeper if useful."

### Wrap-up (~15 sec)

> "So to recap: we go from real production data, to synthetic data, and we validate that synthetic data on fidelity, utility, and privacy before it's used. Happy to answer questions or go deeper on any piece."

---

## Reminders

- Wednesday demos are practice. If you run over, that's fine, Ash may cut you off on time; don't stress about hitting the mark exactly.
- Keep the UI on the high-level / summary view by default. Only expand "Show details" if someone asks.
- Don't use the terms "PII," "SCD," "column shape," or "hub table" unprompted, plain-language equivalents above.
- Don't mention XArch, frame this as a standalone initiative under DII.
