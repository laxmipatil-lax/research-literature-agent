# Autonomous Research Literature Review Agent

2-day scoped MVP. This is deliberately NOT the full iterative-refinement
concept discussed earlier — it's a one-shot-retry version, cut down to fit
the timeline. Say that explicitly in your report; it's better than an
evaluator finding the gap themselves.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # or paste it into the Streamlit sidebar
```
Get a free key at https://aistudio.google.com/apikey

**Note on free-tier limits:** Gemini's free tier caps requests per minute
*and* per day (the daily cap for some models has been as low as 20
requests). Since a single question can use several LLM calls, running
`eval_set.py` across many questions can hit this — `agent.py` auto-retries
on rate-limit (429) and server-overload (503) errors, but a daily quota
exhaustion won't resolve until it resets. Budget your testing accordingly
before a live viva demo.

Test arXiv connectivity first, separately from the LLM parts:
```bash
python arxiv_client.py
```
This was NOT tested against live arXiv from the environment this was written
in (sandboxed, no arxiv.org access). Run it yourself before you build on top
of it. If it breaks, the most likely cause is XML namespace parsing in
`_parse_atom_feed` — arXiv's Atom feed format is stable but worth checking
against `http://export.arxiv.org/api/query?search_query=all:test` directly
in a browser if something looks wrong.

Run the demo:
```bash
streamlit run app.py
```

Run the eval harness (do this before your report is due, not after):
```bash
python eval_set.py
```

## Architecture

```
question
   │
   ▼
PLAN  (LLM decomposes into 2-3 arXiv search queries)
   │
   ▼
RETRIEVE  (arXiv API, dedup by URL)
   │
   ▼
SYNTHESIZE  (LLM drafts answer citing papers)
   │
   ▼
EVALUATE  (LLM judges: is this answer actually sufficient?)
   │
   ├─ sufficient ──────────────► done
   │
   └─ insufficient (max 1x)
        │
        ▼
     REFINE  (LLM writes genuinely different queries targeting the gap)
        │
        └──► back to RETRIEVE
```

The **EVALUATE** step is the part that makes this "agentic" rather than a
static RAG pipeline — the agent inspects its own output and decides whether
to act again, rather than following a fixed script. Point to this
specifically in your viva; it's the one architectural decision that
separates this from a plain retrieve-then-generate system.

## What's honestly weak about this MVP — know these before the viva

1. **Refinement is capped at 1 iteration**, not open-ended. This was a
   deliberate scope cut for time, not a design ideal. If asked "why not
   more iterations," the honest answer is: uncapped loops risk infinite
   retries and runaway API cost, and a real production version would need
   a smarter stopping condition than a hard cap — that's a legitimate
   "future work" line in your report.

2. **No eval baseline was run before submission** unless you actually run
   `eval_set.py` yourself with real questions. Do this — a project with a
   demo but no numbers invites the question "how do you know it's better
   than a single-pass RAG system," and you want an answer, not a shrug.

3. **The self-evaluation step is itself an LLM call**, which means it can
   be wrong — it might say "sufficient" when it isn't, or vice versa.
   This is a real limitation of LLM-as-judge approaches generally, not a
   bug in this code. Naming this yourself in the viva is much better than
   having it pointed out to you.

4. **Single source (arXiv only).** Semantic Scholar or other sources were
   deliberately left out for the 2-day scope — rate limits would have
   eaten remaining build time. Note as future work, don't apologize for it.

5. **Citation accuracy is not independently verified.** The LLM cites
   papers by index ([1], [2]) based on what's in its context, but nothing
   currently checks that the cited claim actually appears in that paper's
   abstract vs. being LLM-inferred. Worth a manual spot-check on a few
   outputs before your demo so you're not caught off guard live.

## Suggested viva talking points, in order

1. Show the trace expander live — walk through plan → retrieve → evaluate →
   (if triggered) refine. This is your strongest visual proof of agentic
   behavior.
2. Deliberately ask a question likely to trigger refinement (narrow/obscure
   topic) so the panel sees the self-correction happen, not just the happy
   path.
3. Have `eval_results.json` open and ready — cite the refinement-rate stat
   ("X% of test questions needed a second pass") as your quantitative result.
4. Lead with limitation #1 and #3 above unprompted. Panels trust students
   who name their own weak points more than students who wait to be asked.
