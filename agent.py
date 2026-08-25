"""
agent.py
--------
The actual "agent" part of the project. This is what you defend in the viva.

Pipeline (NOT a plain RAG pipeline -- this is what makes it agentic):

  1. PLAN     -> LLM decomposes the user's question into 2-3 focused search queries
  2. RETRIEVE -> arXiv API is called for each sub-query
  3. SYNTHESIZE -> LLM drafts an answer citing the retrieved papers
  4. EVALUATE -> LLM judges whether its own answer is actually sufficient
                 (this self-check is the key agentic decision point)
  5. REFINE (capped at ONE retry) -> if insufficient, LLM writes a *different*
                 set of search queries based on what was missing, and the
                 pipeline runs once more. It does NOT just repeat the same query.

Every step is returned in the trace so you can show it live in a viva --
that visible reasoning trail is what convinces an evaluator this is agentic
behavior and not a chatbot with a search API bolted on.

Requires: pip install google-genai
Set GEMINI_API_KEY in your environment before running.
(Get a free key at https://aistudio.google.com/apikey)
"""

import os
import json
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from arxiv_client import search_arxiv

MODEL = "gemini-3.6-flash"  # per Google's API error message pointing to the current model
MAX_REFINEMENTS = 1  # hard cap -- do not raise this without also adding a cost/time budget


class ResearchAgent:
    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.trace = []  # every step gets logged here for demo/report purposes

    def _log(self, step: str, data):
        self.trace.append({"step": step, "data": data})

    def _llm_call(self, system: str, user: str, max_tokens: int = 1000, max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=MODEL,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                    ),
                )
                return resp.text or ""
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                if code in (429, 503) and attempt < max_retries - 1:
                    wait = 20 if code == 429 else 10  # rate limit vs server overload
                    self._log("retrying_after_error", {"code": code, "attempt": attempt + 1, "waiting_seconds": wait})
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Exceeded max retries for API errors.")

    # ---------- Step 1: Planning ----------
    def plan(self, question: str) -> list[str]:
        system = (
            "You break a research question into 2-3 focused arXiv search queries. "
            "Respond ONLY with a JSON array of strings, nothing else. No markdown fences."
        )
        raw = self._llm_call(system, f"Question: {question}")
        queries = self._safe_json_list(raw, fallback=[question])
        self._log("plan", {"question": question, "sub_queries": queries})
        return queries

    # ---------- Step 2: Retrieval ----------
    def retrieve(self, sub_queries: list[str], max_results_per_query: int = 4) -> list[dict]:
        all_papers = {}
        for q in sub_queries:
            try:
                results = search_arxiv(q, max_results=max_results_per_query)
            except Exception as e:
                results = []
                self._log("retrieve_error", {"query": q, "error": str(e)})
            for p in results:
                all_papers[p["url"]] = p  # dedupe by url
        papers = list(all_papers.values())
        self._log("retrieve", {"sub_queries": sub_queries, "papers_found": len(papers)})
        return papers

    # ---------- Step 3: Synthesis ----------
    def synthesize(self, question: str, papers: list[dict]) -> str:
        if not papers:
            return "No relevant papers were retrieved for this question."

        context = "\n\n".join(
            f"[{i+1}] {p['title']}\nAuthors: {', '.join(p['authors'][:4])}\n"
            f"Summary: {p['summary'][:500]}\nURL: {p['url']}"
            for i, p in enumerate(papers)
        )
        system = (
            "You are a research assistant. Answer the user's question using ONLY the "
            "provided papers. Cite papers inline using [1], [2], etc. matching the list. "
            "If the papers don't fully answer the question, say so explicitly."
        )
        user = f"Question: {question}\n\nPapers:\n{context}"
        answer = self._llm_call(system, user, max_tokens=1500)
        self._log("synthesize", {"answer": answer, "num_papers_used": len(papers)})
        return answer

    # ---------- Step 4: Self-evaluation (the key agentic decision) ----------
    def evaluate(self, question: str, answer: str) -> dict:
        system = (
            "Judge whether the ANSWER sufficiently addresses the QUESTION using real "
            "evidence from cited papers. Respond ONLY with JSON: "
            '{"sufficient": true/false, "reason": "short reason", '
            '"missing": "what specific angle/subtopic is missing, if any"}. No markdown fences.'
        )
        user = f"Question: {question}\n\nAnswer: {answer}"
        raw = self._llm_call(system, user, max_tokens=300)
        verdict = self._safe_json_obj(raw, fallback={"sufficient": True, "reason": "parse_fallback", "missing": ""})
        self._log("evaluate", verdict)
        return verdict

    # ---------- Step 5: Refinement (capped, and genuinely different queries) ----------
    def refine_queries(self, question: str, missing: str) -> list[str]:
        system = (
            "The previous search missed something. Given what was missing, write 2-3 NEW "
            "arXiv search queries that are meaningfully different from a naive repeat -- "
            "target the missing angle specifically. Respond ONLY with a JSON array of strings."
        )
        user = f"Original question: {question}\nWhat was missing: {missing}"
        raw = self._llm_call(system, user, max_tokens=200)
        queries = self._safe_json_list(raw, fallback=[question])
        self._log("refine", {"missing": missing, "new_queries": queries})
        return queries

    # ---------- Orchestration ----------
    def run(self, question: str) -> dict:
        self.trace = []
        sub_queries = self.plan(question)
        papers = self.retrieve(sub_queries)
        answer = self.synthesize(question, papers)
        verdict = self.evaluate(question, answer)

        refinements_used = 0
        while not verdict.get("sufficient") and refinements_used < MAX_REFINEMENTS:
            refinements_used += 1
            new_queries = self.refine_queries(question, verdict.get("missing", ""))
            new_papers = self.retrieve(new_queries)
            # merge, dedupe by url
            merged = {p["url"]: p for p in papers}
            merged.update({p["url"]: p for p in new_papers})
            papers = list(merged.values())
            answer = self.synthesize(question, papers)
            verdict = self.evaluate(question, answer)

        return {
            "question": question,
            "answer": answer,
            "final_verdict": verdict,
            "refinements_used": refinements_used,
            "papers": papers,
            "trace": self.trace,
        }

    # ---------- helpers ----------
    @staticmethod
    def _safe_json_list(raw: str, fallback: list) -> list:
        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, list) else fallback
        except Exception:
            return fallback

    @staticmethod
    def _safe_json_obj(raw: str, fallback: dict) -> dict:
        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else fallback
        except Exception:
            return fallback


if __name__ == "__main__":
    agent = ResearchAgent()
    result = agent.run("What are the main approaches to reducing hallucination in RAG systems?")
    print(json.dumps(result, indent=2)[:3000])
