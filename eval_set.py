"""
eval_set.py
-----------
Minimal evaluation harness. This is the difference between "I built a demo"
and "I built something I can measure" -- do NOT skip this, it's the first
thing a sharp evaluator will ask about.

Fill in TEST_QUESTIONS with 10-15 real questions from your domain of choice
and, ideally, a short note on what a correct answer should mention. Then run
this file to get a simple before/after comparison: single-pass synthesis vs
the full agent (with self-eval + refinement).

This won't give you an automatic "accuracy %" -- that requires either manual
grading or a second LLM-as-judge call, both of which are legitimate for an
MCA project. Pick one, be upfront about which, and note the limitation in
your report (an LLM judge is imperfect; say so).

NOTE: Gemini's free tier is rate-limited (5 requests/min for this model),
and each question can burn several requests. A short pause between
questions here keeps you well under that limit during a full eval run.
"""

import json
import time
from agent import ResearchAgent

TEST_QUESTIONS = [
    "What are the main approaches to reducing hallucination in RAG systems?",
    "How do transformer models handle long-context retrieval?",
    "What evaluation metrics are used for question-answering systems?",
    "What techniques are used to detect hallucinations in RAG outputs without ground-truth references?",
    "How does multi-source retrieval increase or decrease hallucination risk compared to single-source retrieval?",
    "What role does chunking strategy play in reducing hallucination in RAG pipelines?",
    "How do iterative or self-refining RAG frameworks differ from single-pass RAG in reducing factual errors?",
    "What are the known limitations of using an LLM as a judge to evaluate its own RAG-generated answers?",
    "How is hallucination measured or benchmarked in medical or legal domain RAG systems specifically?",
    "What retrieval-augmented approaches exist for reducing hallucination in multi-hop reasoning tasks?",
    "How does membership inference or privacy risk relate to hallucination behavior in RAG systems?",
    "What debate-based or multi-agent approaches have been proposed to reduce hallucination in generation pipelines?",
]


def run_eval():
    agent = ResearchAgent()
    results = []

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\nRunning: {q}")
        result = agent.run(q)
        results.append({
            "question": q,
            "answer": result["answer"],
            "refinements_used": result["refinements_used"],
            "papers_count": len(result["papers"]),
            "final_sufficient": result["final_verdict"].get("sufficient"),
        })
        print(f"  refinements_used={result['refinements_used']}  "
              f"papers={len(result['papers'])}  "
              f"sufficient={result['final_verdict'].get('sufficient')}")

        # Pause between questions to stay under the free-tier rate limit.
        if i < len(TEST_QUESTIONS) - 1:
            time.sleep(15)

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary stats you can drop straight into your report
    refined_count = sum(1 for r in results if r["refinements_used"] > 0)
    print(f"\n--- Summary ---")
    print(f"Total questions: {len(results)}")
    print(f"Required refinement: {refined_count} ({refined_count/len(results)*100:.0f}%)")
    print(f"Saved full results to eval_results.json")


if __name__ == "__main__":
    run_eval()
