"""
End-to-end ANSWER-quality evaluation for the RAG chatbot (LLM-as-judge).

Unlike evaluate_search.py (retrieval-only, free), this exercises the full pipeline
the app uses — query condensation, retrieval, and generation, WITH conversation
history — and judges the final answer. It is therefore PAID: it makes OpenAI calls
for both generation (gpt-4o-mini) and judging (gpt-5.4-mini). Use --dry-run to see
a cost estimate first.

The cases (answer_eval_cases.json) come from the 2026-07 "roast" of the deployed
app and target what retrieval metrics can't see:

    factual     — a grounded, on-topic answer (not a refusal)
    multiturn   — the LAST turn stays on the thread's topic (tests history plumbing)
    offtopic    — the assistant refuses ("I don't have information ...")
    wrong_city  — the assistant flags that its guides only cover Berlin

Usage
-----
    uv run evaluate_answers.py
    uv run evaluate_answers.py --cases answer_eval_cases.json --workers 4
    uv run evaluate_answers.py --dry-run
    uv run evaluate_answers.py --output output/answer-eval.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from evaluation_utils import calc_total_price, llm_structured_retry, map_progress
from ingest import build_vector_index, load_documents
from rag_helper import DEFAULT_NUM_RESULTS, RERANK_MODEL, RAGVector, extract_sources, is_refusal

load_dotenv()

DEFAULT_CASES = Path("answer_eval_cases.json")
DEFAULT_WORKERS = 4

JUDGE_INSTRUCTIONS = """
You are grading a Berlin-expat RAG assistant that must answer ONLY from
allaboutberlin.com guides about living in Berlin, Germany.

You are given the conversation, the assistant's FINAL answer, and (when relevant)
the retrieved context that answer was supposed to be based on. Judge only the final
answer. Fill each boolean honestly:

- refused: true if the final answer declines or says it lacks the information
  (e.g. "I don't have information about this in my guides"), or otherwise does not
  attempt a substantive answer.
- grounded: true if the final answer's claims are supported by the retrieved
  context. If no context is provided or the answer is a refusal, set true.
- on_topic: true if the final answer actually addresses the user's latest question
  (and, for a stated TOPIC, stays on that topic rather than drifting to an
  unrelated subject).
- scope_caveat: true only if the answer explicitly signals that the guides cover
  Berlin and may not apply to the asked-about city/country. Otherwise false.

Give one sentence of reasoning.
""".strip()


class CaseVerdict(BaseModel):
    refused: bool
    grounded: bool
    on_topic: bool
    scope_caveat: bool
    reasoning: str


def run_case(pipeline: RAGVector, case: dict) -> dict:
    """Run all turns of a case through the real pipeline, threading history."""
    history: list[dict] = []
    transcript: list[dict] = []
    context = ""
    results: list[dict] = []
    for turn in case["turns"]:
        search_query = pipeline.condense_query(turn, history)
        results = pipeline.search(search_query, num_results=DEFAULT_NUM_RESULTS)
        context = pipeline.build_context(results)
        prompt = pipeline.build_prompt(turn, results)
        answer = pipeline.llm(prompt, history=history)
        transcript.append({"question": turn, "search_query": search_query, "answer": answer})
        history = history + [
            {"role": "user", "content": turn},
            {"role": "assistant", "content": answer},
        ]
    final_answer = transcript[-1]["answer"]
    cited = [s["guide"] for s in extract_sources(results)]
    return {"transcript": transcript, "context": context, "final_answer": final_answer, "cited": cited}


def build_judge_prompt(case: dict, run: dict) -> str:
    lines = [f"Case type: {case['type']}"]
    if case.get("topic"):
        lines.append(f"Topic the answer should stay on: {case['topic']}")
    lines.append("\nConversation:")
    for t in run["transcript"]:
        lines.append(f"User: {t['question']}")
        lines.append(f"Assistant: {t['answer']}")
    if case["type"] == "factual":
        lines.append("\nRetrieved context for the final answer:")
        lines.append(run["context"][:8000])
    return "\n".join(lines)


def passed(case: dict, verdict: CaseVerdict, refused_exact: bool) -> bool:
    ctype = case["type"]
    if ctype == "offtopic":
        return refused_exact or verdict.refused
    if ctype == "factual":
        return not refused_exact and verdict.grounded and verdict.on_topic
    if ctype == "multiturn":
        return not refused_exact and verdict.on_topic
    if ctype == "wrong_city":
        # Acceptable to either flag the Berlin scope or refuse outright — both avoid
        # the failure mode we care about (a confident answer for the wrong city).
        return verdict.scope_caveat or refused_exact
    raise ValueError(f"Unknown case type: {ctype}")


def make_grade_case(client: OpenAI, pipeline: RAGVector):
    def grade_case(case: dict) -> tuple[dict, object | None]:
        try:
            run = run_case(pipeline, case)
            verdict, usage = llm_structured_retry(
                client, JUDGE_INSTRUCTIONS, build_judge_prompt(case, run), CaseVerdict
            )
            refused_exact = is_refusal(run["final_answer"])
            hit = None
            if case.get("expected_guide"):
                hit = case["expected_guide"] in run["cited"]
            record = {
                "id": case["id"],
                "type": case["type"],
                "passed": passed(case, verdict, refused_exact),
                "refused": refused_exact or verdict.refused,
                "grounded": verdict.grounded,
                "on_topic": verdict.on_topic,
                "scope_caveat": verdict.scope_caveat,
                "hit": hit,
                "reasoning": verdict.reasoning,
                "final_answer": run["final_answer"],
            }
            return record, usage
        except Exception as e:  # keep the run alive; mark the case failed
            print(f"Failed case {case.get('id')}: {e}")
            return {"id": case.get("id"), "type": case.get("type"), "passed": False,
                    "error": str(e)}, None

    return grade_case


def print_report(records: list[dict]) -> None:
    print("\n=== Answer-quality results ===")
    print(f"{'id':<5} {'type':<11} {'pass':<5} {'refused':<8} {'grounded':<9} {'on_topic':<9} {'caveat':<7} {'hit':<5}")
    for r in sorted(records, key=lambda r: r["id"]):
        mark = "PASS" if r.get("passed") else "FAIL"
        hit = "-" if r.get("hit") is None else ("yes" if r["hit"] else "no")
        print(f"{r['id']:<5} {r['type']:<11} {mark:<5} "
              f"{str(r.get('refused', '')):<8} {str(r.get('grounded', '')):<9} "
              f"{str(r.get('on_topic', '')):<9} {str(r.get('scope_caveat', '')):<7} {hit:<5}")

    by_type: dict[str, list[bool]] = {}
    for r in records:
        by_type.setdefault(r["type"], []).append(bool(r.get("passed")))
    print("\nPass rate by type:")
    for ctype, flags in sorted(by_type.items()):
        print(f"  {ctype:<11} {sum(flags)}/{len(flags)}")
    total = sum(bool(r.get("passed")) for r in records)
    print(f"\nOverall: {total}/{len(records)} passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Cases JSON")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel cases")
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON here")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost on 2 cases and exit")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder reranker")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    print(f"Cases: {len(cases)} from {args.cases}")

    print("Loading documents and building vector index (~30s)...")
    documents = load_documents()
    vindex, embedder = build_vector_index(documents)
    reranker = None
    if not args.no_rerank:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker {RERANK_MODEL}...")
        reranker = CrossEncoder(RERANK_MODEL)
    client = OpenAI(timeout=90.0)
    pipeline = RAGVector(embedder=embedder, index=vindex, llm_client=client, reranker=reranker)
    grade_case = make_grade_case(client, pipeline)

    if args.dry_run:
        pilot = cases[:2]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = map_progress(pool, pilot, grade_case)
        pilot_cost = calc_total_price([u for _, u in results if u is not None])
        print(f"\nPilot cost ({len(pilot)} cases, judge only): ${pilot_cost:.4f}")
        print(f"Rough total ({len(cases)} cases): ${pilot_cost * len(cases) / len(pilot):.2f} "
              "(judge model only; generation cost is extra and small)")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = map_progress(pool, cases, grade_case)

    records = [r for r, _ in results]
    usages = [u for _, u in results if u is not None]
    print_report(records)
    print(f"\nJudge cost: ${calc_total_price(usages):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
