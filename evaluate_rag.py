"""
Evaluate whether the RAG cites the correct source article, and answer quality.

The chatbot cites its sources deterministically: the unique guides from the
top-k search results, in rank order (see rag_helper.extract_sources). This
script checks those citations against the article-level ground truth
(question -> correct guide, see generate_ground_truth.py):

    correct_source_cited   True if the correct guide is among the cited sources
    source_rank            1-based rank of the correct guide among citations

Citation metrics need NO LLM calls. Optionally, a reference-free LLM judge
rates the generated answers: RELEVANT / PARTLY_RELEVANT / NON_RELEVANT.

Search backends:

    keyword   BM25 full-text search with title/section boosting (fast, no GPU needed)
    vector    Semantic search via all-MiniLM-L6-v2 embeddings (slower to index)
    both      Run both backends on the same questions and produce a comparison

Output
------
Single mode  (keyword or vector):
    output/rag-eval-<type>.csv / .json

Comparison mode (both):
    output/rag-eval-keyword.csv / .json
    output/rag-eval-vector.csv  / .json
    output/rag-eval-comparison.csv / .json   (combined, with search_type column)

Usage
-----
    uv run evaluate_rag.py                               # both backends, judge on
    uv run evaluate_rag.py --judge none                  # citation metrics only (free)
    uv run evaluate_rag.py --search-type vector          # vector backend only
    uv run evaluate_rag.py --sample 25                   # quick sample run
    uv run evaluate_rag.py --dry-run                     # estimate cost and exit

Options
-------
    --search-type  {keyword,vector,both}   Search backend(s) (default: both)
    --judge        {none,reference-free}   Judge mode (default: reference-free)
    --num-results  INT   Top-k search results / citation depth (default: 5)
    --sample       INT   Evaluate a random N-row sample instead of the full set
    --workers      INT   Parallel OpenAI requests (default: 6)
    --rag-model    STR   Model used to generate RAG answers (default: gpt-4o-mini)
    --judge-model  STR   Model used by the judge (default: gpt-4o-mini)
    --ground-truth PATH  Ground-truth CSV (default: eval/ground_truth/ground-truth-guides.csv)
    --output-dir   PATH  Directory to write results (default: output)
    --seed         INT   Random seed for sampling (default: 42)
    --dry-run            Estimate cost on a pilot and exit without a full run
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from evaluation_utils import calc_total_price, llm_structured_retry, map_progress
from ingest import build_index, build_vector_index, load_documents
from rag_helper import DEFAULT_MODEL, RAGBase, RAGVector, extract_sources

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_GROUND_TRUTH = Path("eval/ground_truth/ground-truth-guides.csv")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_NUM_RESULTS = 5
DEFAULT_WORKERS = 6
DEFAULT_JUDGE_MODEL = DEFAULT_MODEL  # gpt-4o-mini
PILOT_SIZE = 10


# ── Judge schema & prompts ───────────────────────────────────────────────────


class RelevanceEvaluation(BaseModel):
    reasoning: str = Field(description="Why the answer does or does not address the question.")
    relevance: Literal["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"] = Field(
        description="How well the answer addresses the question."
    )


RELEVANCE_INSTRUCTIONS = """
You are an expert evaluator for a question-answering system.
You are given ONLY a user question and the system's answer — there is NO
reference answer to compare against.

Judge whether the answer actually addresses the question:
- RELEVANT: fully and directly answers the question.
- PARTLY_RELEVANT: addresses the question but is incomplete or partly off-topic.
- NON_RELEVANT: does not answer the question, or says it has no information.

Explain your reasoning first.
""".strip()

RELEVANCE_PROMPT = """
Question:
{question}

Answer:
{answer_llm}
""".strip()


# ── Citation scoring ─────────────────────────────────────────────────────────


def score_citation(cited_guides: list[str], gt_guide: str) -> tuple[bool, int | None]:
    """Whether the correct guide is cited, and its 1-based rank among citations."""
    for rank, guide in enumerate(cited_guides, start=1):
        if guide == gt_guide:
            return True, rank
    return False, None


# ── RAG pipelines with usage tracking ────────────────────────────────────────


class _UsageTracking:
    """Mixin that records token usage of llm() calls for cost accounting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.usages: list = []

    def llm(self, prompt: str) -> str:
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(model=self.model, input=messages)
        self.usages.append(response.usage)  # list.append is thread-safe under the GIL
        return response.output_text


class RAGTracked(_UsageTracking, RAGBase):
    pass


class RAGVectorTracked(_UsageTracking, RAGVector):
    pass


def build_assistant(
    search_type: str,
    documents: list[dict],
    client: OpenAI,
    model: str,
) -> RAGTracked | RAGVectorTracked:
    if search_type == "vector":
        print("Building vector index (encoding documents — this may take a minute)...")
        index, embedder = build_vector_index(documents)
        return RAGVectorTracked(embedder=embedder, index=index, llm_client=client, model=model)
    else:
        print("Building keyword (BM25) index...")
        index = build_index(documents)
        return RAGTracked(index=index, llm_client=client, model=model)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_eval_inputs(ground_truth_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    documents = load_documents()
    ground_truth = pd.read_csv(ground_truth_path)

    known_guides = {doc["guide"] for doc in documents}
    n_before = len(ground_truth)
    ground_truth = ground_truth[ground_truth["guide"].isin(known_guides)].reset_index(drop=True)

    print(f"Ground-truth questions : {len(ground_truth)}  (dropped {n_before - len(ground_truth)} unresolvable)")
    print(f"Source documents       : {len(documents)}")
    print(f"Guides covered         : {ground_truth['guide'].nunique()}")
    return ground_truth, documents


def select_records(ground_truth: pd.DataFrame, sample: int | None, seed: int) -> list[dict]:
    if sample is not None and sample < len(ground_truth):
        ground_truth = ground_truth.sample(sample, random_state=seed)
    return ground_truth.to_dict(orient="records")


# ── Per-question evaluation ──────────────────────────────────────────────────


def evaluate_questions(
    assistant: RAGTracked | RAGVectorTracked,
    records: list[dict],
    num_results: int,
    with_answers: bool,
    workers: int,
) -> list[dict]:
    """Run search (+ optional answer generation) and score citations per question."""

    def evaluate_one(rec: dict) -> dict:
        if with_answers:
            answer, sources = assistant.rag_with_sources(rec["question"], num_results=num_results)
        else:
            answer = ""
            sources = extract_sources(assistant.search(rec["question"], num_results=num_results))

        cited = [s["guide"] for s in sources]
        correct, rank = score_citation(cited, rec["guide"])
        return {
            "question": rec["question"],
            "guide": rec["guide"],
            "guide_name": rec["guide_name"],
            "url": rec["url"],
            "cited_guides": "|".join(cited),
            "correct_source_cited": correct,
            "source_rank": rank,
            "answer_llm": answer,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = map_progress(pool, records, evaluate_one)

    if with_answers:
        print(f"Generated {len(rows)} RAG answers  |  cost: ${calc_total_price(assistant.usages):.4f}")
    return rows


# ── Judging ──────────────────────────────────────────────────────────────────


def judge_reference_free(
    client: OpenAI,
    rows: list[dict],
    model: str,
    workers: int,
) -> tuple[list[dict], list]:
    def judge_one(rec: dict) -> tuple[dict, object]:
        prompt = RELEVANCE_PROMPT.format(**rec)
        verdict, usage = llm_structured_retry(
            client, RELEVANCE_INSTRUCTIONS, prompt, RelevanceEvaluation, model=model
        )
        return {"relevance": verdict.relevance, "rel_reasoning": verdict.reasoning}, usage

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = map_progress(pool, rows, judge_one)

    verdicts = [r for r, _ in results]
    usages = [u for _, u in results if u is not None]
    return verdicts, usages


# ── Reporting & output ───────────────────────────────────────────────────────


def _citation_stats(df: pd.DataFrame) -> dict[str, str]:
    total = len(df)
    cited = int(df["correct_source_cited"].sum())
    rank1 = int((df["source_rank"] == 1).sum())
    mrr = (1 / df["source_rank"].dropna()).sum() / total
    return {
        "correct source cited": f"{cited}/{total} ({cited / total:.1%})",
        "cited at rank 1": f"{rank1}/{total} ({rank1 / total:.1%})",
        "guide MRR": f"{mrr:.3f}",
    }


def summarize(df: pd.DataFrame) -> None:
    search_types = df["search_type"].unique().tolist()
    print("\n=== Results ===")
    col_w = max(len(st) for st in search_types) + 6

    groups = {st: df[df["search_type"] == st] for st in search_types}
    stats = {st: _citation_stats(groups[st]) for st in search_types}

    header = f"{'Metric':<28}" + "".join(f"{st:>{col_w}}" for st in search_types)
    print(header)
    print("-" * len(header))
    for metric in next(iter(stats.values())):
        row = f"{metric:<28}" + "".join(f"{stats[st][metric]:>{col_w}}" for st in search_types)
        print(row)

    if "relevance" in df:
        for label in ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"):
            row = f"{label:<28}"
            for st in search_types:
                sub = groups[st]
                n = int((sub["relevance"] == label).sum())
                row += f"{f'{n}/{len(sub)} ({n / len(sub):.1%})':>{col_w}}"
            print(row)


def save(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_types = df["search_type"].unique().tolist()

    for st in search_types:
        sub = df[df["search_type"] == st].drop(columns=["search_type"])
        sub.to_csv(output_dir / f"rag-eval-{st}.csv", index=False)
        sub.to_json(output_dir / f"rag-eval-{st}.json", orient="records", indent=2)
        print(f"Saved {len(sub)} records to {output_dir}/rag-eval-{st}.{{csv,json}}")

    if len(search_types) > 1:
        df.to_csv(output_dir / "rag-eval-comparison.csv", index=False)
        df.to_json(output_dir / "rag-eval-comparison.json", orient="records", indent=2)
        print(f"Saved {len(df)} records to {output_dir}/rag-eval-comparison.{{csv,json}}")


# ── Main ─────────────────────────────────────────────────────────────────────


def _run_one_search_type(
    search_type: str,
    documents: list[dict],
    records: list[dict],
    client: OpenAI,
    run_judge: bool,
    args: argparse.Namespace,
) -> pd.DataFrame:
    assistant = build_assistant(search_type, documents, client, args.rag_model)

    print(f"\n[{search_type}] Evaluating {len(records)} questions...")
    rows = evaluate_questions(assistant, records, args.num_results, run_judge, args.workers)

    if run_judge:
        print(f"\n[{search_type}] Judging reference-free (relevance)...")
        verdicts, usages = judge_reference_free(client, rows, args.judge_model, args.workers)
        rows = [{**row, **verdict} for row, verdict in zip(rows, verdicts)]
        print(f"  judge cost: ${calc_total_price(usages):.4f}")

    df = pd.DataFrame(rows)
    df.insert(0, "search_type", search_type)
    return df


def estimate_cost(
    documents: list[dict],
    records: list[dict],
    client: OpenAI,
    args: argparse.Namespace,
) -> None:
    pilot_records = records[: min(PILOT_SIZE, len(records))]
    print(f"\nPilot run on {len(pilot_records)} questions (keyword backend)...")

    assistant = build_assistant("keyword", documents, client, args.rag_model)
    rows = evaluate_questions(assistant, pilot_records, args.num_results, True, args.workers)
    rag_cost = calc_total_price(assistant.usages)

    _, usages = judge_reference_free(client, rows, args.judge_model, args.workers)
    judge_cost = calc_total_price(usages)

    n_backends = 2 if args.search_type == "both" else 1
    pilot_cost = rag_cost + judge_cost
    scale = len(records) / len(pilot_records) * n_backends
    print(f"\nPilot cost ({len(pilot_records)} q)   : ${pilot_cost:.4f}")
    print(f"Estimated total ({len(records)} q x {n_backends} backends) : ${pilot_cost * scale:.2f}")
    print("Note: with --judge none the run is free (no LLM calls).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--search-type", choices=["keyword", "vector", "both"], default="both")
    parser.add_argument("--judge", choices=["none", "reference-free"], default="reference-free")
    parser.add_argument("--num-results", type=int, default=DEFAULT_NUM_RESULTS, metavar="INT")
    parser.add_argument("--sample", type=int, default=None, metavar="INT")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="INT")
    parser.add_argument("--rag-model", default=DEFAULT_MODEL, metavar="STR")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, metavar="STR")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH, metavar="PATH")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="PATH")
    parser.add_argument("--seed", type=int, default=42, metavar="INT")
    parser.add_argument("--dry-run", action="store_true", help="estimate cost on a pilot and exit")
    args = parser.parse_args()

    client = OpenAI(timeout=30.0)
    run_judge = args.judge == "reference-free"

    ground_truth, documents = load_eval_inputs(args.ground_truth)
    records = select_records(ground_truth, args.sample, args.seed)

    if args.dry_run:
        estimate_cost(documents, records, client, args)
        return

    search_types = ["keyword", "vector"] if args.search_type == "both" else [args.search_type]
    dfs = [
        _run_one_search_type(st, documents, records, client, run_judge, args)
        for st in search_types
    ]

    combined = pd.concat(dfs, ignore_index=True)
    summarize(combined)
    save(combined, args.output_dir)


if __name__ == "__main__":
    main()
