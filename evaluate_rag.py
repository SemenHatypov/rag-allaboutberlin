"""
Evaluate answer quality of the RAG system with an LLM-as-a-judge.

Retrieval metrics (Hit Rate, MRR — see evaluate_search.py) only tell us whether
the right document was found. This script measures whether the *generated answer*
is actually correct, using a second LLM call as the judge.

It follows the A -> Q -> A' setup of the ground-truth dataset:

    A   original guide section text          (the "correct" answer)
     └─ Q   a question generated from it      (output/ground-truth-data.csv)
         └─ A'  the answer the RAG returns    (generated here)

Two judge modes are supported:

    reference        Judge sees Q, the original answer A, and the RAG answer A'.
                     Scores each answer 'good' / 'bad'. (offline / development)
    reference-free   Judge sees only Q and A'. Scores relevance
                     RELEVANT / PARTLY_RELEVANT / NON_RELEVANT. (online / production)

Three search backends are supported:

    keyword   BM25 full-text search with title/section boosting (fast, no GPU needed)
    vector    Semantic search via all-MiniLM-L6-v2 embeddings (slower to index)
    both      Run both backends on the same questions and produce a comparison

Output
------
Single mode  (keyword or vector):
    output/rag-eval-<type>.csv / .json
    output/rag-eval.csv / .json   (backwards-compat alias for keyword)

Comparison mode (both):
    output/rag-eval-keyword.csv / .json
    output/rag-eval-vector.csv  / .json
    output/rag-eval-comparison.csv / .json   (combined, with search_type column)

    Each record contains the question, RAG answer, original answer, plus the
    judge verdict columns for the mode(s) that were run:
        score, reasoning                  (reference mode)
        relevance, rel_reasoning          (reference-free mode)

Usage
-----
    uv run evaluate_rag.py                               # keyword, reference judge, full dataset
    uv run evaluate_rag.py --search-type vector          # vector search backend
    uv run evaluate_rag.py --search-type both            # compare both backends
    uv run evaluate_rag.py --sample 25                   # quick sample run
    uv run evaluate_rag.py --judge both                  # both judge modes
    uv run evaluate_rag.py --dry-run --sample 50         # estimate cost and exit

Options
-------
    --search-type  {keyword,vector,both}  Search backend(s) (default: keyword)
    --judge        {reference,reference-free,both}  Judge mode(s) (default: reference)
    --sample       INT   Evaluate a random N-row sample instead of the full set
    --workers      INT   Parallel OpenAI requests (default: 6)
    --rag-model    STR   Model used to generate RAG answers (default: gpt-4o-mini)
    --judge-model  STR   Model used by the judge (default: gpt-4o-mini)
    --ground-truth PATH  Path to ground-truth CSV (default: output/ground-truth-data.csv)
    --output-dir   PATH  Directory to write results (default: output)
    --seed         INT   Random seed for sampling (default: 42)
    --dry-run            Estimate cost on a pilot and exit without a full run
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from evaluation_utils import calc_total_price, llm_structured_retry, map_progress
from ingest import build_index, build_vector_index, load_documents
from rag_helper import DEFAULT_MODEL, RAGBase, RAGVector

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_GROUND_TRUTH = Path("output/ground-truth-data.csv")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_WORKERS = 6
DEFAULT_JUDGE_MODEL = DEFAULT_MODEL  # gpt-4o-mini
PILOT_SIZE = 10


# ── Judge schemas & prompts ──────────────────────────────────────────────────


class AnswerEvaluation(BaseModel):
    reasoning: str = Field(description="Reasoning about the quality of the answer.")
    score: Literal["good", "bad"] = Field(
        description="'good' if the answer is correct and complete, 'bad' otherwise."
    )


class RelevanceEvaluation(BaseModel):
    reasoning: str = Field(description="Why the answer does or does not address the question.")
    relevance: Literal["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"] = Field(
        description="How well the answer addresses the question."
    )


JUDGE_INSTRUCTIONS = """
You compare an AI-generated answer against the original ground-truth answer
for a question. The AI answer does NOT need to match word for word.

Mark it 'good' if it conveys the same key information and is factually correct.
Mark it 'bad' only if the AI answer is wrong, contradicts the original, or
misses the key point.

Always explain your reasoning before giving the score.
""".strip()

JUDGE_PROMPT = """
Question:
{question}

Original Answer (ground truth):
{answer_orig}

AI Answer:
{answer_llm}
""".strip()

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


# ── RAG pipeline with usage tracking ─────────────────────────────────────────


class RAGTracked(RAGBase):
    """RAGBase that records token usage for cost accounting."""

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


class RAGVectorTracked(RAGVector):
    """RAGVector that records token usage for cost accounting."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.usages: list = []

    def llm(self, prompt: str) -> str:
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(model=self.model, input=messages)
        self.usages.append(response.usage)
        return response.output_text


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


def load_eval_inputs(ground_truth_path: Path) -> tuple[pd.DataFrame, dict, list[dict]]:
    documents = load_documents()
    doc_idx = {doc["id"]: doc for doc in documents if doc.get("id")}

    ground_truth = pd.read_csv(ground_truth_path)
    n_before = len(ground_truth)
    ground_truth = ground_truth[ground_truth["document"].isin(doc_idx)].reset_index(drop=True)

    print(f"Ground-truth questions : {len(ground_truth)}  (dropped {n_before - len(ground_truth)} unresolvable)")
    print(f"Source documents       : {len(documents)}")
    print(f"Guides covered         : {ground_truth['guide'].nunique()}")
    return ground_truth, doc_idx, documents


def select_records(ground_truth: pd.DataFrame, sample: int | None, seed: int) -> list[dict]:
    if sample is not None and sample < len(ground_truth):
        ground_truth = ground_truth.sample(sample, random_state=seed)
    return ground_truth.to_dict(orient="records")


# ── RAG answer generation (A') ───────────────────────────────────────────────


def generate_rag_answers(
    assistant: RAGTracked,
    doc_idx: dict,
    records: list[dict],
    workers: int,
) -> list[dict]:
    def generate_one(rec: dict) -> dict:
        return {
            "question": rec["question"],
            "answer_llm": assistant.rag(rec["question"]),
            "answer_orig": doc_idx[rec["document"]]["text"],
            "document": rec["document"],
            "guide": rec["guide"],
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = map_progress(pool, records, generate_one)

    print(f"Generated {len(answers)} RAG answers  |  cost: ${calc_total_price(assistant.usages):.4f}")
    return answers


# ── Judging ──────────────────────────────────────────────────────────────────


def run_judge(
    client: OpenAI,
    answers: list[dict],
    instructions: str,
    prompt_template: str,
    schema: type[BaseModel],
    fields: dict[str, str],
    model: str,
    workers: int,
) -> tuple[list[dict], list]:
    """Run a judge over every answer. `fields` maps schema attr -> output column."""

    def judge_one(rec: dict) -> tuple[dict, object]:
        prompt = prompt_template.format(**rec)
        verdict, usage = llm_structured_retry(client, instructions, prompt, schema, model=model)
        row = {out_key: getattr(verdict, attr) for attr, out_key in fields.items()}
        return row, usage

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = map_progress(pool, answers, judge_one)

    rows = [r for r, _ in results]
    usages = [u for _, u in results if u is not None]
    return rows, usages


def judge_with_reference(client: OpenAI, answers: list[dict], model: str, workers: int):
    return run_judge(
        client, answers, JUDGE_INSTRUCTIONS, JUDGE_PROMPT, AnswerEvaluation,
        {"score": "score", "reasoning": "reasoning"}, model, workers,
    )


def judge_reference_free(client: OpenAI, answers: list[dict], model: str, workers: int):
    return run_judge(
        client, answers, RELEVANCE_INSTRUCTIONS, RELEVANCE_PROMPT, RelevanceEvaluation,
        {"relevance": "relevance", "reasoning": "rel_reasoning"}, model, workers,
    )


# ── Reporting & output ───────────────────────────────────────────────────────


def summarize(df: pd.DataFrame) -> None:
    search_types = df["search_type"].unique().tolist() if "search_type" in df else []
    if len(search_types) > 1:
        _summarize_comparison(df, search_types)
    else:
        label = search_types[0] if search_types else "results"
        print(f"\n=== Results ({label}) ===")
        _summarize_single(df)


def _summarize_single(df: pd.DataFrame) -> None:
    if "score" in df:
        counts = df["score"].value_counts()
        good, total = int(counts.get("good", 0)), len(df)
        print(f"Reference judge   : good {good}/{total} ({good / total:.1%})  |  bad {total - good}/{total}")
    if "relevance" in df:
        print("Reference-free    :")
        for label in ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"):
            n = int((df["relevance"] == label).sum())
            print(f"    {label:<16} {n}/{len(df)} ({n / len(df):.1%})")


def _summarize_comparison(df: pd.DataFrame, search_types: list[str]) -> None:
    print(f"\n{'=== Comparison: ' + ' vs '.join(search_types) + ' ==='}")
    col_w = 12

    def pct(sub: pd.DataFrame, col: str, val: str) -> str:
        if col not in sub:
            return "  n/a   "
        n = int((sub[col] == val).sum())
        return f"{n}/{len(sub)} ({n / len(sub):.1%})"

    header = f"{'Metric':<28}" + "".join(f"{st:>{col_w}}" for st in search_types)
    print(header)
    print("-" * len(header))

    groups = {st: df[df["search_type"] == st] for st in search_types}

    if "score" in df:
        row = f"{'Reference: good':<28}"
        row += "".join(f"{pct(groups[st], 'score', 'good'):>{col_w}}" for st in search_types)
        print(row)

    if "relevance" in df:
        for label in ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"):
            row = f"{label:<28}"
            row += "".join(f"{pct(groups[st], 'relevance', label):>{col_w}}" for st in search_types)
            print(row)


def save(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_types = df["search_type"].unique().tolist() if "search_type" in df else []

    if len(search_types) > 1:
        # Comparison mode: save per-type files + combined
        for st in search_types:
            sub = df[df["search_type"] == st].drop(columns=["search_type"])
            sub.to_csv(output_dir / f"rag-eval-{st}.csv", index=False)
            sub.to_json(output_dir / f"rag-eval-{st}.json", orient="records", indent=2)
            print(f"Saved {len(sub)} records to {output_dir}/rag-eval-{st}.{{csv,json}}")
        df.to_csv(output_dir / "rag-eval-comparison.csv", index=False)
        df.to_json(output_dir / "rag-eval-comparison.json", orient="records", indent=2)
        print(f"Saved {len(df)} records to {output_dir}/rag-eval-comparison.{{csv,json}}")
    else:
        st = search_types[0] if search_types else "keyword"
        out = df.drop(columns=["search_type"]) if "search_type" in df else df
        out.to_csv(output_dir / f"rag-eval-{st}.csv", index=False)
        out.to_json(output_dir / f"rag-eval-{st}.json", orient="records", indent=2)
        print(f"\nSaved {len(out)} records to {output_dir}/rag-eval-{st}.{{csv,json}}")
        if st == "keyword":
            # Backwards-compat alias
            out.to_csv(output_dir / "rag-eval.csv", index=False)
            out.to_json(output_dir / "rag-eval.json", orient="records", indent=2)


def build_dataframe(answers: list[dict], verdict_rows: list[list[dict]]) -> pd.DataFrame:
    rows = []
    for i, answer in enumerate(answers):
        row = dict(answer)
        for verdicts in verdict_rows:
            row.update(verdicts[i])
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────


def _run_one_search_type(
    search_type: str,
    documents: list[dict],
    doc_idx: dict,
    records: list[dict],
    client: OpenAI,
    run_reference: bool,
    run_reference_free: bool,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run RAG generation + judging for a single search backend. Returns a DataFrame."""
    assistant = build_assistant(search_type, documents, client, args.rag_model)

    print(f"\n[{search_type}] Generating RAG answers for {len(records)} questions...")
    answers = generate_rag_answers(assistant, doc_idx, records, args.workers)

    verdict_rows: list[list[dict]] = []
    if run_reference:
        print(f"\n[{search_type}] Judging with reference (good/bad)...")
        rows, usages = judge_with_reference(client, answers, args.judge_model, args.workers)
        verdict_rows.append(rows)
        print(f"  judge cost: ${calc_total_price(usages):.4f}")
    if run_reference_free:
        print(f"\n[{search_type}] Judging reference-free (relevance)...")
        rows, usages = judge_reference_free(client, answers, args.judge_model, args.workers)
        verdict_rows.append(rows)
        print(f"  judge cost: ${calc_total_price(usages):.4f}")

    df = build_dataframe(answers, verdict_rows)
    df.insert(0, "search_type", search_type)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--search-type", choices=["keyword", "vector", "both"], default="keyword")
    parser.add_argument("--judge", choices=["reference", "reference-free", "both"], default="reference")
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
    run_reference = args.judge in ("reference", "both")
    run_reference_free = args.judge in ("reference-free", "both")

    ground_truth, doc_idx, documents = load_eval_inputs(args.ground_truth)
    records = select_records(ground_truth, args.sample, args.seed)

    if args.dry_run:
        # Pilot on keyword backend for cost estimate
        assistant = build_assistant("keyword", documents, client, args.rag_model)
        estimate_cost(client, assistant, doc_idx, records, run_reference, run_reference_free, args)
        return

    search_types = ["keyword", "vector"] if args.search_type == "both" else [args.search_type]
    dfs: list[pd.DataFrame] = []
    for st in search_types:
        df = _run_one_search_type(
            st, documents, doc_idx, records, client,
            run_reference, run_reference_free, args,
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    summarize(combined)
    save(combined, args.output_dir)


def estimate_cost(
    client: OpenAI,
    assistant: RAGTracked | RAGVectorTracked,
    doc_idx: dict,
    records: list[dict],
    run_reference: bool,
    run_reference_free: bool,
    args: argparse.Namespace,
) -> None:
    pilot_records = records[: min(PILOT_SIZE, len(records))]
    print(f"\nPilot run on {len(pilot_records)} questions...")

    answers = generate_rag_answers(assistant, doc_idx, pilot_records, args.workers)
    rag_cost = calc_total_price(assistant.usages)

    judge_cost = 0.0
    if run_reference:
        _, usages = judge_with_reference(client, answers, args.judge_model, args.workers)
        judge_cost += calc_total_price(usages)
    if run_reference_free:
        _, usages = judge_reference_free(client, answers, args.judge_model, args.workers)
        judge_cost += calc_total_price(usages)

    pilot_cost = rag_cost + judge_cost
    scale = len(records) / len(pilot_records)
    print(f"\nPilot cost ({len(pilot_records)} q)   : ${pilot_cost:.4f}")
    print(f"Estimated total ({len(records)} q) : ${pilot_cost * scale:.2f}")


if __name__ == "__main__":
    main()
