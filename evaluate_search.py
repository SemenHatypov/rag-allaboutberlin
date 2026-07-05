"""Evaluate article-level retrieval quality of text (BM25) and vector search.

A query is a "hit" when the correct guide (article) appears among the unique
guides of the top-k search results — the same guides the chatbot cites as
source links. MRR uses the rank of the correct guide among those unique guides.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from evaluation_utils import map_progress
from ingest import build_index, build_vector_index, load_documents

DEFAULT_GROUND_TRUTH = Path("output/ground-truth-guides.csv")
DEFAULT_NUM_RESULTS = 5
DEFAULT_WORKERS = 4
TITLE_BOOSTS = [0.5, 1.0, 2.0, 3.0, 5.0]
SECTION_BOOSTS = [0.1, 0.5, 1.0]


def unique_guides(results: list[dict]) -> list[str]:
    """Unique guide slugs from ranked results, preserving rank order."""
    seen: set[str] = set()
    guides: list[str] = []
    for doc in results:
        if doc["guide"] not in seen:
            seen.add(doc["guide"])
            guides.append(doc["guide"])
    return guides


def hit_rate(relevance_total: list[list[int]]) -> float:
    cnt = sum(1 for line in relevance_total if 1 in line)
    return cnt / len(relevance_total)


def mrr(relevance_total: list[list[int]]) -> float:
    total_score = 0.0
    for line in relevance_total:
        for rank, val in enumerate(line):
            if val == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(
    ground_truth: pd.DataFrame,
    search_function,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, float]:
    def process_row(row):
        results = search_function(row["question"])
        guides = unique_guides(results)
        return [1 if guide == row["guide"] else 0 for guide in guides]

    rows = [row for _, row in ground_truth.iterrows()]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        relevance_total = map_progress(pool, rows, process_row)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


def make_text_search(index, title_boost: float, section_boost: float, num_results: int = DEFAULT_NUM_RESULTS):
    def search(query: str):
        return index.search(
            query,
            boost_dict={"title": title_boost, "section": section_boost},
            num_results=num_results,
        )
    return search


def run_grid_search(
    ground_truth: pd.DataFrame,
    index,
    workers: int,
    num_results: int = DEFAULT_NUM_RESULTS,
) -> pd.DataFrame:
    total = len(TITLE_BOOSTS) * len(SECTION_BOOSTS)
    print(f"\nGrid search: {total} combinations")

    results = []
    for title_b in TITLE_BOOSTS:
        for section_b in SECTION_BOOSTS:
            result = evaluate(
                ground_truth,
                make_text_search(index, title_b, section_b, num_results),
                workers=workers,
            )
            results.append({"title_boost": title_b, "section_boost": section_b, **result})
            print(
                f"  title={title_b:.1f}  section={section_b:.1f}: "
                f"hit_rate={result['hit_rate']:.3f}  mrr={result['mrr']:.3f}"
            )

    return pd.DataFrame(results).sort_values("mrr", ascending=False)


def print_comparison(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).set_index("method")
    df["hit_rate"] = df["hit_rate"].map("{:.3f}".format)
    df["mrr"] = df["mrr"].map("{:.3f}".format)
    print("\n=== Results ===")
    print(df.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality of text (BM25) and vector search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["text", "vector", "all"],
        default="all",
        help="Which search method(s) to evaluate",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run grid search over BM25 boost parameters",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to ground truth CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results JSON to this path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers for evaluation",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=DEFAULT_NUM_RESULTS,
        help="Top-k search results (bounds the number of citable sources)",
    )
    args = parser.parse_args()

    print(f"Ground truth : {args.ground_truth}")
    ground_truth = pd.read_csv(args.ground_truth)
    print(f"  {len(ground_truth)} records  /  {ground_truth['guide'].nunique()} guides")

    print("\nLoading documents...")
    documents = load_documents()
    print(f"  {len(documents)} documents")

    comparison_rows: list[dict] = []

    if args.method in ("text", "all"):
        print("\nBuilding BM25 index...")
        index = build_index(documents)

        best_title, best_section = 2.0, 0.5

        if args.tune:
            grid_df = run_grid_search(ground_truth, index, args.workers, args.num_results)
            best_title = grid_df.iloc[0]["title_boost"]
            best_section = grid_df.iloc[0]["section_boost"]
            print(f"\nBest params: title={best_title}, section={best_section}")

        print(f"\nEvaluating text_search (title=2.0, section=0.5)...")
        default_result = evaluate(
            ground_truth,
            make_text_search(index, 2.0, 0.5, args.num_results),
            workers=args.workers,
        )
        comparison_rows.append({"method": "text_search (title=2.0, section=0.5)", **default_result})

        if args.tune and (best_title, best_section) != (2.0, 0.5):
            print(f"Evaluating text_search (title={best_title}, section={best_section})...")
            tuned_result = evaluate(
                ground_truth,
                make_text_search(index, best_title, best_section, args.num_results),
                workers=args.workers,
            )
            comparison_rows.append({
                "method": f"text_search (title={best_title}, section={best_section})",
                **tuned_result,
            })

    if args.method in ("vector", "all"):
        print("\nBuilding vector index (~30s)...")
        vindex, embedder = build_vector_index(documents)

        def vector_search(query: str):
            vec = embedder.encode(query)
            return vindex.search(vec, num_results=args.num_results)

        print("Evaluating vector_search...")
        vector_result = evaluate(ground_truth, vector_search, workers=args.workers)
        comparison_rows.append({"method": "vector_search (all-MiniLM-L6-v2)", **vector_result})

    print_comparison(comparison_rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comparison_rows, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
