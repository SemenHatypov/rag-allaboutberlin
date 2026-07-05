"""
Generate an article-level ground-truth dataset for RAG evaluation.

Groups the knowledge base by guide (article), builds one text digest per guide,
then uses OpenAI to generate QUESTIONS_PER_GUIDE distinct questions that this
guide answers best. The result is a dataset of "question -> correct guide"
pairs used to evaluate whether the RAG cites the right article as its source.

Output
------
output/ground-truth-guides.csv
output/ground-truth-guides.json

    Each record:
        question    — question a Berlin expat might ask
        guide       — guide slug (e.g. "abmeldung") that best answers it
        guide_name  — human-readable guide title
        url         — canonical article URL on allaboutberlin.com

Usage
-----
    uv run generate_ground_truth.py
    uv run generate_ground_truth.py --questions-per-guide 3 --workers 8

Options
-------
    --questions-per-guide  INT   Questions generated per guide (default: 2)
    --workers              INT   Parallel OpenAI requests (default: 6)
    --output-dir           PATH  Directory to write output files (default: output)
    --dry-run                    Print cost estimate and exit
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from evaluation_utils import calc_total_price, llm_structured_retry, map_progress
from ingest import load_documents

load_dotenv()

MAX_GUIDE_CHARS = 25_000
PER_SECTION_CHARS = 800

PROMPT = """
You emulate an expat or foreign resident living in Germany who needs help navigating
bureaucracy, banking, housing, healthcare, taxes, and everyday life.
Below is the content of ONE guide from allaboutberlin.com titled "{guide_name}".
Formulate {n} DISTINCT questions this person might ask that THIS guide answers.
Each question must be specific enough that this guide is the best single source
for the answer, must be complete on its own, and should use as few words copied
from the guide as possible. Cover different parts of the guide.
""".strip()


class GuideQuestions(BaseModel):
    questions: list[str]


def group_by_guide(documents: list[dict]) -> dict[str, list[dict]]:
    """Group documents by guide slug, keeping only docs with non-empty text."""
    by_guide: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        if doc.get("text"):
            by_guide[doc["guide"]].append(doc)
    return dict(by_guide)


def build_guide_text(
    docs: list[dict],
    max_chars: int = MAX_GUIDE_CHARS,
    per_section_chars: int = PER_SECTION_CHARS,
) -> str:
    """Concatenate a guide's sections into one digest, in document order."""
    blocks: list[str] = []
    total = 0
    for doc in docs:
        heading = " — ".join(part for part in (doc.get("section"), doc.get("title")) if part)
        block = f"## {heading}\n{doc['text'][:per_section_chars]}" if heading else doc["text"][:per_section_chars]
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def make_guide_record(doc: dict, question: str) -> dict:
    """Build one ground-truth record from a guide's representative doc."""
    return {
        "question": question,
        "guide": doc["guide"],
        "guide_name": doc["guide_name"],
        "url": doc["url"],
    }


def make_generate_records(client: OpenAI, questions_per_guide: int):
    def generate_records(docs: list[dict]) -> tuple[list[dict], object | None]:
        first = docs[0]
        try:
            prompt = PROMPT.format(guide_name=first["guide_name"], n=questions_per_guide)
            result, usage = llm_structured_retry(client, prompt, build_guide_text(docs), GuideQuestions)
            questions = result.questions[:questions_per_guide]
            return [make_guide_record(first, q) for q in questions], usage
        except Exception as e:
            print(f"Failed for guide {first['guide']}: {e}")
            return [], None

    return generate_records


def estimate_cost(
    guide_groups: list[list[dict]],
    generate_records,
    workers: int,
) -> float:
    pilot_size = min(5, len(guide_groups))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pilot = map_progress(pool, guide_groups[:pilot_size], generate_records)

    pilot_usages = [u for _, u in pilot if u is not None]
    pilot_cost = calc_total_price(pilot_usages)
    estimated = pilot_cost * (len(guide_groups) / pilot_size)

    print(f"Pilot cost ({pilot_size} guides)          : ${pilot_cost:.4f}")
    print(f"Estimated total ({len(guide_groups)} guides) : ${estimated:.2f}")
    return estimated


def generate(guide_groups: list[list[dict]], generate_records, workers: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = map_progress(pool, guide_groups, generate_records)

    records = [r for recs, _ in results for r in recs]
    usages = [u for _, u in results if u is not None]

    total_cost = calc_total_price(usages)
    print(f"Generated {len(records)} records  |  total cost: ${total_cost:.4f}")
    return records


def save(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "ground-truth-guides.csv", index=False)
    df.to_json(output_dir / "ground-truth-guides.json", orient="records", indent=2)

    print(f"Saved {len(df)} records to {output_dir}/ground-truth-guides.{{csv,json}}")

    print(f"\nTotal questions : {len(df)}")
    print(f"Guides covered  : {df['guide'].nunique()}")
    print(f"Nulls           : {df.isnull().sum().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions-per-guide", type=int, default=2, metavar="INT")
    parser.add_argument("--workers", type=int, default=6, metavar="INT")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), metavar="PATH")
    parser.add_argument("--dry-run", action="store_true", help="estimate cost and exit")
    args = parser.parse_args()

    client = OpenAI(timeout=60.0)
    generate_records = make_generate_records(client, args.questions_per_guide)

    documents = load_documents()
    by_guide = group_by_guide(documents)
    guide_groups = list(by_guide.values())
    print(f"Guides with documents : {len(guide_groups)} ({len(documents)} sections)")

    estimate_cost(guide_groups, generate_records, args.workers)

    if args.dry_run:
        return

    records = generate(guide_groups, generate_records, args.workers)
    save(records, args.output_dir)


if __name__ == "__main__":
    main()
