"""
Generate a ground-truth Q&A dataset for RAG evaluation.

Samples up to DOCS_PER_GUIDE documents from each guide in the knowledge base,
then uses OpenAI to generate one question per document. The result is a dataset
of ~300-400 Q&A pairs that covers all guide topics proportionally.

Only documents with all required fields (id, guide, text, section, title) are
included, so every record in the output is fully populated.

Output
------
output/ground-truth-data.csv
output/ground-truth-data.json

    Each record:
        question  — question a Berlin expat might ask
        document  — SHA1 document ID (links back to the source section)
        guide     — guide slug (e.g. "abmeldung")
        section   — H2 heading of the source section
        title     — H3 heading of the source section

Usage
-----
    uv run generate_ground_truth.py
    uv run generate_ground_truth.py --docs-per-guide 5 --workers 8

Options
-------
    --docs-per-guide  INT   Max documents sampled per guide (default: 3)
    --workers         INT   Parallel OpenAI requests (default: 6)
    --seed            INT   Random seed for reproducible sampling (default: 42)
    --output-dir      PATH  Directory to write output files (default: output)
    --dry-run               Print cost estimate and exit without generating
"""

import argparse
import json
import random
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

REQUIRED_FIELDS = ("id", "guide", "text", "section", "title")

PROMPT = """
You emulate an expat or foreign resident living in Germany who needs help navigating
bureaucracy, banking, housing, healthcare, taxes, and everyday life.
Formulate ONE question this person might ask based on the guide section below.
The section should contain the answer. Make the question complete and specific.
Use as few words from the section as possible.
""".strip()


class Question(BaseModel):
    question: str


def sample_documents(
    documents: list[dict],
    docs_per_guide: int,
    seed: int,
) -> tuple[list[dict], dict]:
    random.seed(seed)

    valid = [doc for doc in documents if all(doc.get(f) for f in REQUIRED_FIELDS)]
    print(f"Valid documents : {len(valid)} / {len(documents)}")

    by_guide: dict[str, list[dict]] = defaultdict(list)
    for doc in valid:
        by_guide[doc["guide"]].append(doc)

    sampled = []
    for guide_docs in by_guide.values():
        sampled.extend(random.sample(guide_docs, min(len(guide_docs), docs_per_guide)))

    print(f"Sampled         : {len(sampled)} documents from {len(by_guide)} guides")
    return sampled, by_guide


def make_generate_record(client: OpenAI):
    def generate_record(doc: dict) -> tuple[dict | None, object | None]:
        try:
            result, usage = llm_structured_retry(client, PROMPT, json.dumps(doc), Question)
            record = {
                "question": result.question,
                "document": doc["id"],
                "guide": doc["guide"],
                "section": doc["section"],
                "title": doc["title"],
            }
            return record, usage
        except Exception as e:
            print(f"Failed for doc {doc['id']}: {e}")
            return None, None

    return generate_record


def estimate_cost(
    sampled: list[dict],
    generate_record,
    workers: int,
) -> float:
    pilot_size = min(10, len(sampled))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pilot = map_progress(pool, sampled[:pilot_size], generate_record)

    pilot_usages = [u for _, u in pilot if u is not None]
    pilot_cost = calc_total_price(pilot_usages)
    estimated = pilot_cost * (len(sampled) / pilot_size)

    print(f"Pilot cost ({pilot_size} docs)          : ${pilot_cost:.4f}")
    print(f"Estimated total ({len(sampled)} docs) : ${estimated:.2f}")
    return estimated


def generate(sampled: list[dict], generate_record, workers: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = map_progress(pool, sampled, generate_record)

    records = [r for r, _ in results if r is not None]
    usages = [u for _, u in results if u is not None]

    total_cost = calc_total_price(usages)
    print(f"Generated {len(records)} records  |  total cost: ${total_cost:.4f}")
    return records


def save(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "ground-truth-data.csv", index=False)
    df.to_json(output_dir / "ground-truth-data.json", orient="records", indent=2)

    print(f"Saved {len(df)} records to {output_dir}/ground-truth-data.{{csv,json}}")

    print(f"\nTotal Q&A pairs : {len(df)}")
    print(f"Guides covered  : {df['guide'].nunique()}")
    print(f"Nulls           : {df.isnull().sum().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-per-guide", type=int, default=3, metavar="INT")
    parser.add_argument("--workers", type=int, default=6, metavar="INT")
    parser.add_argument("--seed", type=int, default=42, metavar="INT")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), metavar="PATH")
    parser.add_argument("--dry-run", action="store_true", help="estimate cost and exit")
    args = parser.parse_args()

    client = OpenAI(timeout=30.0)
    generate_record = make_generate_record(client)

    documents = load_documents()
    sampled, _ = sample_documents(documents, args.docs_per_guide, args.seed)

    estimate_cost(sampled, generate_record, args.workers)

    if args.dry_run:
        return

    records = generate(sampled, generate_record, args.workers)
    save(records, args.output_dir)


if __name__ == "__main__":
    main()
