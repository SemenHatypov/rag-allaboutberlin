"""
Precompute document embeddings into output/json/embeddings.npz.

The Streamlit app loads these vectors at startup instead of re-encoding all
~1800 documents on every cold start (which takes ~1 min). Run this after any
re-scrape so the committed .npz matches the corpus; build_vector_index checks a
fingerprint and silently falls back to encoding if they ever drift.

Usage
-----
    uv run python build_embeddings.py
    uv run python build_embeddings.py --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse

from ingest import DEFAULT_EMBEDDING_MODEL, load_documents, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name")
    args = parser.parse_args()

    documents = load_documents()
    print(f"Encoding {len(documents)} documents with {args.model}...")
    path = save_embeddings(documents, model_name=args.model)
    print(f"Saved embeddings to {path}")


if __name__ == "__main__":
    main()
