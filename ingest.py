"""Data loading and indexing for allaboutberlin.com guides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from minsearch import Index

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from minsearch import VectorSearch

DATA_DIR = Path(__file__).parent / "output" / "json"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents(json_dir: Path = DATA_DIR) -> list[dict]:
    documents: list[dict] = []
    for json_file in sorted(Path(json_dir).glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            file_docs = json.load(f)
        documents.extend(file_docs)
    return documents


def build_index(documents: list[dict]) -> Index:
    index = Index(
        text_fields=["title", "section", "text"],
        keyword_fields=["guide"],
    )
    index.fit(documents)
    return index


def build_vector_index(
    documents: list[dict],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[VectorSearch, SentenceTransformer]:
    from sentence_transformers import SentenceTransformer
    from minsearch import VectorSearch

    embedder = SentenceTransformer(model_name)
    texts = [f"{doc.get('section', '')} {doc.get('title', '')} {doc.get('text', '')}".strip() for doc in documents]
    vectors = embedder.encode(texts, batch_size=50, show_progress_bar=True)

    vindex: VectorSearch = VectorSearch(keyword_fields=["guide"])
    vindex.fit(vectors, documents)
    return vindex, embedder
