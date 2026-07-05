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
GUIDES_FILE = "guides.json"
BASE_GUIDE_URL = "https://allaboutberlin.com/guides"


def guide_url(slug: str) -> str:
    return f"{BASE_GUIDE_URL}/{slug}"


def load_guides(json_dir: Path = DATA_DIR) -> list[dict]:
    guides_path = Path(json_dir) / GUIDES_FILE
    if not guides_path.exists():
        return []
    with open(guides_path, encoding="utf-8") as f:
        return json.load(f)


def load_documents(json_dir: Path = DATA_DIR) -> list[dict]:
    name_map = {g["guide"]: g["guide_name"] for g in load_guides(json_dir)}
    documents: list[dict] = []
    for json_file in sorted(Path(json_dir).glob("*.json")):
        if json_file.name == GUIDES_FILE:
            continue
        with open(json_file, encoding="utf-8") as f:
            file_docs = json.load(f)
        for doc in file_docs:
            doc["guide_name"] = name_map.get(doc["guide"], doc["guide"])
            doc["url"] = guide_url(doc["guide"])
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
