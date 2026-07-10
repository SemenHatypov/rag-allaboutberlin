"""Data loading and indexing for allaboutberlin.com guides."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from minsearch import Index

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from minsearch import VectorSearch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "output" / "json"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GUIDES_FILE = "guides.json"
META_FILE = "meta.json"  # snapshot metadata, not a guide document
EMBEDDINGS_FILE = "embeddings.npz"  # precomputed doc vectors (skips cold-start encode)
BASE_GUIDE_URL = "https://allaboutberlin.com/guides"


def guide_url(slug: str) -> str:
    return f"{BASE_GUIDE_URL}/{slug}"


def load_guides(json_dir: Path = DATA_DIR) -> list[dict]:
    guides_path = Path(json_dir) / GUIDES_FILE
    if not guides_path.exists():
        return []
    with open(guides_path, encoding="utf-8") as f:
        return json.load(f)


def load_meta(json_dir: Path = DATA_DIR) -> dict:
    """Snapshot metadata (scraped_at, counts). Empty dict if not present."""
    meta_path = Path(json_dir) / META_FILE
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_documents(json_dir: Path = DATA_DIR) -> list[dict]:
    guides = load_guides(json_dir)
    name_map = {g["guide"]: g["guide_name"] for g in guides}
    url_path_map = {g["guide"]: g.get("url_path") for g in guides}
    documents: list[dict] = []
    for json_file in sorted(Path(json_dir).glob("*.json")):
        if json_file.name in (GUIDES_FILE, META_FILE):
            continue
        with open(json_file, encoding="utf-8") as f:
            file_docs = json.load(f)
        for doc in file_docs:
            doc["guide_name"] = name_map.get(doc["guide"], doc["guide"])
            url_path = url_path_map.get(doc["guide"])
            doc["url"] = f"https://allaboutberlin.com{url_path}" if url_path else guide_url(doc["guide"])
        documents.extend(file_docs)
    return documents


def build_index(documents: list[dict]) -> Index:
    index = Index(
        text_fields=["title", "section", "text"],
        keyword_fields=["guide"],
    )
    index.fit(documents)
    return index


def embed_text(doc: dict) -> str:
    """The text a document is embedded from (section + title + body)."""
    return f"{doc.get('section', '')} {doc.get('title', '')} {doc.get('text', '')}".strip()


def corpus_fingerprint(documents: list[dict]) -> str:
    """Stable hash of the embed texts, in order — identifies a corpus snapshot."""
    joined = "\n".join(embed_text(doc) for doc in documents)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def load_cached_vectors(
    documents: list[dict],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    json_dir: Path = DATA_DIR,
):
    """Return saved doc vectors if they match this corpus + model, else None."""
    import numpy as np

    path = Path(json_dir) / EMBEDDINGS_FILE
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["model"]) != model_name:
            return None
        if str(data["fingerprint"]) != corpus_fingerprint(documents):
            return None
        vectors = data["vectors"]
        if len(vectors) != len(documents):
            return None
        return vectors


def save_embeddings(
    documents: list[dict],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    json_dir: Path = DATA_DIR,
) -> Path:
    """Encode all documents and persist vectors + fingerprint to embeddings.npz."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name)
    texts = [embed_text(doc) for doc in documents]
    vectors = np.asarray(embedder.encode(texts, batch_size=50, show_progress_bar=True), dtype=np.float32)

    path = Path(json_dir) / EMBEDDINGS_FILE
    np.savez(path, vectors=vectors, model=model_name, fingerprint=corpus_fingerprint(documents))
    return path


def build_vector_index(
    documents: list[dict],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    json_dir: Path = DATA_DIR,
) -> tuple[VectorSearch, SentenceTransformer]:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from minsearch import VectorSearch

    embedder = SentenceTransformer(model_name)
    vectors = load_cached_vectors(documents, model_name, json_dir)
    if vectors is None:
        logger.warning("No matching %s cache; encoding %d documents (slow).", EMBEDDINGS_FILE, len(documents))
        texts = [embed_text(doc) for doc in documents]
        vectors = np.asarray(embedder.encode(texts, batch_size=50, show_progress_bar=True), dtype=np.float32)

    vindex: VectorSearch = VectorSearch(keyword_fields=["guide"])
    vindex.fit(vectors, documents)
    return vindex, embedder
