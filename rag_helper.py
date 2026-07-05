"""RAG pipeline for allaboutberlin.com guides."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from minsearch import Index
from openai import OpenAI

from ingest import build_index, build_vector_index, guide_url, load_documents

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"

INSTRUCTIONS = """
You are a helpful assistant for expats living in or moving to Berlin, Germany.
Answer the question using ONLY the provided context from allaboutberlin.com guides.

Give a short, summary-style answer: 2-4 sentences with the key facts and the
practical next step. Do not list sources, guide names, or links in your answer —
they are shown to the user separately. If the answer is not found in the context,
respond with "I don't have information about this in my guides."
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


# ── Sources ────────────────────────────────────────────────────────────────────


def extract_sources(search_results: list[dict]) -> list[dict]:
    """Unique guides from ranked search results, preserving rank order.

    Each source: {"guide", "guide_name", "url", "sections": [str, ...]}
    where sections are the non-empty section headings retrieved for that guide.
    """
    by_guide: dict[str, dict] = {}
    for doc in search_results:
        slug = doc["guide"]
        source = by_guide.get(slug)
        if source is None:
            source = {
                "guide": slug,
                "guide_name": doc.get("guide_name", slug),
                "url": doc.get("url", guide_url(slug)),
                "sections": [],
            }
            by_guide[slug] = source
        section = doc.get("section", "")
        if section and section not in source["sections"]:
            source["sections"].append(section)
    return list(by_guide.values())


# ── RAG class ──────────────────────────────────────────────────────────────────


class RAGBase:

    def __init__(
        self,
        index: Index,
        llm_client: OpenAI,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        guide: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.guide = guide  # None = search across all guides
        self.model = model

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        boost_dict = {"title": 2.0, "section": 0.5}
        filter_dict = {"guide": self.guide} if self.guide else {}
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results: list[dict]) -> str:
        lines: list[str] = []
        for doc in search_results:
            lines.append(f"Guide: {doc.get('guide_name', doc['guide'])}")
            lines.append(f"Section: {doc['section']}")
            if doc.get("title"):
                lines.append(f"Title: {doc['title']}")
            lines.append(f"Content: {doc['text']}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list[dict]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )
        return response.output_text

    def rag_with_sources(self, query: str, num_results: int = 5) -> tuple[str, list[dict]]:
        search_results = self.search(query, num_results=num_results)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt), extract_sources(search_results)

    def rag(self, query: str) -> str:
        return self.rag_with_sources(query)[0]


# ── Vector RAG class ───────────────────────────────────────────────────────────


class RAGVector(RAGBase):

    def __init__(self, embedder, index, llm_client: OpenAI, **kwargs) -> None:
        super().__init__(index, llm_client, **kwargs)
        self.embedder = embedder

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        query_vector = self.embedder.encode(query)
        filter_dict = {"guide": self.guide} if self.guide else {}
        return self.index.search(query_vector, num_results=num_results, filter_dict=filter_dict)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question about Berlin expat guides")
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--guide", "-g", default=None, help="Filter to a specific guide slug")
    parser.add_argument("--num-results", "-n", type=int, default=5, help="Number of search results to use")
    parser.add_argument(
        "--search-type", "-s",
        choices=["keyword", "vector"],
        default="keyword",
        help="Search backend: keyword (BM25) or vector (semantic)",
    )
    args = parser.parse_args()

    print("Loading documents...")
    docs = load_documents()

    if args.search_type == "vector":
        print("Building vector index (encoding documents)...")
        idx, embedder = build_vector_index(docs)
        print(f"Indexed {len(docs)} document sections.\n")
        pipeline = RAGVector(
            embedder=embedder,
            index=idx,
            llm_client=OpenAI(),
            guide=args.guide,
            model=args.model,
        )
    else:
        print("Building keyword index...")
        idx = build_index(docs)
        print(f"Indexed {len(docs)} document sections.\n")
        pipeline = RAGBase(
            index=idx,
            llm_client=OpenAI(),
            guide=args.guide,
            model=args.model,
        )

    answer, sources = pipeline.rag_with_sources(args.question, num_results=args.num_results)
    print(answer)
    print("\nSources:")
    for source in sources:
        print(f"- {source['guide_name']} — {source['url']}")
