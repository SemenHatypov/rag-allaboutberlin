"""RAG pipeline for allaboutberlin.com guides."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from minsearch import Index
from openai import OpenAI

from ingest import build_index, build_vector_index, embed_text, guide_url, load_documents

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"

# Top-k search results. Chosen from a k-sweep over the ground truth (see README,
# "Search Evaluation"): the smallest k where vector-search hit_rate/MRR plateau
# (k=12: hit_rate 0.982, MRR 0.829; further k gains < 0.005).
DEFAULT_NUM_RESULTS = 12

# Max guides shown to the user as sources. Retrieval still uses all top-k chunks;
# this only trims the citation list so the tail of loosely-related guides doesn't
# dilute trust. Kept separate from DEFAULT_NUM_RESULTS on purpose.
MAX_SOURCES = 3

# Relevance gate for the CITATION list only (never touches the LLM context — it
# still sees all top-k chunks, so answer quality is unchanged). Gates cited guides
# by their best cross-encoder rerank score. The rank-1 guide is always kept; a
# lower-ranked guide is cited only when its best chunk scores at least
# RERANK_SCORE_FLOOR *and* within RERANK_SCORE_MARGIN of the top guide's score.
#
# Values calibrated on real rerank scores (see the 2026-07 source-relevance work):
# ms-marco-MiniLM emits raw logits (0 ≈ 50% relevance). At FLOOR=0/MARGIN=3 the
# single-guide questions (Deutschlandticket, Schufa, Abmeldung) collapse to one
# authoritative source while genuinely multi-guide questions (freelance visa,
# insurance) keep their 2-3 closely-scored guides. Only applied when scores are
# present (i.e. a reranker ran); the keyword path and retrieval eval are untouched.
RERANK_SCORE_FLOOR = 0.0
RERANK_SCORE_MARGIN = 3.0

# Most recent conversation messages fed to the LLM for follow-up context
# (3 exchanges). Older turns are dropped to keep the prompt small.
MAX_HISTORY_MESSAGES = 6

# Cross-encoder that re-scores the top-k bi-encoder hits by true query-passage
# relevance. The MiniLM bi-encoder alone sometimes ranks the wrong section first
# (e.g. Anmeldung "Questions and answers" over "Prepare your documents").
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Exact fallback string. Referenced by INSTRUCTIONS (so the model is told to emit
# it verbatim) and by is_refusal (so the app can detect a refusal). Keep in sync.
NO_ANSWER = "I don't have information about this in my guides."

INSTRUCTIONS = f"""
You are a helpful assistant for expats living in or moving to Berlin, Germany.
Answer the question using ONLY the provided context from allaboutberlin.com guides.

These guides are specific to Berlin, Germany. If the question is about a different
city or country, say up front that your guides only cover Berlin, and only apply
facts from the context that hold for all of Germany.

The context may contain guides written for different situations (for example
employees, freelancers, or students). Use only the parts that match the person's
situation in the question — never mix rules from different audiences.

Give a short, summary-style answer: 2-4 sentences with the key facts and the
practical next step. Do not list the source guides, guide names, or guide links in
your answer — they are shown to the user separately. You MAY name one specific
official website or tool mentioned in the context (for example a government portal)
when it is the practical next step. If the answer is not found in the context,
respond with "{NO_ANSWER}"
""".strip()

# Rewrites a follow-up into a standalone search query so retrieval isn't derailed
# by pronouns/ellipsis ("speed it up", "how much for it").
CONDENSE_INSTRUCTIONS = """
You rewrite a follow-up question into a standalone search query.

Given the conversation so far and a follow-up question, resolve any pronouns and
ellipsis using the conversation, and return a single self-contained question that
makes sense on its own. If the follow-up is already self-contained, return it
unchanged. Return ONLY the rewritten question — no preamble, no quotes.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


def is_refusal(answer: str) -> bool:
    """True when the model emitted the no-information fallback."""
    return NO_ANSWER.lower() in answer.strip().lower()


def format_history(history: list[dict]) -> str:
    """Render prior turns as a plain transcript for the condensation prompt."""
    lines: list[str] = []
    for msg in history:
        speaker = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {msg.get('content', '')}")
    return "\n".join(lines).strip()


# ── Sources ────────────────────────────────────────────────────────────────────


def _passes_relevance_gate(sources: list[dict], min_score: float | None) -> list[dict]:
    """Drop weakly-relevant cited guides using their best rerank score.

    Keeps the rank-1 guide always (a non-refusal answer should show at least one
    source). A later guide survives only when its best chunk scores at least
    ``min_score`` AND within RERANK_SCORE_MARGIN of the top guide's score.

    A no-op when gating is off (``min_score is None``) or when scores are absent
    (no reranker ran, so no guide carries "_best_score") — the keyword path and the
    retrieval eval keep their original behaviour.
    """
    if min_score is None or not sources:
        return sources
    if not any("_best_score" in s for s in sources):
        return sources
    top_score = sources[0]["_best_score"]
    kept = [sources[0]]
    for source in sources[1:]:
        score = source.get("_best_score")
        if score is None:
            continue
        if score >= min_score and (top_score - score) <= RERANK_SCORE_MARGIN:
            kept.append(source)
    return kept


def extract_sources(
    search_results: list[dict],
    limit: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """Unique guides from ranked search results, preserving rank order.

    Each source: {"guide", "guide_name", "url", "section_url", "sections": [str, ...]}
    where sections are the non-empty section headings retrieved for that guide and
    section_url deep-links to the top-ranked retrieved section (falls back to the
    guide URL when the chunk has no anchor).

    limit=None (default) keeps every cited guide — required by the retrieval eval,
    which scores hit_rate/MRR over all cited guides. The app passes limit=MAX_SOURCES
    to trim the citation list shown to the user.

    min_score gates the citation list by rerank relevance (see _passes_relevance_gate
    and RERANK_SCORE_FLOOR). None (default) disables gating; it is also a no-op when
    the results carry no rerank scores.
    """
    by_guide: dict[str, dict] = {}
    for doc in search_results:
        slug = doc["guide"]
        source = by_guide.get(slug)
        if source is None:
            url = doc.get("url", guide_url(slug))
            anchor = doc.get("anchor", "")
            source = {
                "guide": slug,
                "guide_name": doc.get("guide_name", slug),
                "url": url,
                # Deep-link to the first (highest-ranked) retrieved section.
                "section_url": f"{url}#{anchor}" if anchor else url,
                "sections": [],
            }
            if "rerank_score" in doc:
                # Best (first-seen == highest, results are score-sorted) chunk score.
                source["_best_score"] = doc["rerank_score"]
            by_guide[slug] = source
        section = doc.get("section", "")
        if section and section not in source["sections"]:
            source["sections"].append(section)
    sources = list(by_guide.values())
    sources = _passes_relevance_gate(sources, min_score)
    if limit is not None:
        sources = sources[:limit]
    # "_best_score" is an internal ranking aid, not part of the source contract.
    for source in sources:
        source.pop("_best_score", None)
    return sources


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
        reranker=None,
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.guide = guide  # None = search across all guides
        self.model = model
        self.reranker = reranker  # optional CrossEncoder; None = keep bi-encoder order

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> list[dict]:
        boost_dict = {"title": 2.0, "section": 0.5}
        filter_dict = {"guide": self.guide} if self.guide else {}
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Reorder hits by cross-encoder relevance, tagging each with its score.

        No reranker → unchanged order and no score attached. Scores are kept on a
        new "rerank_score" key (fresh dicts, inputs untouched) so downstream code
        can gate citations by relevance; see extract_sources.
        """
        if self.reranker is None or not results:
            return results
        pairs = [(query, embed_text(doc)) for doc in results]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(results, scores), key=lambda rs: float(rs[1]), reverse=True)
        return [{**doc, "rerank_score": float(score)} for doc, score in ranked]

    def retrieve(
        self,
        query: str,
        history: list[dict] | None = None,
        num_results: int = DEFAULT_NUM_RESULTS,
    ) -> list[dict]:
        """Condense the query against history, search, then rerank the hits."""
        search_query = self.condense_query(query, history or [])
        results = self.search(search_query, num_results=num_results)
        return self.rerank(search_query, results)

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

    def condense_query(self, query: str, history: list[dict]) -> str:
        """Rewrite a follow-up into a standalone search query using recent history.

        No history → return the query unchanged (no LLM call). On an empty or failed
        rewrite, fall back to the raw query so retrieval always has something to run.
        """
        if not history:
            return query
        transcript = format_history(history[-MAX_HISTORY_MESSAGES:])
        user_prompt = f"Conversation:\n{transcript}\n\nFollow-up: {query}\n\nStandalone question:"
        try:
            response = self.llm_client.responses.create(
                model=self.model,
                input=[
                    {"role": "developer", "content": CONDENSE_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
            )
            rewritten = (response.output_text or "").strip()
        except Exception:
            return query
        return rewritten or query

    def _build_messages(self, prompt: str, history: list[dict] | None = None) -> list[dict]:
        messages: list[dict] = [{"role": "developer", "content": self.instructions}]
        for msg in (history or [])[-MAX_HISTORY_MESSAGES:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    def llm(self, prompt: str, history: list[dict] | None = None) -> str:
        response = self.llm_client.responses.create(
            model=self.model,
            input=self._build_messages(prompt, history),
        )
        return response.output_text

    def llm_stream(self, prompt: str, history: list[dict] | None = None):
        """Yield answer text deltas as they arrive (for a streamed UI)."""
        stream = self.llm_client.responses.create(
            model=self.model,
            input=self._build_messages(prompt, history),
            stream=True,
        )
        for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                yield event.delta

    def rag_with_sources(
        self,
        query: str,
        history: list[dict] | None = None,
        num_results: int = DEFAULT_NUM_RESULTS,
    ) -> tuple[str, list[dict]]:
        history = history or []
        results = self.retrieve(query, history, num_results=num_results)
        prompt = self.build_prompt(query, results)
        answer = self.llm(prompt, history=history)
        if is_refusal(answer):
            return answer, []
        return answer, extract_sources(results, limit=MAX_SOURCES, min_score=RERANK_SCORE_FLOOR)

    def rag(self, query: str) -> str:
        return self.rag_with_sources(query)[0]


# ── Vector RAG class ───────────────────────────────────────────────────────────


class RAGVector(RAGBase):

    def __init__(self, embedder, index, llm_client: OpenAI, **kwargs) -> None:
        super().__init__(index, llm_client, **kwargs)
        self.embedder = embedder

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> list[dict]:
        query_vector = self.embedder.encode(query)
        filter_dict = {"guide": self.guide} if self.guide else {}
        return self.index.search(query_vector, num_results=num_results, filter_dict=filter_dict)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question about Berlin expat guides")
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--guide", "-g", default=None, help="Filter to a specific guide slug")
    parser.add_argument(
        "--num-results", "-n",
        type=int,
        default=DEFAULT_NUM_RESULTS,
        help="Number of search results to use",
    )
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
