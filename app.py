"""Streamlit chat app for All About Berlin RAG."""

from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from ingest import build_vector_index, load_documents, load_guides, load_meta
from rag_helper import DEFAULT_NUM_RESULTS, RERANK_MODEL, RAGVector

load_dotenv()

logger = logging.getLogger(__name__)

FRIENDLY_ERROR = "Something went wrong while answering. Please try again in a moment."


def friendly_error(exc: Exception) -> str:
    """Log the full traceback server-side; return a neutral message for the UI.

    Never surface the raw exception text — it can leak provider errors or internals.
    """
    logger.error("RAG pipeline failed", exc_info=exc)
    return FRIENDLY_ERROR


@st.cache_resource(show_spinner=False)
def _load_index() -> tuple:
    documents = load_documents()
    vindex, embedder = build_vector_index(documents)
    return vindex, embedder


@st.cache_resource(show_spinner=False)
def _load_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for i, source in enumerate(sources, 1):
        url = source.get("section_url") or source["url"]
        st.markdown(f"{i}. [{source['guide_name']}]({url})")
        if source.get("sections"):
            st.caption(" · ".join(source["sections"][:3]))


def main() -> None:
    st.set_page_config(
        page_title="All About Berlin Assistant",
        page_icon="🐻",
        layout="wide",
    )
    st.title("All About Berlin — Ask Me Anything")
    snapshot = load_meta().get("scraped_at", "")
    snapshot_note = f" · snapshot {snapshot[:10]}" if snapshot else ""
    st.caption(
        f"Answers based on {len(load_guides())} guides from allaboutberlin.com"
        f"{snapshot_note} · Semantic vector search"
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
        st.stop()

    client = OpenAI(api_key=api_key)

    with st.spinner("Loading the search index..."):
        vindex, embedder = _load_index()
        reranker = _load_reranker()

    with st.sidebar:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    if query := st.chat_input("Ask a question about living in Berlin..."):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        pipeline = RAGVector(
            embedder=embedder,
            index=vindex,
            llm_client=client,
            reranker=reranker,
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = pipeline.rag_with_sources(
                        query, history=history, num_results=DEFAULT_NUM_RESULTS
                    )
                    error: str | None = None
                except Exception as e:
                    answer = ""
                    sources = []
                    error = friendly_error(e)

            if error:
                st.error(error)
            else:
                st.markdown(answer)
                _render_sources(sources)

        if not error:
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })


if __name__ == "__main__":
    main()
