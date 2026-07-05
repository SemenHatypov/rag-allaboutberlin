"""Streamlit chat app for All About Berlin RAG."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from ingest import build_vector_index, load_documents
from rag_helper import RAGVector

load_dotenv()


@st.cache_resource(show_spinner=False)
def _load_index() -> tuple:
    documents = load_documents()
    vindex, embedder = build_vector_index(documents)
    name_to_slug = {doc["guide_name"]: doc["guide"] for doc in documents}
    return vindex, embedder, name_to_slug


def _render_sources(sources: list[dict]) -> None:
    st.markdown("**Sources**")
    for i, source in enumerate(sources, 1):
        st.markdown(f"{i}. [{source['guide_name']}]({source['url']})")
        if source.get("sections"):
            st.caption(" · ".join(source["sections"][:3]))


def main() -> None:
    st.set_page_config(
        page_title="All About Berlin Assistant",
        page_icon="🐻",
        layout="wide",
    )
    st.title("All About Berlin — Ask Me Anything")
    st.caption("Answers based on 142 guides from allaboutberlin.com · Semantic vector search")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
        st.stop()

    client = OpenAI(api_key=api_key)

    with st.spinner("Building vector index (first load ~1 min)..."):
        vindex, embedder, name_to_slug = _load_index()

    with st.sidebar:
        st.header("Settings")
        guide_options = ["All guides"] + sorted(name_to_slug)
        selected = st.selectbox("Filter by guide", guide_options)
        guide_filter: str | None = None if selected == "All guides" else name_to_slug[selected]
        num_results = st.slider("Number of results", min_value=3, max_value=15, value=5)
        st.divider()
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
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        pipeline = RAGVector(
            embedder=embedder,
            index=vindex,
            llm_client=client,
            guide=guide_filter,
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = pipeline.rag_with_sources(query, num_results=num_results)
                    error: str | None = None
                except Exception as e:
                    answer = ""
                    sources = []
                    error = str(e)

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
