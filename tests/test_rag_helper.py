"""Tests for rag_helper.py — source extraction and the RAG pipeline plumbing."""

from rag_helper import RAGBase, RAGVector, extract_sources

DOCS = [
    {
        "id": "aaaaaaaaaa",
        "guide": "anmeldung",
        "guide_name": "How to register your address",
        "url": "https://allaboutberlin.com/guides/anmeldung",
        "section": "How to register",
        "title": "",
        "text": "Book an appointment.",
    },
    {
        "id": "bbbbbbbbbb",
        "guide": "schufa",
        "guide_name": "How to get a free Schufa",
        "url": "https://allaboutberlin.com/guides/schufa",
        "section": "Free copy",
        "title": "",
        "text": "Order it online.",
    },
    {
        "id": "cccccccccc",
        "guide": "anmeldung",
        "guide_name": "How to register your address",
        "url": "https://allaboutberlin.com/guides/anmeldung",
        "section": "Documents",
        "title": "Passport",
        "text": "Bring your passport.",
    },
]


class StubIndex:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, num_results=5, boost_dict=None, filter_dict=None):
        self.calls.append({"query": query, "num_results": num_results, "filter_dict": filter_dict})
        return self.results[:num_results]


class StubResponses:
    def create(self, model, input):
        self.last_input = input

        class Response:
            output_text = "A short answer."

        return Response()


class StubClient:
    def __init__(self):
        self.responses = StubResponses()


class StubEmbedder:
    def encode(self, text):
        return [0.0, 1.0]


class TestExtractSources:
    def test_dedupes_guides_preserving_rank_order(self):
        sources = extract_sources(DOCS)
        assert [s["guide"] for s in sources] == ["anmeldung", "schufa"]

    def test_aggregates_sections_per_guide(self):
        sources = extract_sources(DOCS)
        anmeldung = sources[0]
        assert anmeldung["sections"] == ["How to register", "Documents"]

    def test_includes_name_and_url(self):
        sources = extract_sources(DOCS)
        assert sources[1]["guide_name"] == "How to get a free Schufa"
        assert sources[1]["url"] == "https://allaboutberlin.com/guides/schufa"

    def test_falls_back_to_slug_and_built_url(self):
        bare_doc = {"guide": "taxes", "section": "", "title": "", "text": "..."}
        sources = extract_sources([bare_doc])
        assert sources == [
            {
                "guide": "taxes",
                "guide_name": "taxes",
                "url": "https://allaboutberlin.com/guides/taxes",
                "sections": [],
            }
        ]

    def test_skips_empty_and_duplicate_sections(self):
        docs = [
            {"guide": "taxes", "section": "", "text": "..."},
            {"guide": "taxes", "section": "VAT", "text": "..."},
            {"guide": "taxes", "section": "VAT", "text": "..."},
        ]
        assert extract_sources(docs)[0]["sections"] == ["VAT"]

    def test_empty_results(self):
        assert extract_sources([]) == []


class TestBuildContext:
    def test_uses_guide_name_when_present(self):
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=StubClient())
        context = pipeline.build_context(DOCS[:1])
        assert "Guide: How to register your address" in context

    def test_falls_back_to_slug(self):
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=StubClient())
        context = pipeline.build_context([{"guide": "taxes", "section": "VAT", "text": "..."}])
        assert "Guide: taxes" in context


class TestRagWithSources:
    def test_returns_answer_and_deduped_sources(self):
        index = StubIndex(DOCS)
        pipeline = RAGBase(index=index, llm_client=StubClient())
        answer, sources = pipeline.rag_with_sources("How do I register?", num_results=3)
        assert answer == "A short answer."
        assert [s["guide"] for s in sources] == ["anmeldung", "schufa"]
        assert index.calls[0]["num_results"] == 3

    def test_rag_returns_answer_only(self):
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=StubClient())
        assert pipeline.rag("How do I register?") == "A short answer."

    def test_vector_pipeline_encodes_query(self):
        index = StubIndex(DOCS)
        pipeline = RAGVector(embedder=StubEmbedder(), index=index, llm_client=StubClient())
        answer, sources = pipeline.rag_with_sources("free schufa", num_results=2)
        assert answer == "A short answer."
        assert index.calls[0]["query"] == [0.0, 1.0]
