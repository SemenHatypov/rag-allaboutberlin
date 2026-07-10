"""Tests for rag_helper.py — source extraction and the RAG pipeline plumbing."""

from rag_helper import (
    DEFAULT_NUM_RESULTS,
    MAX_SOURCES,
    NO_ANSWER,
    RAGBase,
    RAGVector,
    extract_sources,
    format_history,
    is_refusal,
)

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


class RecordingResponses:
    """Records every create() input; returns scripted output_text in order."""

    def __init__(self, outputs=None):
        self.outputs = outputs or ["A short answer."]
        self.inputs = []
        self._i = 0

    def create(self, model, input):
        self.inputs.append(input)
        text = self.outputs[min(self._i, len(self.outputs) - 1)]
        self._i += 1

        class Response:
            output_text = text

        return Response()


class RecordingClient:
    def __init__(self, outputs=None):
        self.responses = RecordingResponses(outputs)


class RaisingResponses:
    def create(self, model, input):
        raise RuntimeError("boom")


class RaisingClient:
    def __init__(self):
        self.responses = RaisingResponses()


MANY_GUIDES = [
    {"guide": f"g{i}", "guide_name": f"Guide {i}",
     "url": f"https://allaboutberlin.com/guides/g{i}", "section": f"S{i}", "text": "..."}
    for i in range(5)
]

ANCHORED_DOCS = [
    {"guide": "anmeldung", "guide_name": "Anmeldung",
     "url": "https://allaboutberlin.com/guides/anmeldung",
     "anchor": "prepare-your-documents", "section": "Documents", "text": "Bring your passport."},
    {"guide": "anmeldung", "guide_name": "Anmeldung",
     "url": "https://allaboutberlin.com/guides/anmeldung",
     "anchor": "go-to-appointment", "section": "Appointment", "text": "Go in person."},
]


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
                "section_url": "https://allaboutberlin.com/guides/taxes",
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

    def test_defaults_to_fixed_num_results(self):
        index = StubIndex(DOCS)
        pipeline = RAGBase(index=index, llm_client=StubClient())
        pipeline.rag_with_sources("How do I register?")
        assert index.calls[0]["num_results"] == DEFAULT_NUM_RESULTS

    def test_vector_pipeline_encodes_query(self):
        index = StubIndex(DOCS)
        pipeline = RAGVector(embedder=StubEmbedder(), index=index, llm_client=StubClient())
        answer, sources = pipeline.rag_with_sources("free schufa", num_results=2)
        assert answer == "A short answer."
        assert index.calls[0]["query"] == [0.0, 1.0]

    def test_sources_capped_at_max_sources(self):
        index = StubIndex(MANY_GUIDES)
        pipeline = RAGBase(index=index, llm_client=StubClient())
        _, sources = pipeline.rag_with_sources("anything", num_results=5)
        assert len(sources) == MAX_SOURCES

    def test_refusal_suppresses_sources(self):
        index = StubIndex(DOCS)
        pipeline = RAGBase(index=index, llm_client=RecordingClient(outputs=[NO_ANSWER]))
        answer, sources = pipeline.rag_with_sources("who won the world cup?")
        assert answer == NO_ANSWER
        assert sources == []


class TestIsRefusal:
    def test_exact_match(self):
        assert is_refusal(NO_ANSWER)

    def test_surrounding_whitespace(self):
        assert is_refusal(f"  {NO_ANSWER}  ")

    def test_case_insensitive(self):
        assert is_refusal(NO_ANSWER.upper())

    def test_normal_answer_is_not_refusal(self):
        assert not is_refusal("Book an appointment at the Bürgeramt.")


class TestFormatHistory:
    def test_empty_history(self):
        assert format_history([]) == ""

    def test_renders_speakers(self):
        history = [
            {"role": "user", "content": "How do I register?"},
            {"role": "assistant", "content": "Book an appointment."},
        ]
        assert format_history(history) == (
            "User: How do I register?\nAssistant: Book an appointment."
        )


class TestExtractSourcesLimit:
    def test_default_returns_all_guides(self):
        assert len(extract_sources(MANY_GUIDES)) == 5

    def test_limit_caps_count(self):
        assert len(extract_sources(MANY_GUIDES, limit=2)) == 2

    def test_limit_none_is_explicit_all(self):
        assert len(extract_sources(MANY_GUIDES, limit=None)) == 5


class TestSectionUrl:
    def test_appends_anchor_of_top_ranked_chunk(self):
        sources = extract_sources(ANCHORED_DOCS)
        assert sources[0]["section_url"] == (
            "https://allaboutberlin.com/guides/anmeldung#prepare-your-documents"
        )

    def test_falls_back_to_guide_url_without_anchor(self):
        sources = extract_sources(DOCS)
        assert sources[0]["section_url"] == sources[0]["url"]


class TestCondenseQuery:
    def test_no_history_returns_query_unchanged_without_llm_call(self):
        client = RecordingClient()
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=client)
        assert pipeline.condense_query("How much does it cost?", []) == "How much does it cost?"
        assert client.responses.inputs == []

    def test_rewrites_with_history(self):
        client = RecordingClient(outputs=["How much does the freelance visa cost?"])
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=client)
        history = [{"role": "user", "content": "Which visa to freelance?"}]
        rewritten = pipeline.condense_query("How much for it?", history)
        assert rewritten == "How much does the freelance visa cost?"
        # the transcript and follow-up are both in the condense prompt
        sent = client.responses.inputs[0][-1]["content"]
        assert "Which visa to freelance?" in sent
        assert "How much for it?" in sent

    def test_empty_rewrite_falls_back_to_query(self):
        client = RecordingClient(outputs=["   "])
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=client)
        history = [{"role": "user", "content": "prior"}]
        assert pipeline.condense_query("original", history) == "original"

    def test_llm_error_falls_back_to_query(self):
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=RaisingClient())
        history = [{"role": "user", "content": "prior"}]
        assert pipeline.condense_query("original", history) == "original"


class TestLlmHistory:
    def test_history_messages_included_in_input(self):
        client = RecordingClient()
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=client)
        history = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "answer 1"},
        ]
        pipeline.llm("current prompt", history=history)
        sent = client.responses.inputs[0]
        roles_contents = [(m["role"], m["content"]) for m in sent]
        assert roles_contents[0] == ("developer", pipeline.instructions)
        assert ("user", "turn 1") in roles_contents
        assert ("assistant", "answer 1") in roles_contents
        assert roles_contents[-1] == ("user", "current prompt")

    def test_no_history_is_just_instructions_and_prompt(self):
        client = RecordingClient()
        pipeline = RAGBase(index=StubIndex(DOCS), llm_client=client)
        pipeline.llm("current prompt")
        sent = client.responses.inputs[0]
        assert len(sent) == 2
        assert sent[0]["role"] == "developer"
        assert sent[1] == {"role": "user", "content": "current prompt"}


class TestMultiTurnPlumbing:
    def test_history_drives_condensation_then_answer(self):
        # 1st create() = condense (returns standalone query), 2nd = answer
        client = RecordingClient(outputs=["standalone rewritten query", "Final answer."])
        index = StubIndex(DOCS)
        pipeline = RAGBase(index=index, llm_client=client)
        history = [{"role": "user", "content": "earlier question"}]
        answer, _ = pipeline.rag_with_sources("follow up", history=history)
        assert answer == "Final answer."
        # retrieval used the rewritten query, not the raw follow-up
        assert index.calls[0]["query"] == "standalone rewritten query"
        # the answer call carried the conversation history
        answer_input = client.responses.inputs[1]
        assert any(m["content"] == "earlier question" for m in answer_input)
