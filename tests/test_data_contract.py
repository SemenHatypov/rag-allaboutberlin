"""Contract test: the real committed corpus satisfies what the app depends on.

Unlike the other tests (which use fabricated fixtures), this one loads the
actual data from output/json/ and asserts the fields that app.py, rag_helper
and the eval scripts rely on. It catches data regressions — e.g. a re-scrape
that drops fields, or a broken guides.json — before they reach the deployed app.
"""

from ingest import DATA_DIR, load_documents, load_guides

REQUIRED_DOC_FIELDS = ("id", "guide", "guide_name", "url", "section", "title", "text")


class TestRealCorpusContract:
    def test_corpus_is_present_and_nonempty(self):
        assert DATA_DIR.exists(), "output/json is tracked in git and must exist"
        assert len(load_documents()) > 1000

    def test_every_document_has_required_fields(self):
        documents = load_documents()
        for doc in documents:
            missing = [f for f in REQUIRED_DOC_FIELDS if f not in doc]
            assert not missing, f"doc {doc.get('id')} in guide {doc.get('guide')} missing {missing}"

    def test_no_index_records_leak_into_documents(self):
        assert not any("sections_count" in doc for doc in load_documents())

    def test_urls_point_to_allaboutberlin_guides(self):
        documents = load_documents()
        assert all(doc["url"] == f"https://allaboutberlin.com/guides/{doc['guide']}" for doc in documents)

    def test_guides_index_covers_all_document_slugs(self):
        known = {g["guide"] for g in load_guides()}
        doc_slugs = {doc["guide"] for doc in load_documents()}
        assert doc_slugs <= known, f"slugs missing from guides.json: {doc_slugs - known}"
