"""Tests for ingest.py — document loading and guide metadata enrichment."""

import json

from ingest import guide_url, load_documents, load_guides

GUIDES_INDEX = [
    {
        "guide": "anmeldung",
        "guide_name": "How to register your address",
        "category": "Bureaucracy",
        "path": "/json/anmeldung.json",
        "sections_count": 2,
    },
    {
        "guide": "empty-guide",
        "guide_name": "Guide without documents",
        "category": "Misc",
        "path": "/json/empty-guide.json",
        "sections_count": 0,
    },
    {
        "guide": "for-employees",
        "guide_name": "German health insurance for employees",
        "category": "Personal finance",
        "path": "/json/for-employees.json",
        "sections_count": 1,
        "url_path": "/guides/german-health-insurance/for-employees",
    },
]

NESTED_DOCS = [
    {
        "id": "dddddddddd",
        "guide": "for-employees",
        "section": "Overview",
        "title": "",
        "text": "Employees get statutory health insurance by default.",
    },
]

ANMELDUNG_DOCS = [
    {
        "id": "aaaaaaaaaa",
        "guide": "anmeldung",
        "section": "How to register",
        "title": "",
        "text": "Book an appointment at the Buergeramt.",
    },
    {
        "id": "bbbbbbbbbb",
        "guide": "anmeldung",
        "section": "Documents",
        "title": "Passport",
        "text": "Bring your passport.",
    },
]

ORPHAN_DOCS = [
    {
        "id": "cccccccccc",
        "guide": "orphan",
        "section": "Intro",
        "title": "",
        "text": "This guide is missing from guides.json.",
    },
]


def write_corpus(json_dir):
    (json_dir / "guides.json").write_text(json.dumps(GUIDES_INDEX), encoding="utf-8")
    (json_dir / "anmeldung.json").write_text(json.dumps(ANMELDUNG_DOCS), encoding="utf-8")
    (json_dir / "orphan.json").write_text(json.dumps(ORPHAN_DOCS), encoding="utf-8")
    (json_dir / "for-employees.json").write_text(json.dumps(NESTED_DOCS), encoding="utf-8")


class TestGuideUrl:
    def test_builds_canonical_url(self):
        assert guide_url("anmeldung") == "https://allaboutberlin.com/guides/anmeldung"


class TestLoadGuides:
    def test_loads_index(self, tmp_path):
        write_corpus(tmp_path)
        guides = load_guides(tmp_path)
        assert [g["guide"] for g in guides] == ["anmeldung", "empty-guide", "for-employees"]

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert load_guides(tmp_path) == []


class TestLoadDocuments:
    def test_excludes_guides_index_from_documents(self, tmp_path):
        write_corpus(tmp_path)
        documents = load_documents(tmp_path)
        assert len(documents) == 4
        assert not any("sections_count" in doc for doc in documents)

    def test_enriches_with_guide_name_and_url(self, tmp_path):
        write_corpus(tmp_path)
        documents = load_documents(tmp_path)
        doc = next(d for d in documents if d["guide"] == "anmeldung")
        assert doc["guide_name"] == "How to register your address"
        assert doc["url"] == "https://allaboutberlin.com/guides/anmeldung"

    def test_missing_guide_name_falls_back_to_slug(self, tmp_path):
        write_corpus(tmp_path)
        documents = load_documents(tmp_path)
        orphan = next(d for d in documents if d["guide"] == "orphan")
        assert orphan["guide_name"] == "orphan"
        assert orphan["url"] == "https://allaboutberlin.com/guides/orphan"

    def test_uses_nested_url_path_from_manifest_when_present(self, tmp_path):
        write_corpus(tmp_path)
        documents = load_documents(tmp_path)
        doc = next(d for d in documents if d["guide"] == "for-employees")
        assert doc["url"] == "https://allaboutberlin.com/guides/german-health-insurance/for-employees"

    def test_works_without_guides_index(self, tmp_path):
        (tmp_path / "anmeldung.json").write_text(json.dumps(ANMELDUNG_DOCS), encoding="utf-8")
        documents = load_documents(tmp_path)
        assert len(documents) == 2
        assert documents[0]["guide_name"] == "anmeldung"
