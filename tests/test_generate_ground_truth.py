"""Tests for generate_ground_truth.py — guide grouping and text digests."""

from generate_ground_truth import build_guide_text, group_by_guide, make_guide_record


def make_doc(guide="anmeldung", section="How to register", title="", text="Some text."):
    return {
        "id": "aaaaaaaaaa",
        "guide": guide,
        "guide_name": "How to register your address",
        "url": f"https://allaboutberlin.com/guides/{guide}",
        "section": section,
        "title": title,
        "text": text,
    }


class TestGroupByGuide:
    def test_groups_by_slug(self):
        docs = [make_doc(guide="anmeldung"), make_doc(guide="schufa"), make_doc(guide="anmeldung")]
        groups = group_by_guide(docs)
        assert set(groups) == {"anmeldung", "schufa"}
        assert len(groups["anmeldung"]) == 2

    def test_drops_docs_without_text(self):
        docs = [make_doc(text=""), make_doc(text="Real content.")]
        groups = group_by_guide(docs)
        assert len(groups["anmeldung"]) == 1


class TestBuildGuideText:
    def test_formats_section_and_title_headings(self):
        docs = [make_doc(section="Documents", title="Passport", text="Bring it.")]
        assert build_guide_text(docs) == "## Documents — Passport\nBring it."

    def test_omits_heading_when_empty(self):
        docs = [make_doc(section="", title="", text="Intro text.")]
        assert build_guide_text(docs) == "Intro text."

    def test_truncates_each_section(self):
        docs = [make_doc(text="x" * 2000)]
        text = build_guide_text(docs, per_section_chars=100)
        assert len(text) < 200

    def test_respects_total_max_chars(self):
        docs = [make_doc(section=f"S{i}", text="x" * 800) for i in range(10)]
        text = build_guide_text(docs, max_chars=2000, per_section_chars=800)
        assert len(text) <= 2000


class TestMakeGuideRecord:
    def test_record_shape(self):
        record = make_guide_record(make_doc(), "How do I register?")
        assert record == {
            "question": "How do I register?",
            "guide": "anmeldung",
            "guide_name": "How to register your address",
            "url": "https://allaboutberlin.com/guides/anmeldung",
        }
