"""Tests for scraper.py — written before confirming implementation correctness."""

import json

import pytest
import requests
import responses as responses_lib

from scraper import (
    BASE_URL,
    GUIDES_PATH,
    GuideEntry,
    GuideSection,
    fetch_html,
    make_id,
    parse_guide_page,
    parse_guides_index,
    run,
    save_json,
    slug_from_url,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal HTML stubs
# ---------------------------------------------------------------------------

GUIDES_INDEX_HTML = """
<html><body>
  <h3>Housing</h3>
  <ul>
    <li><h4><a href="https://allaboutberlin.com/guides/find-a-flat-in-berlin"
               title="How to find an apartment in Berlin">Find a flat</a></h4></li>
    <li><h4><a href="https://allaboutberlin.com/guides/apartment-deposit"
               title="Apartment deposit in Berlin">Apartment deposit</a></h4></li>
  </ul>
  <h3>Bureaucracy</h3>
  <ul>
    <li><h4><a href="https://allaboutberlin.com/guides/anmeldung"
               title="Anmeldung in Berlin">Anmeldung</a></h4></li>
  </ul>
  <!-- link without /guides/ should be ignored -->
  <a href="https://allaboutberlin.com/about">About</a>
</body></html>
"""

GUIDE_PAGE_HTML = """
<html><body>
<article>
  <h2>Finding a flat</h2>
  <p>The most popular platforms are ImmobilienScout24 and WG-Gesucht.</p>
  <h3>Where to search</h3>
  <p>Start with ImmobilienScout24.</p>
  <ul><li>ImmobilienScout24</li><li>WG-Gesucht</li></ul>
  <h2>The application</h2>
  <p>Write a short cover letter.</p>
  <h3>What to include</h3>
  <p>Proof of income, SCHUFA report.</p>
</article>
</body></html>
"""

GUIDE_PAGE_NO_H2_HTML = """
<html><body>
<article>
  <p>Berlin has a tight housing market.</p>
  <h3>Tips</h3>
  <p>Apply quickly.</p>
</article>
</body></html>
"""


# ---------------------------------------------------------------------------
# slug_from_url
# ---------------------------------------------------------------------------

class TestSlugFromUrl:
    def test_absolute_url(self):
        assert slug_from_url("https://allaboutberlin.com/guides/anmeldung") == "anmeldung"

    def test_relative_url(self):
        assert slug_from_url("/guides/find-a-flat-in-berlin") == "find-a-flat-in-berlin"

    def test_trailing_slash(self):
        assert slug_from_url("https://allaboutberlin.com/guides/anmeldung/") == "anmeldung"

    def test_path_traversal_last_segment_rejected(self):
        # ".." as a standalone last segment is rejected by the allowlist
        with pytest.raises(ValueError):
            slug_from_url("https://allaboutberlin.com/guides/..")

    def test_path_traversal_multi_segment_neutralised(self):
        # split("/")[-1] takes only the last component, so ../../etc/passwd
        # reduces to "passwd" — a safe, accepted slug
        assert slug_from_url("https://allaboutberlin.com/guides/../../../etc/passwd") == "passwd"

    def test_uppercase_rejected(self):
        with pytest.raises(ValueError):
            slug_from_url("https://allaboutberlin.com/guides/My-Guide")

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError):
            slug_from_url("https://allaboutberlin.com/guides/evil\x00slug")

    def test_empty_slug_rejected(self):
        # URL with no path segment at all yields an empty string → rejected
        with pytest.raises(ValueError):
            slug_from_url("https://allaboutberlin.com/")

    def test_special_chars_rejected(self):
        with pytest.raises(ValueError):
            slug_from_url("https://allaboutberlin.com/guides/evil;rm -rf")


# ---------------------------------------------------------------------------
# make_id
# ---------------------------------------------------------------------------

class TestMakeId:
    def test_returns_10_chars(self):
        assert len(make_id("slug", "section", "title")) == 10

    def test_hex_string(self):
        result = make_id("slug", "section", "title")
        int(result, 16)  # raises ValueError if not hex

    def test_deterministic(self):
        assert make_id("a", "b", "c") == make_id("a", "b", "c")

    def test_different_inputs_give_different_ids(self):
        assert make_id("a", "b", "c") != make_id("a", "b", "d")


# ---------------------------------------------------------------------------
# parse_guides_index
# ---------------------------------------------------------------------------

class TestParseGuidesIndex:
    def setup_method(self):
        self.entries = parse_guides_index(GUIDES_INDEX_HTML)

    def test_returns_correct_count(self):
        assert len(self.entries) == 3

    def test_entry_type(self):
        for e in self.entries:
            assert isinstance(e, GuideEntry)

    def test_slugs(self):
        slugs = [e.guide for e in self.entries]
        assert "find-a-flat-in-berlin" in slugs
        assert "apartment-deposit" in slugs
        assert "anmeldung" in slugs

    def test_categories(self):
        housing = [e for e in self.entries if e.category == "Housing"]
        bureaucracy = [e for e in self.entries if e.category == "Bureaucracy"]
        assert len(housing) == 2
        assert len(bureaucracy) == 1

    def test_uses_title_attribute_as_guide_name(self):
        anmeldung = next(e for e in self.entries if e.guide == "anmeldung")
        assert anmeldung.guide_name == "Anmeldung in Berlin"

    def test_json_path_format(self):
        for e in self.entries:
            assert e.path == f"/json/{e.guide}.json"

    def test_ignores_non_guide_links(self):
        slugs = [e.guide for e in self.entries]
        assert "about" not in slugs

    def test_sections_count_defaults_to_zero(self):
        for e in self.entries:
            assert e.sections_count == 0


# ---------------------------------------------------------------------------
# parse_guide_page
# ---------------------------------------------------------------------------

class TestParseGuidePage:
    def setup_method(self):
        self.sections = parse_guide_page(GUIDE_PAGE_HTML, "find-a-flat-in-berlin")

    def test_returns_list_of_guide_sections(self):
        for s in self.sections:
            assert isinstance(s, GuideSection)

    def test_guide_slug_set_on_all_sections(self):
        for s in self.sections:
            assert s.guide == "find-a-flat-in-berlin"

    def test_h2_captured_as_section(self):
        sections_set = {s.section for s in self.sections}
        assert "Finding a flat" in sections_set
        assert "The application" in sections_set

    def test_h3_captured_as_title(self):
        titles = {s.title for s in self.sections}
        assert "Where to search" in titles
        assert "What to include" in titles

    def test_text_is_non_empty(self):
        for s in self.sections:
            assert s.text.strip() != ""

    def test_list_items_included_in_text(self):
        where_section = next(
            s for s in self.sections
            if s.title == "Where to search"
        )
        assert "ImmobilienScout24" in where_section.text

    def test_id_is_10_chars(self):
        for s in self.sections:
            assert len(s.id) == 10

    def test_ids_are_unique(self):
        ids = [s.id for s in self.sections]
        assert len(ids) == len(set(ids))

    def test_no_h2_page_uses_empty_section(self):
        sections = parse_guide_page(GUIDE_PAGE_NO_H2_HTML, "housing-tips")
        assert all(s.section == "" for s in sections)

    def test_empty_html_returns_empty_list(self):
        assert parse_guide_page("<html><body></body></html>", "empty") == []


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_creates_file(self, tmp_path):
        entries = [GuideEntry("slug", "Name", "Cat", "/json/slug.json", 3)]
        save_json(entries, tmp_path / "out.json")
        assert (tmp_path / "out.json").exists()

    def test_valid_json(self, tmp_path):
        entries = [GuideEntry("slug", "Name", "Cat", "/json/slug.json", 3)]
        out = tmp_path / "out.json"
        save_json(entries, out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert data[0]["guide"] == "slug"

    def test_creates_parent_dirs(self, tmp_path):
        entries = [GuideSection("abc1234567", "slug", "Sec", "Title", "text")]
        out = tmp_path / "deep" / "nested" / "out.json"
        save_json(entries, out)
        assert out.exists()

    def test_unicode_preserved(self, tmp_path):
        entries = [GuideEntry("slug", "Ärger mit Behörden", "Cat", "/json/slug.json")]
        out = tmp_path / "out.json"
        save_json(entries, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data[0]["guide_name"] == "Ärger mit Behörden"


# ---------------------------------------------------------------------------
# fetch_html
# ---------------------------------------------------------------------------

class TestFetchHtml:
    @responses_lib.activate
    def test_returns_html_body(self):
        responses_lib.add(responses_lib.GET, "https://example.com", body="<html>ok</html>", status=200)
        assert fetch_html("https://example.com") == "<html>ok</html>"

    @responses_lib.activate
    def test_raises_on_http_error(self):
        responses_lib.add(responses_lib.GET, "https://example.com", status=404)
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_html("https://example.com")

    @responses_lib.activate
    def test_sends_user_agent(self):
        responses_lib.add(responses_lib.GET, "https://example.com", body="ok", status=200)
        fetch_html("https://example.com")
        assert "User-Agent" in responses_lib.calls[0].request.headers


# ---------------------------------------------------------------------------
# parse_guides_index edge cases
# ---------------------------------------------------------------------------

class TestParseGuidesIndexEdgeCases:
    def test_h3_with_no_sibling_is_skipped(self):
        html = "<html><body><h3>Empty Section</h3></body></html>"
        entries = parse_guides_index(html)
        assert entries == []

    def test_link_without_guides_prefix_skipped(self):
        html = """
        <html><body>
          <h3>Misc</h3>
          <ul><li><a href="/about">About</a></li></ul>
        </body></html>
        """
        entries = parse_guides_index(html)
        assert entries == []


# ---------------------------------------------------------------------------
# run (integration — all HTTP mocked)
# ---------------------------------------------------------------------------

class TestRun:
    @responses_lib.activate
    def test_creates_guides_json_and_per_guide_files(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}{GUIDES_PATH}",
            body=GUIDES_INDEX_HTML,
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/guides/find-a-flat-in-berlin",
            body=GUIDE_PAGE_HTML,
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/guides/apartment-deposit",
            body=GUIDE_PAGE_HTML,
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/guides/anmeldung",
            body=GUIDE_PAGE_HTML,
            status=200,
        )

        run(output_dir=tmp_path, delay=0)

        assert (tmp_path / "guides.json").exists()
        assert (tmp_path / "find-a-flat-in-berlin.json").exists()
        assert (tmp_path / "anmeldung.json").exists()

    @responses_lib.activate
    def test_guides_json_has_correct_structure(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}{GUIDES_PATH}",
            body=GUIDES_INDEX_HTML,
            status=200,
        )
        for slug in ("find-a-flat-in-berlin", "apartment-deposit", "anmeldung"):
            responses_lib.add(
                responses_lib.GET,
                f"{BASE_URL}/guides/{slug}",
                body=GUIDE_PAGE_HTML,
                status=200,
            )

        run(output_dir=tmp_path, delay=0)

        data = json.loads((tmp_path / "guides.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 3
        first = data[0]
        assert "guide" in first
        assert "guide_name" in first
        assert "category" in first
        assert "path" in first
        assert "sections_count" in first
        assert first["sections_count"] > 0

    @responses_lib.activate
    def test_per_guide_json_has_correct_structure(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}{GUIDES_PATH}",
            body=GUIDES_INDEX_HTML,
            status=200,
        )
        for slug in ("find-a-flat-in-berlin", "apartment-deposit", "anmeldung"):
            responses_lib.add(
                responses_lib.GET,
                f"{BASE_URL}/guides/{slug}",
                body=GUIDE_PAGE_HTML,
                status=200,
            )

        run(output_dir=tmp_path, delay=0)

        data = json.loads((tmp_path / "find-a-flat-in-berlin.json").read_text())
        assert isinstance(data, list)
        assert len(data) > 0
        entry = data[0]
        assert set(entry.keys()) == {"id", "guide", "section", "title", "text"}

    @responses_lib.activate
    def test_continues_on_guide_fetch_error(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}{GUIDES_PATH}",
            body=GUIDES_INDEX_HTML,
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/guides/find-a-flat-in-berlin",
            status=500,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/guides/apartment-deposit",
            body=GUIDE_PAGE_HTML,
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/guides/anmeldung",
            body=GUIDE_PAGE_HTML,
            status=200,
        )

        run(output_dir=tmp_path, delay=0)

        assert (tmp_path / "guides.json").exists()
        assert not (tmp_path / "find-a-flat-in-berlin.json").exists()
        assert (tmp_path / "anmeldung.json").exists()

        index = json.loads((tmp_path / "guides.json").read_text())
        failed = next(e for e in index if e["guide"] == "find-a-flat-in-berlin")
        assert failed["sections_count"] == 0
