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

NESTED_GUIDES_INDEX_HTML = """
<html><body>
  <h3>Personal Finance</h3>
  <ul>
    <li><h4><a href="https://allaboutberlin.com/guides/german-health-insurance/for-employees"
               title="German health insurance for employees">For employees</a></h4></li>
  </ul>
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
  <p>Write a short cover letter introducing yourself to the landlord.</p>
  <h3>What to include</h3>
  <p>Proof of income, a SCHUFA report, and copies of your passport.</p>
</article>
</body></html>
"""

GUIDE_PAGE_WITH_NAV_HTML = """
<html><body>
<article>
  <nav aria-label="Breadcrumbs" class="breadcrumbs">
    <ol><li><a href="/guides">Guides</a></li><li>Housing</li><li>Schufa</li></ol>
  </nav>
  <p>Real intro paragraph that is long enough to pass the section filter easily.</p>
  <nav aria-label="Table of contents" class="table-of-contents">
    <h2>On this page</h2>
    <ol><li><a href="#a">What is a Schufa</a></li><li><a href="#b">How to get one</a></li></ol>
  </nav>
  <h2>What is a Schufa</h2>
  <p>Real section content that is long enough to pass the fifty character filter.</p>
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

GUIDE_PAGE_WITH_TABLE_HTML = """
<html><body>
<article>
  <h2>Costs</h2>
  <p>Here is what the different services cost you in total.</p>
  <table>
    <tr><th>Service</th><th>Price</th></tr>
    <tr><td>Anmeldung</td><td>Free</td></tr>
    <tr><td>Vehicle re-registration</td><td>10.80 EUR</td></tr>
  </table>
</article>
</body></html>
"""

GUIDE_PAGE_GLUED_HTML = """
<html><body>
<article>
  <h2>Deregister your address</h2>
  <p>You deregister at the <strong>Bürgeramt</strong><sup id="fnref:1"><a class="footnote-ref" href="#fn:1">31</a></sup>. It is the fastest way to get the certificate you need.</p>
  <p>Every person needs 9 m² of living space in the shared apartment overall.</p>
</article>
</body></html>
"""

GUIDE_PAGE_FOOTNOTES_HTML = """
<html><body>
<article>
  <h2>Register your address</h2>
  <p>You have 14 days to register your address after moving in<sup id="fnref:20"><a class="footnote-ref" href="#fn:20">21</a></sup>, but in Berlin enforcement is lax<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">2</a></sup>.</p>
  <ol>
    <li id="fn:1"><p>reddit.com/r/berlin ⤴</p></li>
    <li id="fn:20"><p><a href="https://example.com">§ 17 Abs. 1 BMG</a>, Berlin.de (January 2026) ⤴</p></li>
  </ol>
</article>
</body></html>
"""

GUIDE_PAGE_WITH_IMAGES_HTML = """
<html><body>
<article>
  <h2>Your tax ID letter</h2>
  <p>You get a letter by post. Here is <a href="https://allaboutberlin.com/images/tax-id-document-bzst.jpg">the letter from the Bundeszentralamt für Steuern</a> that you should keep safe.</p>
  <p>See also this <a href="https://allaboutberlin.com/guides/anmeldung">related guide</a> for the next step in the process.</p>
</article>
</body></html>
"""

GUIDE_PAGE_WITH_IDS_HTML = """
<html><body>
<article>
  <h2 id="finding-a-flat">Finding a flat</h2>
  <p>The most popular platforms are ImmobilienScout24 and WG-Gesucht here.</p>
  <h3 id="where-to-search">Where to search</h3>
  <p>Start with ImmobilienScout24 and check listings every single morning.</p>
  <h2 id="the-application">The application</h2>
  <p>Write a short cover letter and attach proof of income documents.</p>
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

    def test_url_path_matches_href(self):
        anmeldung = next(e for e in self.entries if e.guide == "anmeldung")
        assert anmeldung.url_path == "/guides/anmeldung"


class TestParseGuidesIndexNested:
    def setup_method(self):
        self.entries = parse_guides_index(NESTED_GUIDES_INDEX_HTML)

    def test_slug_is_last_segment(self):
        assert self.entries[0].guide == "for-employees"

    def test_url_path_preserves_full_nested_path(self):
        assert self.entries[0].url_path == "/guides/german-health-insurance/for-employees"

    def test_duplicate_slug_from_different_path_is_skipped(self):
        html = """
        <html><body>
          <h3>A</h3>
          <ul><li><a href="https://allaboutberlin.com/guides/parent-a/child">A</a></li></ul>
          <h3>B</h3>
          <ul><li><a href="https://allaboutberlin.com/guides/parent-b/child">B</a></li></ul>
        </body></html>
        """
        entries = parse_guides_index(html)
        assert len(entries) == 1
        assert entries[0].url_path == "/guides/parent-a/child"


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

    def test_breadcrumbs_and_toc_nav_excluded_from_content(self):
        sections = parse_guide_page(GUIDE_PAGE_WITH_NAV_HTML, "schufa")
        all_text = " ".join(s.text for s in sections)
        assert "Guides" not in all_text
        assert "On this page" not in all_text
        assert not any(s.section == "On this page" for s in sections)
        assert "Real intro paragraph" in all_text
        assert "Real section content" in all_text


class TestParseGuidePageTables:
    def setup_method(self):
        self.sections = parse_guide_page(GUIDE_PAGE_WITH_TABLE_HTML, "costs-guide")

    def test_table_cells_captured_in_section_text(self):
        costs = next(s for s in self.sections if s.section == "Costs")
        assert "Anmeldung | Free" in costs.text
        assert "Vehicle re-registration | 10.80 EUR" in costs.text
        assert "Service | Price" in costs.text

    def test_intro_paragraph_and_table_share_the_section(self):
        costs = next(s for s in self.sections if s.section == "Costs")
        assert "what the different services cost" in costs.text

    def test_no_duplicate_cell_text(self):
        costs = next(s for s in self.sections if s.section == "Costs")
        assert costs.text.count("10.80 EUR") == 1


class TestTextCleaning:
    def setup_method(self):
        self.sections = parse_guide_page(GUIDE_PAGE_GLUED_HTML, "abmeldung")

    def test_words_not_glued_across_inline_tags(self):
        sec = self.sections[0]
        assert "the Bürgeramt" in sec.text
        assert "Bürgeramt31" not in sec.text and "BürgeramtIt" not in sec.text

    def test_footnote_marker_removed(self):
        sec = self.sections[0]
        assert "31" not in sec.text
        assert "Bürgeramt. It is" in sec.text  # de-glued and no space before the period

    def test_unit_superscript_preserved(self):
        text = " ".join(s.text for s in self.sections)
        assert "9 m²" in text  # literal ², not a <sup>, must survive


class TestImages:
    def setup_method(self):
        self.sections = parse_guide_page(GUIDE_PAGE_WITH_IMAGES_HTML, "tax-id")

    def test_image_link_captured_with_caption(self):
        sec = self.sections[0]
        assert sec.images == [{
            "url": "https://allaboutberlin.com/images/tax-id-document-bzst.jpg",
            "caption": "the letter from the Bundeszentralamt für Steuern",
        }]

    def test_non_image_links_ignored(self):
        sec = self.sections[0]
        urls = [im["url"] for im in sec.images]
        assert "https://allaboutberlin.com/guides/anmeldung" not in urls

    def test_sections_without_images_have_empty_list(self):
        sections = parse_guide_page(GUIDE_PAGE_HTML, "find-a-flat-in-berlin")
        assert all(s.images == [] for s in sections)


class TestFootnotes:
    def setup_method(self):
        self.sections = parse_guide_page(GUIDE_PAGE_FOOTNOTES_HTML, "anmeldung")

    def test_legal_footnote_inlined_at_the_claim(self):
        sec = next(s for s in self.sections if s.section == "Register your address")
        assert "§ 17 Abs. 1 BMG" in sec.text
        # law sits in the same chunk as the rule it supports
        assert "14 days to register" in sec.text

    def test_non_legal_footnote_dropped(self):
        text = " ".join(s.text for s in self.sections)
        assert "reddit" not in text  # source-URL footnote is not inlined or dumped

    def test_footnote_list_not_dumped_as_content(self):
        # no chunk is just the footnote wall; back-ref arrows are gone
        text = " ".join(s.text for s in self.sections)
        assert "⤴" not in text
        assert not any("Berlin.de (January 2026)" in s.text and "Register" not in s.section for s in self.sections)

    def test_marker_digits_not_glued(self):
        sec = self.sections[0]
        assert "lax2" not in sec.text and "moving in21" not in sec.text


class TestParseGuidePageAnchors:
    def setup_method(self):
        self.sections = parse_guide_page(GUIDE_PAGE_WITH_IDS_HTML, "find-a-flat-in-berlin")

    def test_h3_id_used_as_anchor_when_present(self):
        where = next(s for s in self.sections if s.title == "Where to search")
        assert where.anchor == "where-to-search"

    def test_h2_id_used_when_no_h3(self):
        application = next(s for s in self.sections if s.section == "The application")
        assert application.anchor == "the-application"

    def test_anchor_empty_when_headings_have_no_ids(self):
        sections = parse_guide_page(GUIDE_PAGE_HTML, "find-a-flat-in-berlin")
        assert all(s.anchor == "" for s in sections)


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
        assert set(entry.keys()) == {"id", "guide", "section", "title", "text", "anchor", "images"}

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

    @responses_lib.activate
    def test_fetches_nested_guide_at_its_real_path_not_flat_reconstruction(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}{GUIDES_PATH}",
            body=NESTED_GUIDES_INDEX_HTML,
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            f"{BASE_URL}/guides/german-health-insurance/for-employees",
            body=GUIDE_PAGE_HTML,
            status=200,
        )

        run(output_dir=tmp_path, delay=0)

        requested_urls = [call.request.url for call in responses_lib.calls]
        assert f"{BASE_URL}/guides/german-health-insurance/for-employees" in requested_urls
        assert f"{BASE_URL}/guides/for-employees" not in requested_urls

        data = json.loads((tmp_path / "guides.json").read_text())
        assert data[0]["sections_count"] > 0
        assert (tmp_path / "for-employees.json").exists()

    @responses_lib.activate
    def test_removes_stale_guide_file_no_longer_in_index(self, tmp_path):
        (tmp_path / "renamed-old-slug.json").write_text("[]", encoding="utf-8")

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

        assert not (tmp_path / "renamed-old-slug.json").exists()
        assert (tmp_path / "guides.json").exists()

    @responses_lib.activate
    def test_writes_meta_json_with_scraped_at(self, tmp_path):
        responses_lib.add(
            responses_lib.GET, f"{BASE_URL}{GUIDES_PATH}", body=GUIDES_INDEX_HTML, status=200,
        )
        for slug in ("find-a-flat-in-berlin", "apartment-deposit", "anmeldung"):
            responses_lib.add(
                responses_lib.GET, f"{BASE_URL}/guides/{slug}", body=GUIDE_PAGE_HTML, status=200,
            )

        run(output_dir=tmp_path, delay=0)

        meta = json.loads((tmp_path / "meta.json").read_text())
        assert "scraped_at" in meta and meta["scraped_at"]
        assert meta["guides"] == 3
        assert meta["sections"] > 0

    @responses_lib.activate
    def test_meta_json_not_deleted_as_stale_on_rerun(self, tmp_path):
        # a meta.json from a previous run must survive the stale-file sweep
        (tmp_path / "meta.json").write_text('{"scraped_at": "old"}', encoding="utf-8")

        responses_lib.add(
            responses_lib.GET, f"{BASE_URL}{GUIDES_PATH}", body=GUIDES_INDEX_HTML, status=200,
        )
        for slug in ("find-a-flat-in-berlin", "apartment-deposit", "anmeldung"):
            responses_lib.add(
                responses_lib.GET, f"{BASE_URL}/guides/{slug}", body=GUIDE_PAGE_HTML, status=200,
            )

        run(output_dir=tmp_path, delay=0)

        assert (tmp_path / "meta.json").exists()
        # and it was refreshed, not left as the old placeholder
        assert json.loads((tmp_path / "meta.json").read_text())["scraped_at"] != "old"
