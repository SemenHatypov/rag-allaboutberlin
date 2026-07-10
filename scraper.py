"""
Scraper for allaboutberlin.com/guides.

Produces two kinds of JSON output (analogous to datatalks.club/faq):
  output/json/guides.json          — index of all guides
  output/json/<slug>.json          — full article text per guide
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://allaboutberlin.com"
GUIDES_PATH = "/guides"
OUTPUT_DIR = Path("output/json")
GUIDES_FILE = "guides.json"
META_FILE = "meta.json"  # snapshot metadata (scraped_at); not a guide document
REQUEST_DELAY = 1.0
REQUEST_TIMEOUT = 15
MIN_SECTION_TEXT_LEN = 50  # shorter blocks are nav fragments, not substantive content

_SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9\-]{0,99}$")


@dataclass
class GuideEntry:
    guide: str          # slug, e.g. "find-a-flat-in-berlin"
    guide_name: str     # display title
    category: str       # top-level section on /guides page
    path: str           # relative JSON path, e.g. "/json/find-a-flat-in-berlin.json"
    sections_count: int = 0
    url_path: str = ""  # real site path, e.g. "/guides/german-health-insurance/for-employees"


@dataclass
class GuideSection:
    id: str             # 10-char hex hash
    guide: str          # parent guide slug
    section: str        # h2 heading (may be empty string if article has no h2)
    title: str          # h3 sub-heading or empty string
    text: str           # plain text content of that block
    anchor: str = ""    # heading id for deep-linking (h3 id, else h2 id, else "")


def fetch_html(url: str, session: requests.Session | None = None) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AAB-scraper/1.0)"}
    if session is not None:
        response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    with requests.Session() as s:
        response = s.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text


def make_id(guide: str, section: str, title: str) -> str:
    # Non-security use: content-addressable ID for deduplication, not auth.
    # 10 hex chars = 40-bit space; collision-free at current scale (~2k sections).
    raw = f"{guide}|{section}|{title}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:10]


def slug_from_url(url: str) -> str:
    path = url.replace(BASE_URL, "").rstrip("/")
    slug = path.split("/")[-1]
    if not _SAFE_SLUG.match(slug):
        raise ValueError(f"Unexpected slug format: {slug!r}")
    return slug


def parse_guides_index(html: str) -> list[GuideEntry]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[GuideEntry] = []
    seen_slugs: dict[str, str] = {}

    for h3 in soup.find_all("h3"):
        category = h3.get_text(strip=True)
        container = h3.find_next_sibling()
        if container is None:
            continue
        for a in container.find_all("a", href=True):
            href = str(a["href"])
            if "/guides/" not in href:
                continue
            slug = slug_from_url(href)
            url_path = href.replace(BASE_URL, "").rstrip("/")
            if slug in seen_slugs and seen_slugs[slug] != url_path:
                logger.warning(
                    "Duplicate slug %r from %r collides with earlier %r; skipping",
                    slug, url_path, seen_slugs[slug],
                )
                continue
            seen_slugs[slug] = url_path
            name = str(a.get("title") or a.get_text(strip=True))
            entries.append(GuideEntry(
                guide=slug,
                guide_name=name,
                category=category,
                path=f"/json/{slug}.json",
                url_path=url_path,
            ))

    return entries


def parse_guide_page(html: str, slug: str) -> list[GuideSection]:
    soup = BeautifulSoup(html, "lxml")

    # Try common content wrappers; fall back to <main> or <body>
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="content")
        or soup.find("body")
    )
    if content is None:
        return []

    sections: list[GuideSection] = []
    current_h2 = ""
    current_h3 = ""
    current_h2_id = ""
    current_h3_id = ""
    buffer: list[str] = []

    def flush(h2: str, h3: str, h2_id: str, h3_id: str, buf: list[str]) -> None:
        text = " ".join(buf).strip()
        if text:
            sections.append(GuideSection(
                id=make_id(slug, h2, h3),
                guide=slug,
                section=h2,
                title=h3,
                text=text,
                anchor=h3_id or h2_id,  # prefer the more specific h3 anchor
            ))

    for tag in content.find_all(["h2", "h3", "p", "li", "ul", "ol", "table"], recursive=True):
        if tag.find_parent("nav") is not None:
            continue  # skip breadcrumbs and the table-of-contents nav
        if tag.name == "h2":
            flush(current_h2, current_h3, current_h2_id, current_h3_id, buffer)
            buffer = []
            current_h2 = tag.get_text(strip=True)
            current_h2_id = tag.get("id") or ""
            current_h3 = ""
            current_h3_id = ""
        elif tag.name == "h3":
            flush(current_h2, current_h3, current_h2_id, current_h3_id, buffer)
            buffer = []
            current_h3 = tag.get_text(strip=True)
            current_h3_id = tag.get("id") or ""
        elif tag.name == "p":
            if tag.find_parent("table") is not None:
                continue  # cell content is captured by the table handler
            text = tag.get_text(strip=True)
            if text:
                buffer.append(text)
        elif tag.name == "li":
            if tag.find_parent("table") is not None:
                continue  # cell content is captured by the table handler
            if tag.parent and tag.parent.name in ("ul", "ol"):
                text = tag.get_text(strip=True)
                if text:
                    buffer.append(text)
        elif tag.name == "table":
            text = serialize_table(tag)
            if text:
                buffer.append(text)
        elif tag.name in ("ul", "ol"):
            pass  # handled via <li>

    flush(current_h2, current_h3, current_h2_id, current_h3_id, buffer)
    return [s for s in sections if is_valid_section(s)]


def serialize_table(table) -> str:
    """Flatten a <table> to text: cells joined by ' | ', rows by newline."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()


def is_valid_section(section: GuideSection) -> bool:
    return len(section.text.strip()) >= MIN_SECTION_TEXT_LEN


def save_json(data: list[GuideEntry | GuideSection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in data], f, ensure_ascii=False, indent=2)


def run(output_dir: Path = OUTPUT_DIR, delay: float = REQUEST_DELAY) -> None:
    with requests.Session() as session:
        _scrape(session, output_dir, delay)


def _scrape(session: requests.Session, output_dir: Path, delay: float) -> None:
    logger.info("Fetching guides index...")
    index_html = fetch_html(f"{BASE_URL}{GUIDES_PATH}", session)
    entries = parse_guides_index(index_html)
    logger.info("Found %d guides across categories", len(entries))

    for i, entry in enumerate(entries, 1):
        url = f"{BASE_URL}{entry.url_path}"
        logger.info("[%d/%d] %s", i, len(entries), entry.guide)
        try:
            html = fetch_html(url, session)
            sections = parse_guide_page(html, entry.guide)
            entries[i - 1] = replace(entry, sections_count=len(sections))
            save_json(sections, output_dir / f"{entry.guide}.json")
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning("Failed to scrape %s: %s", entry.guide, e)
        if i < len(entries):
            time.sleep(delay)

    _remove_stale_guide_files(output_dir, {e.guide for e in entries})
    save_json(entries, output_dir / GUIDES_FILE)
    total_sections = sum(e.sections_count for e in entries)
    _write_meta(output_dir, len(entries), total_sections)
    logger.info("Done. %d guides, %d sections total.", len(entries), total_sections)
    logger.info("Output: %s", output_dir.resolve())


def _write_meta(output_dir: Path, guides: int, sections: int) -> None:
    meta = {
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "guides": guides,
        "sections": sections,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _remove_stale_guide_files(output_dir: Path, current_slugs: set[str]) -> None:
    if not output_dir.exists():
        return
    for json_file in output_dir.glob("*.json"):
        if json_file.name in (GUIDES_FILE, META_FILE):
            continue
        if json_file.stem not in current_slugs:
            logger.info("Removing stale guide file: %s", json_file.name)
            json_file.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
