# All About Berlin RAG

A scraper and RAG pipeline for [allaboutberlin.com](https://allaboutberlin.com) guides. Scrapes 149 guides into structured JSON, then lets you ask questions about living in Berlin and get grounded answers from an LLM.

Built as a capstone project for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) by DataTalksClub.

**Coverage:** ![Coverage Badge Placeholder](https://img.shields.io/badge/coverage-98%25-brightgreen)

## Overview

This scraper crawls https://allaboutberlin.com/guides and produces two types of JSON output:

1. **`output/json/guides.json`** — Index of all 149 guides with metadata (analogous to `courses.json`)
2. **`output/json/<slug>.json`** — Full article text split by sections for each guide (analogous to `llm-zoomcamp.json`)

The resulting data contains **149 guides** organized into **1,905 sections**, ready for search indexing, RAG systems, or downstream processing.

> **Note:** The `output/` directory is not tracked in git. Run `uv run python scraper.py` first to generate the data locally before using the RAG or agent pipelines.

## JSON Output Format

### guides.json — Index

```json
[
  {
    "guide": "find-a-flat-in-berlin",
    "guide_name": "How to find an apartment in Berlin",
    "category": "Housing",
    "path": "/json/find-a-flat-in-berlin.json",
    "sections_count": 40
  },
  {
    "guide": "schufa",
    "guide_name": "How to get a free Schufa",
    "category": "Housing",
    "path": "/json/schufa.json",
    "sections_count": 7
  }
]
```

**Fields:**
- `guide` (str) — URL-safe slug, used as file identifier
- `guide_name` (str) — Display title of the guide
- `category` (str) — Top-level section from the /guides page
- `path` (str) — Relative path to the per-guide JSON file
- `sections_count` (int) — Number of sections scraped for this guide (0 if fetch failed)

### Per-Guide JSON — Sections

Example: `output/json/find-a-flat-in-berlin.json`

```json
[
  {
    "id": "da23b67dd1",
    "guide": "find-a-flat-in-berlin",
    "section": "Is it hard to find an apartment?",
    "title": "",
    "text": "Yes, it's really hard. It can take months and hundreds of messages..."
  },
  {
    "id": "435fd2b9ad",
    "guide": "find-a-flat-in-berlin",
    "section": "1. Look for apartments",
    "title": "Apartment search websites",
    "text": "ImmoScout24 is the biggest housing website..."
  }
]
```

**Fields:**
- `id` (str) — 10-character hex ID for deduplication (SHA1 of guide+section+title)
- `guide` (str) — Parent guide slug
- `section` (str) — H2 heading; empty string if article has no H2 headers
- `title` (str) — H3 sub-heading; empty string if no H3 present
- `text` (str) — Plain text content (paragraphs and list items combined)

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv automatically)
uv sync
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=sk-...
```

## Usage

All commands use `uv run` — no need to activate the virtual environment manually.

### RAG Pipeline (ask questions)

> **Prerequisite:** Run `uv run python scraper.py` first to populate `output/json/` with scraped data.

```bash
# Ask a question (keyword/BM25 search — default)
uv run python rag_helper.py --question "How do I register my address in Berlin?"

# Use vector/semantic search
uv run python rag_helper.py --question "How do I register my address in Berlin?" --search-type vector

# Use a different model
uv run python rag_helper.py --question "What is a Schufa?" --model gpt-4o

# Filter to a specific guide
uv run python rag_helper.py --question "What documents do I need?" --guide anmeldung

# Use more search results for context
uv run python rag_helper.py --question "How does health insurance work?" --num-results 10
```

`rag_helper.py` can also be imported as a module in notebooks or scripts:

```python
from openai import OpenAI
from ingest import load_documents, build_index, build_vector_index
from rag_helper import RAGBase, RAGVector

documents = load_documents()           # loads output/json/*.json

# Keyword search
index = build_index(documents)
pipeline = RAGBase(index=index, llm_client=OpenAI())
print(pipeline.rag("How do I find a flat?"))

# Vector/semantic search
vindex, embedder = build_vector_index(documents)
vpipeline = RAGVector(embedder=embedder, index=vindex, llm_client=OpenAI())
print(vpipeline.rag("How do I find a flat?"))
```

### Agentic RAG (iterative tool-calling loop)

> **Prerequisite:** Run `uv run python scraper.py` first to populate `output/json/` with scraped data.

`agent.py` implements an agentic loop: the model calls the `search` tool repeatedly, refining its queries, until it has enough context to answer. It also has guardrails — off-topic questions are rejected.

```bash
# Ask with the agentic loop (default model: gpt-4o-mini)
uv run python agent.py --question "What is Schufa?"

# Test typo self-correction
uv run python agent.py --question "What is shufa?"

# Off-topic query — rejected by guardrails
uv run python agent.py --question "What is the queen's gambit?"

# Use a different model
uv run python agent.py --question "How do I get health insurance?" --model gpt-4o
```

`agent.py` can also be imported as a module:

```python
from openai import OpenAI
from ingest import load_documents, build_index
from agent import AgentRAG

documents = load_documents()
index = build_index(documents)
agent = AgentRAG(index=index, llm_client=OpenAI())
print(agent.loop("How do I find a flat in Berlin?"))
```

### Notebooks

Experiments live in `notebooks/`. To open them:

```bash
uv run jupyter notebook notebooks/
```

| Notebook | Description |
|----------|-------------|
| `01_rag_intro.ipynb` | Step-by-step RAG pipeline — search, prompt building, LLM call, cost tracking |
| `02_vector_search.ipynb` | Vector/semantic search demo — embeddings, VectorSearch index, RAGVector pipeline |
| `03_agents.ipynb` | Agentic RAG prototype — function calling, iterative tool-use loop, guardrails |

> **Note:** Notebook files (`*.ipynb`) are not tracked in git — they exist locally as development artifacts.

### Run the Scraper

```bash
uv run python scraper.py
```

This will:
1. Fetch the guides index from https://allaboutberlin.com/guides
2. Scrape each guide page
3. Parse sections by H2, H3, and content blocks
4. Save JSON files to `output/json/`

**Output directory structure** (generated locally, not tracked in git):
```
output/json/
├── guides.json
├── find-a-flat-in-berlin.json
├── housing-scams.json
├── anmeldung.json
└── ... (one file per guide)
```

### Configuration

Edit the constants at the top of `scraper.py`:

```python
BASE_URL = "https://allaboutberlin.com"  # Source domain
OUTPUT_DIR = Path("output/json")          # Output directory
REQUEST_DELAY = 1.0                       # Seconds between requests
REQUEST_TIMEOUT = 15                      # HTTP timeout
```

## Testing

Run tests with pytest (44 tests, 98% coverage):

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=scraper --cov-report=term-missing

# Run specific test class
uv run pytest tests/test_scraper.py::TestParseGuidesIndex

# Verbose output
uv run pytest -v
```

## Data Structure

### GuideEntry (from guides.json)

```python
@dataclass
class GuideEntry:
    guide: str          # slug, e.g. "find-a-flat-in-berlin"
    guide_name: str     # display title
    category: str       # top-level section
    path: str           # relative JSON path
    sections_count: int # number of sections (default: 0)
```

### GuideSection (from per-guide JSON)

```python
@dataclass
class GuideSection:
    id: str             # 10-char hex hash
    guide: str          # parent guide slug
    section: str        # h2 heading (empty if none)
    title: str          # h3 sub-heading (empty if none)
    text: str           # plain text content
```

## Architecture

```
scraper.py
├── fetch_html(url, session)          — Fetch and return HTML
├── parse_guides_index(html)          — Extract all guides from /guides page
├── parse_guide_page(html, slug)      — Parse sections from a single guide
├── make_id(guide, section, title)    — Generate content-addressable ID
├── slug_from_url(url)                — Extract and validate slug
├── save_json(data, path)             — Write JSON output
└── run(output_dir, delay)            — Main scraper orchestration

ingest.py
├── load_documents(json_dir)          — Load all per-guide JSON files
├── build_index(documents)            — BM25 keyword index (minsearch.Index)
└── build_vector_index(documents)     — Semantic vector index (minsearch.VectorSearch)
                                         + SentenceTransformer embedder

rag_helper.py
├── RAGBase                           — Keyword-search RAG pipeline
│   ├── search(query)                 — BM25 search with boost/filter
│   ├── build_context(results)        — Format results into LLM context
│   ├── build_prompt(query, results)  — Combine context with question
│   ├── llm(prompt)                   — Call OpenAI API
│   └── rag(query)                    — End-to-end: search → prompt → LLM
└── RAGVector(RAGBase)                — Semantic-search RAG pipeline
    └── search(query)                 — Encode query → vector search

agent.py
└── AgentRAG                          — Agentic RAG with iterative tool-calling loop
    ├── _search(query)                — Search index (no boost, direct lookup)
    ├── _execute_tool_call(call)      — Run tool call, return function_call_output
    └── loop(question)               — Agentic loop: call LLM → dispatch tools → repeat
```

## Error Handling

- Failed guide fetches (HTTP 4xx/5xx) are logged and skipped
- Sections with invalid slugs raise `ValueError`
- Parent directories are created automatically
- Unicode content is preserved in JSON output

## Security Notes

- **Slug validation:** All slugs are validated against a strict allowlist pattern (`^[a-z0-9][a-z0-9\-]{0,99}$`)
- **Content-addressable IDs:** IDs are deterministic but non-cryptographic (safe for deduplication only)
- **User-Agent header:** Sent with all requests for polite scraping
- **Session reuse:** HTTP session is reused for connection pooling and efficiency

## Dependencies

Managed via `pyproject.toml` and `uv.lock`.

| Package | Purpose |
|---------|---------|
| `openai` | LLM API calls |
| `minsearch` | In-memory full-text search (BM25 and vector) |
| `sentence-transformers` | Embedding model for vector search |
| `python-dotenv` | Load `OPENAI_API_KEY` from `.env` |
| `requests` | HTTP client for the scraper |
| `beautifulsoup4` + `lxml` | HTML parsing |
| `jupyter` | Notebook environment |
| `pytest` + `pytest-cov` | Tests and coverage |
| `responses` | HTTP mocking in tests |

## License

[Add license info if applicable]
