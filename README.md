# All About Berlin RAG

A scraper, RAG pipeline, and chat interface for [allaboutberlin.com](https://allaboutberlin.com) guides. Scrapes 149 guides into structured JSON, then lets you ask questions about living in Berlin and get grounded answers from an LLM — via CLI or a Streamlit chat app.

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

### Streamlit Chat App

The easiest way to interact with the RAG system. Run a local web app and ask questions in a chat interface.

> **Prerequisite:** Run `uv run python scraper.py` first to populate `output/json/` with scraped data.

```bash
uv run streamlit run app.py
```

Opens at **http://localhost:8501**.

**Features:**
- Chat interface with full conversation history
- Semantic vector search (all-MiniLM-L6-v2) — the highest-accuracy backend (92% good answers vs 75% for BM25)
- "Sources" expander under each answer showing the retrieved guide sections
- Sidebar to filter answers to a specific guide (149 options)
- Slider to control how many source sections are retrieved (3–15)
- "Clear chat" button to reset the conversation

**Example questions to try:**
- `How do I register my address in Berlin (Anmeldung)?`
- `What is Schufa and how can I get my credit report for free?`
- `I'm a freelancer — which health insurance should I choose?`
- `What's the difference between a freelance visa and a Blue Card?`
- `How do I find an apartment and what's a typical rental deposit?`

---

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

`rag_helper.py` can also be imported as a module in your own scripts:

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

### Search Evaluation

Evaluates retrieval quality of BM25 and vector search using Hit Rate and MRR metrics against the ground truth dataset.

> **Prerequisite:** Run `uv run python scraper.py` and `uv run python generate_ground_truth.py` first.

```bash
# Evaluate both text and vector search (default)
uv run python evaluate_search.py

# Evaluate only BM25 text search
uv run python evaluate_search.py --method text

# Evaluate only vector search
uv run python evaluate_search.py --method vector

# Run grid search to find optimal BM25 boost parameters
uv run python evaluate_search.py --method text --tune

# Save results to JSON
uv run python evaluate_search.py --output output/search-eval-results.json
```

**Example output:**

```
=== Results ===
                                        hit_rate    mrr
method
text_search (title=2.0, section=0.5)       0.649  0.520
text_search (title=0.5, section=0.5)       0.770  0.641
vector_search (all-MiniLM-L6-v2)           0.868  0.690
```

**Metrics:**
- **Hit Rate** — fraction of queries where the correct document appears in top-5 results
- **MRR (Mean Reciprocal Rank)** — rewards finding the correct document at higher ranks (rank 1 = 1.0, rank 2 = 0.5, …)

### Answer Quality Evaluation (LLM-as-a-Judge)

Search evaluation tells us whether the *right document* was retrieved — it says nothing about whether the *generated answer* is correct. `evaluate_rag.py` measures answer quality with an **LLM-as-a-judge**: a second LLM call that reads each answer and rates it.

It uses the same **A → Q → A′** structure the ground truth was built from:

```
A   original guide section text          (the "correct" answer)
 └─ Q   a question generated from it      (output/ground-truth-data.csv)
     └─ A′  the answer the RAG returns    (generated and judged here)
```

> **Prerequisite:** Run `uv run python scraper.py` and `uv run python generate_ground_truth.py` first.

```bash
# Estimate cost on a 10-question pilot, then exit (recommended first step)
uv run python evaluate_rag.py --dry-run

# Quick run on a random 25-question sample
uv run python evaluate_rag.py --sample 25

# Full run with both judge modes
uv run python evaluate_rag.py --judge both

# Compare keyword vs vector backends (both judge modes)
uv run python evaluate_rag.py --search-type both --judge both
```

**Two judge modes:**

| Mode | Judge sees | Verdict | Use case |
|------|-----------|---------|----------|
| `reference` (default) | question + original answer **A** + RAG answer **A′** | `good` / `bad` | Offline / development (needs ground truth) |
| `reference-free` | question + RAG answer **A′** only | `RELEVANT` / `PARTLY_RELEVANT` / `NON_RELEVANT` | Online / production (no reference available) |
| `both` | — | both sets of columns | Side-by-side comparison |

**Three search backends** (`--search-type`):

| Value | Backend | Description |
|-------|---------|-------------|
| `keyword` (default) | BM25 | Fast, no GPU needed |
| `vector` | all-MiniLM-L6-v2 | Semantic embeddings |
| `both` | — | Run both and compare; writes per-type files + combined `rag-eval-comparison.{csv,json}` |

Results are written to `output/rag-eval-<type>.{csv,json}` (one row per question, with the judge's `reasoning` and verdict), and a summary is printed. Keyword results are also aliased to `output/rag-eval.{csv,json}` for backwards compatibility.

**Keyword vs vector comparison** (289 questions, both judge modes, `gpt-4o-mini`):

| Metric | keyword (BM25) | vector (MiniLM) |
|--------|---------------|-----------------|
| Reference judge — `good` | 216 / 289 (74.7%) | **266 / 289 (92.0%)** |
| Reference judge — `bad` | 73 / 289 (25.3%) | **23 / 289 (8.0%)** |
| Reference-free — `RELEVANT` | 228 / 289 (78.9%) | **274 / 289 (94.8%)** |
| Reference-free — `PARTLY_RELEVANT` | 23 / 289 (8.0%) | 11 / 289 (3.8%) |
| Reference-free — `NON_RELEVANT` | 38 / 289 (13.1%) | **4 / 289 (1.4%)** |

Vector search produces **+17 pp more good answers** and **9× fewer non-relevant responses** — consistent with its higher retrieval Hit Rate (0.868 vs 0.770).

The judge always returns a `reasoning` field alongside its verdict. A `bad` / `NON_RELEVANT` verdict is a **pointer for investigation**, not a final truth — read the flagged rows in the output to decide whether the RAG answer, the question, or the ground truth is at fault.

**Options:** `--search-type {keyword,vector,both}`, `--judge {reference,reference-free,both}`, `--sample N`, `--workers`, `--rag-model`, `--judge-model`, `--ground-truth`, `--output-dir`, `--seed`, `--dry-run`. Both models default to `gpt-4o-mini`.

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

generate_ground_truth.py              — Build Q&A ground truth (samples docs → generates one question each)

evaluate_search.py                    — Retrieval evaluation: Hit Rate & MRR, BM25 vs vector, boost grid search

evaluate_rag.py                       — Answer evaluation (LLM-as-a-judge)
├── RAGTracked(RAGBase)               — Keyword RAG pipeline with token usage tracking
├── RAGVectorTracked(RAGVector)        — Vector RAG pipeline with token usage tracking
├── build_assistant(search_type, ...) — Factory: builds the right pipeline for keyword/vector
├── generate_rag_answers(...)         — Produce A′ for each ground-truth question
├── judge_with_reference(...)         — Score good/bad vs the original answer A
├── judge_reference_free(...)         — Score relevance from the question alone
└── main()                            — CLI: generate → judge → summarize → output/rag-eval-<type>.{csv,json}

evaluation_utils.py                   — Shared helpers: structured LLM calls, cost accounting, parallel map

app.py                                — Streamlit chat app (vector search only)
├── _load_index()                     — @st.cache_resource: loads docs + builds vector index once
├── _render_sources(sources)          — Renders retrieved sections in an expander
└── main()                            — Page layout, sidebar settings, chat loop
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
| `streamlit` | Chat web app |
| `pytest` + `pytest-cov` | Tests and coverage |
| `responses` | HTTP mocking in tests |

## License

[Add license info if applicable]
