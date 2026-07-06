# All About Berlin RAG

A scraper, RAG pipeline, and chat interface for [allaboutberlin.com](https://allaboutberlin.com) guides. Scrapes 151 guides into structured JSON, then lets you ask questions about living in Berlin and get short grounded answers from an LLM with clickable links to the source articles — via CLI or a Streamlit chat app.

**🐻 Try it live: [rag-all-about-berlin.streamlit.app](https://rag-all-about-berlin.streamlit.app/)** — no installation needed, just open the link and ask a question in the chat.

Built as a capstone project for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) by DataTalksClub.

**Coverage:** ![Coverage Badge Placeholder](https://img.shields.io/badge/coverage-98%25-brightgreen)

## Overview

This scraper crawls https://allaboutberlin.com/guides and produces two types of JSON output:

1. **`output/json/guides.json`** — Index of all 151 guides with metadata (analogous to `courses.json`)
2. **`output/json/<slug>.json`** — Full article text split by sections for each guide (analogous to `llm-zoomcamp.json`)

The resulting data contains **151 guides** organized into **1,808 sections**, ready for search indexing, RAG systems, or downstream processing. At load time each document is enriched with the guide's human-readable title (`guide_name`) and its canonical article URL (`https://allaboutberlin.com<url_path>`, using the guide's real site path — some guides are nested under a parent guide, e.g. `/guides/german-health-insurance/for-employees`).

> **Note:** The scraped data in `output/json/` is tracked in git (it is required by the deployed Streamlit app), so the RAG and agent pipelines work out of the box. Run `uv run python scraper.py` only if you want to re-scrape fresh data. Other `output/` artifacts (evaluation results, ground truth) are not tracked.

## JSON Output Format

### guides.json — Index

```json
[
  {
    "guide": "find-a-flat-in-berlin",
    "guide_name": "How to find an apartment in Berlin",
    "category": "Housing",
    "path": "/json/find-a-flat-in-berlin.json",
    "sections_count": 40,
    "url_path": "/guides/find-a-flat-in-berlin"
  },
  {
    "guide": "for-employees",
    "guide_name": "German health insurance for employees",
    "category": "Personal finance",
    "path": "/json/for-employees.json",
    "sections_count": 12,
    "url_path": "/guides/german-health-insurance/for-employees"
  }
]
```

**Fields:**
- `guide` (str) — URL-safe slug, used as file identifier (last path segment; for nested guides this drops the parent segment, so it isn't unique enough to reconstruct the URL — use `url_path` for that)
- `guide_name` (str) — Display title of the guide
- `category` (str) — Top-level section from the /guides page
- `path` (str) — Relative path to the per-guide JSON file
- `sections_count` (int) — Number of sections scraped for this guide (0 if fetch failed)
- `url_path` (str) — Real site path after the domain, used to build the guide's canonical URL and to fetch it (some guides live under a parent guide, e.g. `/guides/german-health-insurance/for-employees`)

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

The easiest way to interact with the RAG system — ask questions in a chat interface.

**Hosted version:** the app is deployed on Streamlit Community Cloud at **[rag-all-about-berlin.streamlit.app](https://rag-all-about-berlin.streamlit.app/)**. Every push to `main` redeploys it automatically. (On the free tier the app goes to sleep after ~12 hours without visitors — the first visitor wakes it up with one click, then the vector index builds in about a minute.)

To run it locally instead:

```bash
uv run streamlit run app.py
```

Opens at **http://localhost:8501**.

**Features:**
- Chat interface with full conversation history
- Short summary-style answers (2–4 sentences) grounded in the retrieved guides
- Clickable **source links** under each answer — the allaboutberlin.com articles the answer is based on, in relevance order (correct article cited in 94.7% of eval questions)
- Semantic vector search (all-MiniLM-L6-v2) — the highest-accuracy backend
- Sidebar to filter answers to a specific guide (142 options)
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

The CLI prints a short answer followed by a `Sources:` block with links to the articles the answer is based on.

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

# Vector/semantic search, with source articles
vindex, embedder = build_vector_index(documents)
vpipeline = RAGVector(embedder=embedder, index=vindex, llm_client=OpenAI())
answer, sources = vpipeline.rag_with_sources("How do I find a flat?")
print(answer)
for src in sources:                     # unique guides from top-k results, rank order
    print(f"{src['guide_name']} — {src['url']}")
```

### Agentic RAG (iterative tool-calling loop)

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

### Ground Truth Generation

Builds the article-level ground truth dataset: for every guide with content (142 of them), an LLM generates 2 distinct questions that this guide answers best. Result: ~284 "question → correct article" pairs in `output/ground-truth-guides.{csv,json}` with columns `question, guide, guide_name, url`.

```bash
# Estimate cost and exit (recommended first step; full run costs ~$0.20)
uv run python generate_ground_truth.py --dry-run

# Generate the dataset
uv run python generate_ground_truth.py

# More questions per guide
uv run python generate_ground_truth.py --questions-per-guide 3
```

**Options:** `--questions-per-guide` (default 2), `--workers`, `--output-dir`, `--dry-run`.

> An earlier section-level ground truth and evaluation (hit = exact section match) is available in the git history.

### Search Evaluation

The evaluation of this project is deliberately retrieval-only: given a natural-language question in English, **did the system find the correct guide article?** `evaluate_search.py` measures this for BM25 and vector search: a query is a hit when the correct guide appears among the sources the chatbot would cite for the top-k results (`rag_helper.extract_sources` — unique guides in rank order). Questions whose guide is no longer in the corpus (e.g. removed by a re-scrape) are dropped automatically. The run makes **no LLM calls** and is free.

> **Prerequisite:** Run `uv run python generate_ground_truth.py` first to build the ground truth dataset.

```bash
# Evaluate both text and vector search (default)
uv run python evaluate_search.py

# Evaluate only BM25 text search
uv run python evaluate_search.py --method text

# Run grid search to find optimal BM25 boost parameters
uv run python evaluate_search.py --method text --tune

# Deeper retrieval (more citable sources per question)
uv run python evaluate_search.py --num-results 10
```

**Results** (284 questions, top-5, guide-level):

```
=== Results ===
                                     hit_rate    mrr
method
text_search (title=2.0, section=0.5)    0.539  0.412
text_search (title=0.5, section=0.5)    0.813  0.669
vector_search (all-MiniLM-L6-v2)        0.947  0.829
```

**Metrics:**
- **Hit Rate** — fraction of queries where the correct guide appears among the unique guides of the top-5 results
- **MRR (Mean Reciprocal Rank)** — rewards citing the correct guide at higher ranks (rank 1 = 1.0, rank 2 = 0.5, …)

Vector search finds the correct article **1.8× more often** than BM25 — it is the backend the Streamlit app uses.

**Options:** `--method {text,vector,all}` (default `all`), `--tune` (BM25 boost grid search), `--num-results`, `--workers`, `--ground-truth`, `--output` (save results JSON).

> An earlier end-to-end RAG evaluation (answer generation + LLM-as-a-judge relevance verdicts) is available in the git history; it was removed to keep the evaluation focused on retrieval quality.

### Run the Scraper

```bash
uv run python scraper.py
```

This will:
1. Fetch the guides index from https://allaboutberlin.com/guides
2. Scrape each guide page
3. Parse sections by H2, H3, and content blocks
4. Save JSON files to `output/json/`

**Output directory structure** (tracked in git — re-running the scraper refreshes it):
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

## Deployment

The chat app is hosted on [Streamlit Community Cloud](https://streamlit.io/cloud): **[rag-all-about-berlin.streamlit.app](https://rag-all-about-berlin.streamlit.app/)**

- **Entry point:** `app.py`, deployed from the `main` branch — every push redeploys automatically
- **Dependencies:** installed from `uv.lock` (Community Cloud supports uv natively)
- **Data:** the scraped guides in `output/json/` are tracked in git, so the app indexes them directly from the repo
- **Secrets:** `OPENAI_API_KEY` is set in the app's Secrets settings on Community Cloud (exposed to the app as an environment variable)

## Testing

Run tests with pytest (~80 tests covering the scraper, ingest, RAG helpers, and evaluation scoring):

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
    url_path: str       # real site path, e.g. "/guides/german-health-insurance/for-employees"
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
├── load_guides(json_dir)             — Load guides.json index (slug → guide_name)
├── guide_url(slug)                   — Fallback flat article URL (used if url_path missing)
├── load_documents(json_dir)          — Load per-guide JSON files, enrich each doc
│                                        with guide_name + url (from guides.json's url_path)
├── build_index(documents)            — BM25 keyword index (minsearch.Index)
└── build_vector_index(documents)     — Semantic vector index (minsearch.VectorSearch)
                                         + SentenceTransformer embedder

rag_helper.py
├── extract_sources(results)          — Unique guides from ranked results (rank order):
│                                        {guide, guide_name, url, sections}
├── RAGBase                           — Keyword-search RAG pipeline
│   ├── search(query)                 — BM25 search with boost/filter
│   ├── build_context(results)        — Format results into LLM context
│   ├── build_prompt(query, results)  — Combine context with question
│   ├── llm(prompt)                   — Call OpenAI API
│   ├── rag_with_sources(query)       — End-to-end: (short answer, cited sources)
│   └── rag(query)                    — Answer only
└── RAGVector(RAGBase)                — Semantic-search RAG pipeline
    └── search(query)                 — Encode query → vector search

agent.py
└── AgentRAG                          — Agentic RAG with iterative tool-calling loop
    ├── _search(query)                — Search index (no boost, direct lookup)
    ├── _execute_tool_call(call)      — Run tool call, return function_call_output
    └── loop(question)               — Agentic loop: call LLM → dispatch tools → repeat

generate_ground_truth.py              — Build article-level ground truth
├── group_by_guide(documents)         — Group sections by guide slug
├── build_guide_text(docs)            — One text digest per guide (truncated)
└── main()                            — 2 questions per guide → output/ground-truth-guides.{csv,json}

evaluate_search.py                    — Article-level retrieval evaluation
├── cited_guides(results)             — Guide slugs the chatbot would cite (via extract_sources)
├── filter_known_guides(gt, docs)     — Drop questions for guides no longer in the corpus
└── hit_rate / mrr                    — Guide-level Hit Rate & MRR, BM25 vs vector, boost grid search

evaluation_utils.py                   — Shared helpers: structured LLM calls, cost accounting, parallel map

app.py                                — Streamlit chat app (vector search only)
├── _load_index()                     — @st.cache_resource: loads docs + builds vector index once
├── _render_sources(sources)          — Numbered clickable links to source articles
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
