# Contributing

This document describes how to set up your development environment, run tests, and contribute to the scraper.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

### Initial Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repo-url>
   cd rag-allaboutberlin
   ```

2. **Install dependencies** (creates `.venv` automatically):
   ```bash
   uv sync
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY
   ```

### Verify Setup

```bash
uv run python scraper.py --help  # Should run without error
uv run pytest --version          # Should show pytest version
```

## Running Tests

### All Tests

```bash
uv run pytest
```

### With Coverage Report

```bash
uv run pytest --cov=scraper --cov-report=term-missing
```

This shows:
- Coverage percentage per function
- Lines not yet covered (marked with `>`), allowing you to identify gaps
- Coverage fails if below the 80% threshold (configured in `pyproject.toml`)

### Run Specific Tests

```bash
# All tests for a class
uv run pytest tests/test_scraper.py::TestParseGuidesIndex

# Single test
uv run pytest tests/test_scraper.py::TestParseGuidesIndex::test_returns_correct_count

# Tests matching a name pattern
uv run pytest -k "slug" -v

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

### Test Organization

Tests are organized by function/class in `tests/test_scraper.py`:

- **TestSlugFromUrl** — Slug validation, including security edge cases
- **TestMakeId** — Content-addressable ID generation
- **TestParseGuidesIndex** — Index page parsing
- **TestParseGuidePage** — Individual guide page parsing
- **TestSaveJson** — JSON output and directory creation
- **TestFetchHtml** — HTTP client behavior
- **TestRun** — Integration tests with mocked HTTP

All tests use HTTP mocking (`responses` library) and temporary directories (`tmp_path` fixture) — no live network calls.

## Code Style

### Use Logging, Not Print

Avoid `print()` statements. Use the logging module instead:

```python
# WRONG
print("Scraped 5 guides")

# CORRECT
import logging
logger = logging.getLogger(__name__)
logger.info("Scraped %d guides", count)
```

The main script (`scraper.py`) configures logging:
```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
```

### Immutability and Dataclasses

Use `@dataclass` for data structures, and preserve immutability where possible:

```python
from dataclasses import dataclass, replace

@dataclass
class GuideEntry:
    guide: str
    sections_count: int = 0

# DON'T mutate in-place:
# entry.sections_count = 10

# DO create a new object:
entry = replace(entry, sections_count=10)
```

Serialize dataclasses to JSON using `asdict()`:

```python
from dataclasses import asdict
import json

entries = [GuideEntry("slug", 5)]
json.dump([asdict(e) for e in entries], f)
```

### Function Guidelines

- Keep functions small (<50 lines)
- Use meaningful names (`parse_guide_page` not `process_html`)
- Include type annotations on all parameters and return values:
  ```python
  def fetch_html(url: str, session: requests.Session | None = None) -> str:
      ...
  ```

### File Organization

The scraper is a single module (`scraper.py`, ~200 lines). If you add significant functionality:
- Extract utilities to separate functions
- Keep related logic together (all parsing functions grouped, all I/O functions grouped)
- Split only if a file exceeds ~400 lines

## Security

### Slug Validation

All slugs are validated against a strict allowlist:

```python
_SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9\-]{0,99}$")

def slug_from_url(url: str) -> str:
    # Extract last path segment and validate
    slug = url.replace(BASE_URL, "").rstrip("/").split("/")[-1]
    if not _SAFE_SLUG.match(slug):
        raise ValueError(f"Unexpected slug format: {slug!r}")
    return slug
```

This prevents path traversal attacks (e.g., `../../../etc/passwd` → `passwd`, then rejected as invalid).

**Do not disable or weaken this check.** It protects against:
- Path traversal (`../../etc/passwd`)
- Null bytes (`slug\x00.json`)
- Uppercase characters (not in URL slugs)
- Special characters and command injection

See `tests/test_scraper.py::TestSlugFromUrl` for comprehensive test coverage.

### Session Lifecycle

The scraper reuses an HTTP session for efficiency:

```python
def run(output_dir: Path = OUTPUT_DIR, delay: float = REQUEST_DELAY) -> None:
    with requests.Session() as session:
        _scrape(session, output_dir, delay)
```

The session is:
- Created once per run
- Passed to all fetch operations
- Automatically closed after scraping completes
- Configured with a safe User-Agent header

**Do not** open new sessions for every request — it defeats connection pooling.

### No Secrets in Code

Do not hardcode:
- API keys or tokens
- Credentials
- Sensitive URLs or domains

All configuration (timeouts, delays, base URL) should be constants at the top of the file:

```python
BASE_URL = "https://allaboutberlin.com"
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 1.0
```

## Testing Best Practices

### Arrange-Act-Assert (AAA) Pattern

```python
def test_parses_guide_slug():
    # Arrange
    url = "https://allaboutberlin.com/guides/find-a-flat-in-berlin"
    
    # Act
    slug = slug_from_url(url)
    
    # Assert
    assert slug == "find-a-flat-in-berlin"
```

### Use Fixtures for Common Data

The test file includes HTML stubs for common scenarios:

```python
GUIDES_INDEX_HTML = """..."""  # Minimal index page
GUIDE_PAGE_HTML = """..."""     # Sample guide with h2, h3, content
GUIDE_PAGE_NO_H2_HTML = """..."""  # Guide without H2 headers
```

Use these instead of creating new test data.

### Mock External Calls

Use the `responses` library to mock HTTP requests (no live network calls):

```python
@responses_lib.activate
def test_fetches_and_parses(tmp_path):
    # Mock the index page
    responses_lib.add(
        responses_lib.GET,
        f"{BASE_URL}{GUIDES_PATH}",
        body=GUIDES_INDEX_HTML,
        status=200,
    )
    
    # Mock a guide page
    responses_lib.add(
        responses_lib.GET,
        f"{BASE_URL}/guides/find-a-flat-in-berlin",
        body=GUIDE_PAGE_HTML,
        status=200,
    )
    
    # Act
    run(output_dir=tmp_path, delay=0)
    
    # Assert
    assert (tmp_path / "guides.json").exists()
```

### Test Error Paths

Verify that errors are handled gracefully:

```python
@responses_lib.activate
def test_continues_on_guide_fetch_error(tmp_path):
    # Mock failed fetch
    responses_lib.add(responses_lib.GET, url, status=500)
    
    # Act
    run(output_dir=tmp_path, delay=0)
    
    # Assert
    # Scraper still produces guides.json, but failed guide has sections_count=0
    assert (tmp_path / "guides.json").exists()
    index = json.loads((tmp_path / "guides.json").read_text())
    failed = next(e for e in index if e["guide"] == "failed-guide")
    assert failed["sections_count"] == 0
```

## Making Changes

### Adding a New Feature

1. Write a test first (TDD approach)
   ```python
   def test_new_feature():
       # Describe what should happen
       result = new_function()
       assert result == expected
   ```

2. Run the test — it should fail (RED)
   ```bash
   uv run pytest tests/test_scraper.py::test_new_feature
   ```

3. Implement the feature
   ```python
   def new_function():
       return expected
   ```

4. Run the test — it should pass (GREEN)
   ```bash
   uv run pytest tests/test_scraper.py::test_new_feature
   ```

5. Refactor and verify coverage still >= 80%
   ```bash
   uv run pytest --cov=scraper --cov-report=term-missing
   ```

### Fixing a Bug

1. Write a test that reproduces the bug (should fail)
2. Fix the implementation
3. Verify the test passes
4. Run full test suite to ensure no regressions

## Troubleshooting

### Tests Fail with ImportError

Ensure dependencies are installed:
```bash
uv sync
```

### Coverage Below 80%

Check which lines are not covered:
```bash
uv run pytest --cov=scraper --cov-report=term-missing | grep ">"
```

This shows untested lines marked with `>`. Add tests to cover them.

### HTTP Mocking Not Working

Ensure `responses_lib.activate` is used as a decorator or context manager:
```python
@responses_lib.activate  # This is required
def test_fetches_html():
    responses_lib.add(...)
    ...
```

### Slug Validation Errors in Tests

If a test fails with "Unexpected slug format," verify the slug matches `^[a-z0-9][a-z0-9\-]{0,99}$`:
- Only lowercase letters, digits, and hyphens
- Must start with a letter or digit
- No more than 100 characters

## Questions?

Refer to the main README.md for project overview and usage instructions.
