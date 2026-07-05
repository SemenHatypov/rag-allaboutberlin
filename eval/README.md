# eval/

Versioned evaluation inputs — data that must stay reproducible across runs,
as opposed to `output/`, which holds ephemeral run results (`rag-eval-*.csv`,
cost estimates) that differ on every run and are not committed.

## ground_truth/

Article-level ground truth used by `evaluate_search.py` and `evaluate_rag.py`:
pairs of (question, correct guide) generated from the knowledge base content
by `generate_ground_truth.py`.

Generate it once and commit the result:

```bash
uv run python generate_ground_truth.py
```

This writes:

- `ground-truth-guides.csv` / `.json` — the dataset itself (`question, guide, guide_name, url`)
- `metadata.json` — provenance: model, prompt version, and a content fingerprint
  of the `output/json/` corpus used to generate it (see `corpus_fingerprint()`
  in `generate_ground_truth.py`)

Regenerate and re-commit whenever the underlying guide content changes
meaningfully — review the diff before committing, since LLM-generated
questions are not byte-for-byte deterministic between runs.
`tests/test_ground_truth_contract.py` warns when the corpus fingerprint no
longer matches `metadata.json`, and fails if a `guide` in the dataset no
longer exists in `output/json/`.
