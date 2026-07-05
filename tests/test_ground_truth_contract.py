"""Contract test: the committed ground-truth dataset matches the real corpus.

Unlike test_data_contract.py (which checks output/json/ against what the app
depends on), this checks eval/ground_truth/ against output/json/ — it catches
the ground truth drifting out of sync after guides are re-scraped, renamed,
or removed, without requiring a fresh (paid) regeneration on every change.
"""

import json
import warnings

import pandas as pd
import pytest

from generate_ground_truth import GROUND_TRUTH_DIR, corpus_fingerprint
from ingest import load_documents

CSV_PATH = GROUND_TRUTH_DIR / "ground-truth-guides.csv"
METADATA_PATH = GROUND_TRUTH_DIR / "metadata.json"
REQUIRED_COLUMNS = ["question", "guide", "guide_name", "url"]

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason=f"{CSV_PATH} not generated yet — run `uv run python generate_ground_truth.py`",
)


class TestGroundTruthContract:
    def test_has_required_columns_and_no_nulls(self):
        df = pd.read_csv(CSV_PATH)
        assert list(df.columns) == REQUIRED_COLUMNS
        assert not df.isnull().any().any()

    def test_is_nonempty(self):
        df = pd.read_csv(CSV_PATH)
        assert len(df) > 0

    def test_every_guide_exists_in_current_corpus(self):
        df = pd.read_csv(CSV_PATH)
        known_guides = {doc["guide"] for doc in load_documents()}
        orphaned = set(df["guide"]) - known_guides
        assert not orphaned, f"ground truth references guides no longer in output/json/: {orphaned}"

    def test_corpus_fingerprint_matches_metadata(self):
        if not METADATA_PATH.exists():
            pytest.skip(f"{METADATA_PATH} missing — generated before provenance tracking was added")

        metadata = json.loads(METADATA_PATH.read_text())
        current_fingerprint = corpus_fingerprint(load_documents())
        if metadata.get("source_corpus_hash") != current_fingerprint:
            warnings.warn(
                "output/json/ has changed since the ground truth was generated — "
                "consider regenerating with `uv run python generate_ground_truth.py`",
                stacklevel=1,
            )
