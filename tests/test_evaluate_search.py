"""Tests for evaluate_search.py — guide-level relevance metrics."""

import pandas as pd

from evaluate_search import cited_guides, filter_known_guides, hit_rate, mrr


class TestCitedGuides:
    def test_dedupes_preserving_rank_order(self):
        results = [
            {"guide": "anmeldung"},
            {"guide": "schufa"},
            {"guide": "anmeldung"},
            {"guide": "taxes"},
        ]
        assert cited_guides(results) == ["anmeldung", "schufa", "taxes"]

    def test_empty_results(self):
        assert cited_guides([]) == []


class TestFilterKnownGuides:
    def test_drops_questions_for_missing_guides(self):
        ground_truth = pd.DataFrame(
            [
                {"question": "q1", "guide": "anmeldung"},
                {"question": "q2", "guide": "removed-guide"},
                {"question": "q3", "guide": "schufa"},
            ]
        )
        documents = [{"guide": "anmeldung"}, {"guide": "schufa"}]

        filtered = filter_known_guides(ground_truth, documents)

        assert filtered["guide"].tolist() == ["anmeldung", "schufa"]
        assert len(ground_truth) == 3  # input is not mutated

    def test_keeps_everything_when_all_guides_known(self):
        ground_truth = pd.DataFrame([{"question": "q1", "guide": "anmeldung"}])
        documents = [{"guide": "anmeldung"}]

        filtered = filter_known_guides(ground_truth, documents)

        assert len(filtered) == 1


class TestHitRate:
    def test_counts_queries_with_a_hit(self):
        relevance = [[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]]
        assert hit_rate(relevance) == 0.75

    def test_all_misses(self):
        assert hit_rate([[0, 0], [0, 0]]) == 0.0


class TestMrr:
    def test_rewards_higher_ranks(self):
        relevance = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        assert mrr(relevance) == (1.0 + 0.5 + 0.0) / 3

    def test_uses_first_hit_only(self):
        assert mrr([[0, 1, 1]]) == 0.5
