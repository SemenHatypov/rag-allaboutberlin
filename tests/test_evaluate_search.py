"""Tests for evaluate_search.py — guide-level relevance metrics."""

from evaluate_search import hit_rate, mrr, unique_guides


class TestUniqueGuides:
    def test_dedupes_preserving_rank_order(self):
        results = [
            {"guide": "anmeldung"},
            {"guide": "schufa"},
            {"guide": "anmeldung"},
            {"guide": "taxes"},
        ]
        assert unique_guides(results) == ["anmeldung", "schufa", "taxes"]

    def test_empty_results(self):
        assert unique_guides([]) == []


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
