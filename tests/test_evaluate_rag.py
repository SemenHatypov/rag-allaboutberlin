"""Tests for evaluate_rag.py — citation scoring."""

from evaluate_rag import score_citation


class TestScoreCitation:
    def test_hit_at_rank_1(self):
        assert score_citation(["anmeldung", "schufa"], "anmeldung") == (True, 1)

    def test_hit_at_rank_3(self):
        assert score_citation(["schufa", "taxes", "anmeldung"], "anmeldung") == (True, 3)

    def test_miss(self):
        assert score_citation(["schufa", "taxes"], "anmeldung") == (False, None)

    def test_empty_citations(self):
        assert score_citation([], "anmeldung") == (False, None)
