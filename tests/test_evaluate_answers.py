"""Tests for evaluate_answers.py — the pass/fail decision logic (no LLM calls)."""

from evaluate_answers import CaseVerdict, passed, source_precision_ok


def verdict(refused=False, grounded=True, on_topic=True, scope_caveat=False):
    return CaseVerdict(
        refused=refused,
        grounded=grounded,
        on_topic=on_topic,
        scope_caveat=scope_caveat,
        reasoning="",
    )


class TestFactual:
    def test_grounded_on_topic_answer_passes(self):
        assert passed({"type": "factual"}, verdict(), refused_exact=False)

    def test_refusal_fails(self):
        assert not passed({"type": "factual"}, verdict(), refused_exact=True)

    def test_ungrounded_fails(self):
        assert not passed({"type": "factual"}, verdict(grounded=False), refused_exact=False)

    def test_off_topic_fails(self):
        assert not passed({"type": "factual"}, verdict(on_topic=False), refused_exact=False)


class TestMultiturn:
    def test_on_topic_passes(self):
        assert passed({"type": "multiturn"}, verdict(), refused_exact=False)

    def test_topic_drift_fails(self):
        assert not passed({"type": "multiturn"}, verdict(on_topic=False), refused_exact=False)

    def test_refusal_fails(self):
        assert not passed({"type": "multiturn"}, verdict(), refused_exact=True)


class TestOfftopic:
    def test_exact_refusal_passes(self):
        assert passed({"type": "offtopic"}, verdict(refused=False), refused_exact=True)

    def test_judge_refusal_passes(self):
        assert passed({"type": "offtopic"}, verdict(refused=True), refused_exact=False)

    def test_substantive_answer_fails(self):
        assert not passed({"type": "offtopic"}, verdict(refused=False), refused_exact=False)


class TestWrongCity:
    def test_scope_caveat_passes(self):
        assert passed({"type": "wrong_city"}, verdict(scope_caveat=True), refused_exact=False)

    def test_refusal_passes(self):
        assert passed({"type": "wrong_city"}, verdict(scope_caveat=False), refused_exact=True)

    def test_confident_answer_without_caveat_fails(self):
        assert not passed({"type": "wrong_city"}, verdict(scope_caveat=False), refused_exact=False)


class TestSourcePrecision:
    def test_no_expectation_is_always_ok(self):
        assert source_precision_ok({"type": "factual"}, ["a", "b", "c"])

    def test_cited_none_is_ok(self):
        assert source_precision_ok({"single_guide": True, "expected_guide": "x"}, None)

    def test_single_guide_exact_match_passes(self):
        case = {"single_guide": True, "expected_guide": "deutschland-ticket"}
        assert source_precision_ok(case, ["deutschland-ticket"])

    def test_single_guide_over_citation_fails(self):
        case = {"single_guide": True, "expected_guide": "deutschland-ticket"}
        assert not source_precision_ok(case, ["deutschland-ticket", "pfand-bottles"])

    def test_single_guide_wrong_guide_fails(self):
        case = {"single_guide": True, "expected_guide": "deutschland-ticket"}
        assert not source_precision_ok(case, ["public-transit"])

    def test_min_sources_met_with_expected_present_passes(self):
        case = {"min_sources": 2, "expected_guides": ["legal-insurance", "insurance"]}
        assert source_precision_ok(case, ["legal-insurance", "insurance", "taxes"])

    def test_min_sources_over_trimmed_fails(self):
        case = {"min_sources": 2, "expected_guides": ["legal-insurance", "insurance"]}
        assert not source_precision_ok(case, ["legal-insurance"])

    def test_min_sources_expected_guide_dropped_fails(self):
        case = {"min_sources": 2, "expected_guides": ["legal-insurance", "insurance"]}
        assert not source_precision_ok(case, ["legal-insurance", "taxes"])


class TestPassedThreadsSourcePrecision:
    def test_factual_fails_when_over_cited(self):
        case = {"type": "factual", "single_guide": True, "expected_guide": "deutschland-ticket"}
        assert not passed(case, verdict(), refused_exact=False, cited=["deutschland-ticket", "pfand-bottles"])

    def test_factual_passes_when_precise(self):
        case = {"type": "factual", "single_guide": True, "expected_guide": "deutschland-ticket"}
        assert passed(case, verdict(), refused_exact=False, cited=["deutschland-ticket"])

    def test_factual_without_cited_ignores_precision(self):
        # Back-compat: existing callers/tests that omit cited keep the old behaviour.
        case = {"type": "factual", "single_guide": True, "expected_guide": "deutschland-ticket"}
        assert passed(case, verdict(), refused_exact=False)
