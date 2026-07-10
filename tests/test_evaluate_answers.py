"""Tests for evaluate_answers.py — the pass/fail decision logic (no LLM calls)."""

from evaluate_answers import CaseVerdict, passed


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
