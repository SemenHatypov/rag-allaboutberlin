"""Tests for app.py helpers that don't require a running Streamlit session."""

from app import FRIENDLY_ERROR, escape_caption, friendly_error


class TestFriendlyError:
    def test_returns_neutral_message(self):
        assert friendly_error(ValueError("boom")) == FRIENDLY_ERROR

    def test_does_not_leak_raw_exception_text(self):
        secret = "Invalid API key sk-abc123SECRET"
        message = friendly_error(RuntimeError(secret))
        assert "sk-abc123SECRET" not in message
        assert "Invalid API key" not in message


class TestEscapeCaption:
    def test_escapes_leading_number_dot(self):
        assert escape_caption("5. Apply for the apartment") == "5\\. Apply for the apartment"

    def test_leaves_normal_heading_untouched(self):
        assert escape_caption("How to register") == "How to register"

    def test_only_escapes_at_the_start(self):
        # a number-dot later in the joined caption is not a list marker
        assert escape_caption("Intro · 5. Apply") == "Intro · 5. Apply"
