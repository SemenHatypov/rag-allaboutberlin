"""Tests for app.py helpers that don't require a running Streamlit session."""

from app import FRIENDLY_ERROR, friendly_error


class TestFriendlyError:
    def test_returns_neutral_message(self):
        assert friendly_error(ValueError("boom")) == FRIENDLY_ERROR

    def test_does_not_leak_raw_exception_text(self):
        secret = "Invalid API key sk-abc123SECRET"
        message = friendly_error(RuntimeError(secret))
        assert "sk-abc123SECRET" not in message
        assert "Invalid API key" not in message
