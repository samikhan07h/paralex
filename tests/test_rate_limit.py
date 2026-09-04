"""
Tests for evaluation/rate_limit.py

Constructs real groq.RateLimitError instances (with a minimal fake httpx
response) to test backoff behavior precisely, without needing to trigger
an actual rate limit against the live API.
"""

import time
import httpx
import pytest
from unittest.mock import MagicMock

from groq import RateLimitError
from evaluation.rate_limit import call_with_backoff, _parse_suggested_wait_seconds


def _make_rate_limit_error(message: str) -> RateLimitError:
    fake_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    fake_response = httpx.Response(429, request=fake_request)
    return RateLimitError(message, response=fake_response, body=None)


# --- _parse_suggested_wait_seconds ---

def test_parse_suggested_wait_handles_milliseconds():
    msg = "Rate limit reached. Please try again in 224.999999ms. Need more tokens?"
    assert _parse_suggested_wait_seconds(msg) == pytest.approx(0.225, abs=0.001)


def test_parse_suggested_wait_handles_seconds():
    msg = "Rate limit reached. Please try again in 1.5s."
    assert _parse_suggested_wait_seconds(msg) == pytest.approx(1.5)


def test_parse_suggested_wait_returns_none_when_unparseable():
    assert _parse_suggested_wait_seconds("Some unrelated error message.") is None


# --- call_with_backoff ---

def test_call_with_backoff_returns_result_on_first_success():
    fn = MagicMock(return_value="ok")
    result = call_with_backoff(fn)
    assert result == "ok"
    assert fn.call_count == 1


def test_call_with_backoff_retries_after_rate_limit_then_succeeds(monkeypatch):
    # Avoid actually sleeping during the test.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    fn = MagicMock(side_effect=[
        _make_rate_limit_error("Please try again in 10ms."),
        "ok",
    ])

    result = call_with_backoff(fn, max_retries=3)
    assert result == "ok"
    assert fn.call_count == 2


def test_call_with_backoff_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    fn = MagicMock(side_effect=_make_rate_limit_error("Please try again in 10ms."))

    with pytest.raises(RateLimitError):
        call_with_backoff(fn, max_retries=3)
    assert fn.call_count == 3


def test_call_with_backoff_uses_suggested_wait_time(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    fn = MagicMock(side_effect=[
        _make_rate_limit_error("Please try again in 2s."),
        "ok",
    ])

    call_with_backoff(fn, max_retries=3)
    # Should sleep ~2.5s (2s suggested + 0.5s buffer), not a generic
    # exponential-backoff guess.
    assert sleep_calls[0] == pytest.approx(2.5)
