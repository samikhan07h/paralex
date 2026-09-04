"""
Retry-with-backoff utility for Groq API calls.

WHY THIS EXISTS:
Groq's free tier enforces a tokens-per-minute (TPM) limit per model. A
single question/answer round trip is cheap, but running many questions
back-to-back (exactly what evaluation does — 18 questions, each needing a
generate + judge call) can exceed that limit well before hitting any
per-request size issue. Groq's client raises groq.RateLimitError (HTTP
429) when this happens, with a suggested wait time in the error message.

Rather than letting evaluation runs fail partway through whenever the
free-tier quota is hit, we retry with exponential backoff — a standard,
expected pattern for any code calling a rate-limited external API. This is
NOT unique to evaluation: production generation calls (e.g. from the
Streamlit app, Phase 4) would benefit from the same handling, but it's
introduced here because evaluation is where the rate limit is first
actually hit in this project, at our current free-tier usage level.
"""

import re
import time
from typing import Callable, TypeVar

from groq import RateLimitError

T = TypeVar("T")


def call_with_backoff(fn: Callable[[], T], max_retries: int = 5, base_delay: float = 2.0) -> T:
    """
    Call `fn` (a zero-argument callable wrapping a Groq API call), retrying
    with exponential backoff if a RateLimitError (HTTP 429) is raised.

    We first try to honor the wait time Groq suggests in its error message
    (e.g. "Please try again in 224.999999ms") since that's more precise
    than a blind backoff; if that can't be parsed, we fall back to
    base_delay * 2^attempt, which is a standard exponential backoff shape.
    """
    last_error: RateLimitError = None

    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as e:
            last_error = e
            suggested_wait = _parse_suggested_wait_seconds(str(e))
            delay = suggested_wait if suggested_wait is not None else base_delay * (2 ** attempt)
            # Add a small buffer on top of Groq's own suggestion so we don't
            # retry right at the boundary and immediately hit the limit again.
            time.sleep(delay + 0.5)

    raise last_error


def _parse_suggested_wait_seconds(error_message: str) -> float | None:
    """
    Extract a suggested wait time from Groq's rate-limit error message,
    e.g. "Please try again in 224.999999ms" or "Please try again in 1.5s".
    Returns None if no wait time could be parsed, so the caller can fall
    back to standard exponential backoff.
    """
    ms_match = re.search(r"try again in ([\d.]+)ms", error_message)
    if ms_match:
        return float(ms_match.group(1)) / 1000.0

    s_match = re.search(r"try again in ([\d.]+)s", error_message)
    if s_match:
        return float(s_match.group(1))

    return None
