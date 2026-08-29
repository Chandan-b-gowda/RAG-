"""
Robustness helpers for LLM calls.

Two things make experiments reliable and cheap:

1. Backoff on rate limits: the Gemini API returns HTTP 429 when you exceed your
   per-minute or per-day quota. Instead of crashing a long experiment, we wait the
   amount the API asks for and retry. This means a temporary limit just slows the
   run instead of losing all progress.

2. (Caching lives in the experiment layer — see experiments/evaluate.py.)
"""

from __future__ import annotations

import re
import time

# These exceptions signal "you are going too fast / out of quota".
try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
    _RATE_LIMIT_ERRORS: tuple = (ResourceExhausted, ServiceUnavailable)
except Exception:  # pragma: no cover - defensive
    _RATE_LIMIT_ERRORS = ()


def _suggested_delay(error: Exception, default: float) -> float:
    """Read the 'retry in Ns' hint the Gemini API includes in its error text."""
    match = re.search(r"retry.*?(\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0  # small safety margin
    return default


def invoke_with_backoff(llm, prompt, max_retries: int = 5, base_delay: float = 5.0):
    """
    Call llm.invoke(prompt), retrying politely on rate-limit errors.

    Returns the model's text content (a string). Raises the last error if every
    retry is exhausted (e.g. a hard daily cap that won't clear soon).
    """
    attempt = 0
    while True:
        try:
            return llm.invoke(prompt).content
        except _RATE_LIMIT_ERRORS as err:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = _suggested_delay(err, base_delay * attempt)
            print(f"    [rate-limit] waiting {delay:.0f}s then retrying "
                  f"(attempt {attempt}/{max_retries})...")
            time.sleep(delay)
