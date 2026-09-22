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
