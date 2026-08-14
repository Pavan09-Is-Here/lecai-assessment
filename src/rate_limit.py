import re
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Observed free-tier cap during testing was 5 requests/minute on gemini-2.5-flash.
# Proactively spacing calls avoids most 429s; the retry loop below is the backstop
# for whatever slips through (shared quota with other traffic, jitter, etc).
MIN_INTERVAL_SECONDS = 13.0
_last_call_at: float | None = None


def throttle() -> None:
    global _last_call_at
    now = time.monotonic()
    if _last_call_at is not None:
        wait = MIN_INTERVAL_SECONDS - (now - _last_call_at)
        if wait > 0:
            time.sleep(wait)
    _last_call_at = time.monotonic()


def call_with_retry(fn: Callable[..., T], *args: Any, max_attempts: int = 5, **kwargs: Any) -> T:
    """Call `fn`, throttling proactively and retrying on rate-limit errors.

    Deliberately pattern-matches on the error message/type name instead of
    importing google.genai's internal `_gaos` exception classes -- those live
    under a private module path that isn't part of the SDK's public surface.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- intentionally broad, see docstring
            last_exc = e
            message = str(e)
            is_rate_limit = (
                "429" in message
                or "RateLimit" in type(e).__name__
                or "quota" in message.lower()
                or "too_many_requests" in message.lower()
            )
            if not is_rate_limit or attempt == max_attempts - 1:
                raise
            match = re.search(r"retry in ([\d.]+)s", message)
            delay = float(match.group(1)) + 2.0 if match else 15.0 * (attempt + 1)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
