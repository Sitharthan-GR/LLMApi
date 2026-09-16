"""Retry 429/5xx with exponential backoff. Fail fast on other 4xx."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetryableError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def with_retries(fn: Callable[[], T], attempts: int = 4, base_delay: float = 0.6) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            last = exc
            if i == attempts - 1:
                break
            delay = exc.retry_after if exc.retry_after is not None else base_delay * (2**i)
            delay += random.random() * 0.2
            time.sleep(delay)
        except RuntimeError as exc:
            text = str(exc)
            if "HTTP 429" in text or "HTTP 5" in text:
                last = exc
                if i == attempts - 1:
                    break
                time.sleep(base_delay * (2**i) + random.random() * 0.2)
                continue
            raise
    raise last or RuntimeError("retry failed")
