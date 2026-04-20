"""Internal timing helper — used by phases, llm, and evals for wall-clock measurement."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def elapsed_ms() -> Iterator[Callable[[], int]]:
    """Yield a closure that returns milliseconds elapsed since the context entered.

    Replaces the repeated `t0 = time.monotonic(); ... int((time.monotonic() - t0) * 1000)`
    pattern across phase runners, the LLM wrapper, and the eval harness.

    Usage:
        with elapsed_ms() as ms:
            await do_work()
        return Result(..., elapsed_ms=ms())
    """
    t0 = time.monotonic()
    yield lambda: int((time.monotonic() - t0) * 1000)
