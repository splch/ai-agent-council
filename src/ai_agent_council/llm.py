"""Thin LiteLLM wrapper.

Exposes a single async function, `complete`, that the rest of the package calls. Callers never
import `litellm` directly — this is the seam for tests (monkeypatched with a FakeLLM).
"""

import time
from typing import Any, cast

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

# Meta bag returned alongside each completion. Keys: tokens_in, tokens_out (int | None),
# latency_ms (int), cost_usd (float — zero for models LiteLLM doesn't price, e.g. Ollama).
CompletionMeta = dict[str, Any]


def _completion_cost(resp: Any) -> float:
    """Best-effort cost for a completion response in USD. LiteLLM's completion_cost is
    lossy (unknown models → 0); we swallow its exceptions for the same reason."""
    try:
        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:
        return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1.0, max=10.0),
    retry=retry_if_exception_type((LLMTimeoutError, LLMRateLimitError)),
    reraise=True,
)
async def complete(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    json_mode: bool = False,
) -> tuple[str, CompletionMeta]:
    """Call an LLM via LiteLLM. Returns (content, meta).

    Raises `LLMError` (or subclass) on failure. Retries timeouts and rate-limits with
    exponential-jitter backoff; surface other errors immediately.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_s,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.monotonic()
    try:
        resp = await litellm.acompletion(**kwargs)
    except litellm.exceptions.Timeout as e:
        raise LLMTimeoutError(str(e)) from e
    except litellm.exceptions.RateLimitError as e:
        raise LLMRateLimitError(str(e)) from e
    except litellm.exceptions.APIError as e:
        raise LLMError(str(e)) from e
    # Non-litellm exceptions (TypeError, CancelledError, …) propagate — those are
    # programming bugs or cooperative cancellation, not LLM errors.

    latency_ms = int((time.monotonic() - t0) * 1000)

    choices = cast(list[Any], getattr(resp, "choices", []))
    if not choices:
        raise LLMError("provider returned no choices")
    content = getattr(choices[0].message, "content", None) or ""

    usage = getattr(resp, "usage", None)
    meta: CompletionMeta = {
        "tokens_in": getattr(usage, "prompt_tokens", None) if usage else None,
        "tokens_out": getattr(usage, "completion_tokens", None) if usage else None,
        "latency_ms": latency_ms,
        "cost_usd": _completion_cost(resp),
    }
    return content, meta
