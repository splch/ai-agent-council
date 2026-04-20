"""Thin LiteLLM wrapper.

Exposes a single async function, `complete`, that the rest of the package calls. Callers never
import `litellm` directly — this is the seam for tests (monkeypatched with a FakeLLM).
"""

import time
from collections.abc import Callable
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

# Per-token streaming handler. Called once per content delta as it arrives.
StreamHandler = Callable[[str], None]


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
    stream_handler: StreamHandler | None = None,
) -> tuple[str, CompletionMeta]:
    """Call an LLM via LiteLLM. Returns (content, meta).

    When `stream_handler` is given, `stream=True` is requested from LiteLLM and each content
    delta is passed to the handler as it arrives. The full content is still accumulated and
    returned in the same tuple shape — callers that want both streaming and a final value
    get both.

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

    if stream_handler is not None:
        return await _complete_streaming(kwargs, messages, stream_handler)

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


async def _complete_streaming(
    kwargs: dict[str, Any], messages: list[dict[str, str]], on_token: StreamHandler
) -> tuple[str, CompletionMeta]:
    """Streaming variant: emit each content delta via `on_token`, return assembled content.

    Usage info arrives in a trailing chunk when `include_usage=True` (OpenAI-compatible
    providers honor it; others may not). Cost is computed from the rebuilt response via
    `litellm.stream_chunk_builder`; when the provider doesn't give usage back, cost is 0.
    """
    kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic()
    try:
        resp_stream = await litellm.acompletion(**kwargs)
    except litellm.exceptions.Timeout as e:
        raise LLMTimeoutError(str(e)) from e
    except litellm.exceptions.RateLimitError as e:
        raise LLMRateLimitError(str(e)) from e
    except litellm.exceptions.APIError as e:
        raise LLMError(str(e)) from e

    chunks: list[Any] = []
    parts: list[str] = []
    tokens_in: int | None = None
    tokens_out: int | None = None
    async for chunk in resp_stream:
        chunks.append(chunk)
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0].delta, "content", None)
            if delta:
                parts.append(delta)
                on_token(delta)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            tokens_in = getattr(usage, "prompt_tokens", tokens_in) or tokens_in
            tokens_out = getattr(usage, "completion_tokens", tokens_out) or tokens_out

    latency_ms = int((time.monotonic() - t0) * 1000)
    content = "".join(parts)
    try:
        assembled = litellm.stream_chunk_builder(chunks, messages=messages)
        cost = _completion_cost(assembled)
    except Exception:
        cost = 0.0
    meta: CompletionMeta = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "cost_usd": cost,
    }
    return content, meta
