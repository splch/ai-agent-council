"""Thin LiteLLM wrapper.

Exposes a single async function, `complete`, that the rest of the package calls. Callers never
import `litellm` directly — this is the seam for tests (monkeypatched with a FakeLLM).
"""

import json
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
from .tools import Tool

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
    tools: list[Tool] | None = None,
    max_tool_iterations: int = 5,
) -> tuple[str, CompletionMeta]:
    """Call an LLM via LiteLLM. Returns (content, meta).

    When `stream_handler` is given, `stream=True` is requested from LiteLLM and each content
    delta is passed to the handler as it arrives. The full content is still accumulated and
    returned in the same tuple shape — callers that want both streaming and a final value
    get both.

    When `tools` is given, the multi-turn tool-calling loop is used: the model may request
    tool invocations, which are executed locally and fed back for up to
    `max_tool_iterations` rounds before the final content is returned. Streaming is not
    combined with tool-calling in this release — if both are requested, tool-calling wins

    Retry note: the @retry wraps the *whole* function, so a timeout in round 3 of a tool
    loop restarts the loop from round 0. That's arguably the right semantics (the
    provider's conversation state is gone) but it re-pays the cost of earlier rounds.
    and the stream_handler is ignored.

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

    if tools:
        return await _complete_with_tools(kwargs, messages, tools, max_tool_iterations)

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


async def _complete_with_tools(
    kwargs: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[Tool],
    max_iterations: int,
) -> tuple[str, CompletionMeta]:
    """Multi-turn tool-calling loop.

    Each round sends the current message history to the model. If the model responds with
    `tool_calls`, each is executed locally via the registered tool's Python function, and
    the results are appended as `role=tool` messages for the next round. The loop exits
    when the model answers with plain content (no tool_calls) or after `max_iterations`.
    Errors from individual tool invocations are captured and fed back as the tool result,
    so the model can see and react to them.
    """
    by_name = {t.name: t for t in tools}
    kwargs = {**kwargs, "tools": [t.to_openai_schema() for t in tools]}

    recorded: list[dict[str, Any]] = []
    total_cost = 0.0
    total_in = 0
    total_out = 0
    t0 = time.monotonic()
    content = ""

    for _ in range(max_iterations):
        kwargs["messages"] = messages
        try:
            resp = await litellm.acompletion(**kwargs)
        except litellm.exceptions.Timeout as e:
            raise LLMTimeoutError(str(e)) from e
        except litellm.exceptions.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except litellm.exceptions.APIError as e:
            raise LLMError(str(e)) from e

        total_cost += _completion_cost(resp)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            total_in += getattr(usage, "prompt_tokens", 0) or 0
            total_out += getattr(usage, "completion_tokens", 0) or 0

        choices = cast(list[Any], getattr(resp, "choices", []))
        if not choices:
            raise LLMError("provider returned no choices")
        msg = choices[0].message
        content = getattr(msg, "content", None) or ""
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Record the assistant turn verbatim so the next round has full context. Some
        # providers reject assistant messages whose content is None — use an empty string.
        assistant_entry: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            break  # final answer

        for tc in tool_calls:
            name = tc.function.name
            args_str = tc.function.arguments or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError as e:
                args, result, error = {}, f"error: invalid JSON arguments: {e}", str(e)
            else:
                tool = by_name.get(name)
                if tool is None:
                    result, error = f"error: unknown tool {name!r}", f"unknown tool {name!r}"
                else:
                    try:
                        result = str(tool.fn(**args))
                        error = None
                    except Exception as e:  # tool crashes are reported back, not raised
                        result = f"error: {type(e).__name__}: {e}"
                        error = str(e)
            recorded.append({"name": name, "arguments": args, "result": result, "error": error})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result,
                }
            )

    meta: CompletionMeta = {
        "tokens_in": total_in or None,
        "tokens_out": total_out or None,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "cost_usd": total_cost,
        "tool_calls_made": recorded,
    }
    return content, meta
