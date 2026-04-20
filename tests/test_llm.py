"""Exception translation at the LiteLLM seam.

Unit tests here exercise the real `ai_agent_council.llm.complete` against a patched
`litellm.acompletion`, not via the FakeLLM (which replaces `complete` wholesale). The
point is to pin down the contract that every litellm-layer exception becomes an
`LLMError` — not a bare provider exception that would escape `Agent.respond` and get
wrapped by TaskGroup into a BaseExceptionGroup.
"""

import litellm
import pytest

from ai_agent_council.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from ai_agent_council.llm import complete


def _patch_acompletion(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    """Make `litellm.acompletion` raise `exc` on first call (no retry cost for non-
    Timeout/RateLimit types because tenacity's retry_if_exception_type excludes them)."""

    async def _raise(**kwargs: object) -> object:
        raise exc

    monkeypatch.setattr(litellm, "acompletion", _raise)


async def test_api_connection_error_is_wrapped_as_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the 'model not found' crash: litellm raises APIConnectionError when
    Ollama rejects an unpulled model. Before the fix this escaped as-is and TaskGroup
    wrapped it into a BaseExceptionGroup. Now it's an LLMError caught by Agent.respond."""
    _patch_acompletion(
        monkeypatch,
        litellm.exceptions.APIConnectionError(
            message="OllamaException - model 'foo' not found",
            llm_provider="ollama",
            model="foo",
        ),
    )
    with pytest.raises(LLMError, match="model 'foo' not found"):
        await complete(
            model="ollama/foo",
            system="sys",
            user="usr",
            temperature=0.0,
            max_tokens=10,
            timeout_s=5.0,
        )


async def test_timeout_surfaces_as_llm_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_acompletion(
        monkeypatch,
        litellm.exceptions.Timeout(
            message="timed out",
            llm_provider="openai",
            model="gpt-x",
        ),
    )
    # Tenacity will retry up to 3 times; that's fine — eventually it re-raises.
    with pytest.raises(LLMTimeoutError):
        await complete(
            model="openai/gpt-x",
            system="sys",
            user="usr",
            temperature=0.0,
            max_tokens=10,
            timeout_s=5.0,
        )


async def test_rate_limit_surfaces_as_llm_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_acompletion(
        monkeypatch,
        litellm.exceptions.RateLimitError(
            message="slow down",
            llm_provider="openai",
            model="gpt-x",
        ),
    )
    with pytest.raises(LLMRateLimitError):
        await complete(
            model="openai/gpt-x",
            system="sys",
            user="usr",
            temperature=0.0,
            max_tokens=10,
            timeout_s=5.0,
        )


async def test_cancelled_error_propagates_not_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """asyncio.CancelledError is a BaseException on 3.8+, so our `except Exception` does
    NOT catch it. Cooperative cancellation must still work through the LiteLLM seam."""
    import asyncio

    _patch_acompletion(monkeypatch, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await complete(
            model="openai/gpt-x",
            system="sys",
            user="usr",
            temperature=0.0,
            max_tokens=10,
            timeout_s=5.0,
        )
