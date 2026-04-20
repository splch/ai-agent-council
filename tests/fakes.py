"""Test doubles for the LLM layer."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ai_agent_council.exceptions import LLMError
from ai_agent_council.llm import CompletionMeta
from ai_agent_council.tools import Tool


@dataclass(frozen=True)
class FakeCall:
    model: str
    system: str
    user: str
    temperature: float
    max_tokens: int
    json_mode: bool


FakeResponder = Callable[[FakeCall], str | LLMError]


@dataclass
class FakeLLM:
    """Scriptable stand-in for `ai_agent_council.llm.complete`.

    Two modes:
        * `responder` — a callable that produces a response per call (or raises by returning
          an LLMError instance).
        * `by_role` — dict keyed on role/agent_name returning canned content.

    Records every call in `.calls` so tests can assert what the agent saw.
    """

    responder: FakeResponder | None = None
    by_name: dict[str, str] = field(default_factory=dict)
    default: str = "fake response"
    calls: list[FakeCall] = field(default_factory=list)
    raise_for: set[str] = field(default_factory=set)
    cost_per_call: float = 0.0001

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
        json_mode: bool = False,
        stream_handler: Callable[[str], None] | None = None,
        tools: list[Tool] | None = None,
        max_tool_iterations: int = 5,
    ) -> tuple[str, CompletionMeta]:
        call = FakeCall(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        self.calls.append(call)
        if model in self.raise_for:
            raise LLMError(f"synthetic failure for {model!r}")
        content: str | LLMError
        if self.responder is not None:
            content = self.responder(call)
        else:
            content = self.by_name.get(agent_from_system(system), self.default)
        if isinstance(content, LLMError):
            raise content
        if stream_handler is not None:
            # Emit three deterministic chunks so tests can assert the handler was driven.
            third = max(1, len(content) // 3)
            for start in (0, third, 2 * third):
                piece = content[start : start + third] if start < len(content) else ""
                if piece:
                    stream_handler(piece)
        meta: CompletionMeta = {
            "tokens_in": 1,
            "tokens_out": 1,
            "latency_ms": 1,
            "cost_usd": self.cost_per_call,
        }
        # If tools are wired in, drive them once with empty args so a test can assert the
        # loop plumbs through. Real model-driven tool-call tests bypass the fake and hit
        # litellm directly under COUNCIL_OLLAMA_IT.
        if tools:
            recorded = []
            for tool in tools:
                try:
                    result = str(tool.fn())
                    err = None
                except TypeError as e:
                    result, err = f"error: {e}", str(e)
                recorded.append(
                    {"name": tool.name, "arguments": {}, "result": result, "error": err}
                )
            meta["tool_calls_made"] = recorded
        return content, meta


def agent_from_system(system: str) -> str:
    """Best-effort: pick the agent name out of the first line of the system prompt.
    System prompts all begin with ``You are {name}, the ...``."""
    first_line = system.strip().splitlines()[0] if system.strip() else ""
    head, _, _ = first_line.partition(",")
    prefix = "You are "
    return head[len(prefix) :].strip() if head.startswith(prefix) else first_line


def install_fake_llm(monkeypatch: Any, fake: FakeLLM) -> None:
    """Monkeypatch ``ai_agent_council.llm.complete``. agent.py does ``from . import llm``
    which binds the same module object, so this one setattr is visible through both paths."""
    import ai_agent_council.llm as llm_mod

    monkeypatch.setattr(llm_mod, "complete", fake.complete)
