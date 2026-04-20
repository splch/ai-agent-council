"""Test doubles for the LLM layer."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ai_agent_council.exceptions import LLMError
from ai_agent_council.llm import CompletionMeta


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
            content = self.by_name.get(_agent_from_system(system), self.default)
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
        return content, meta


def _agent_from_system(system: str) -> str:
    """Best-effort: pick the agent name out of the first line of the system prompt."""
    first_line = system.strip().splitlines()[0] if system.strip() else ""
    # system prompts all begin with "You are {name}, the ..."
    head, _, _ = first_line.partition(",")
    prefix = "You are "
    return head[len(prefix) :].strip() if head.startswith(prefix) else first_line


def install_fake_llm(monkeypatch: Any, fake: FakeLLM) -> None:
    """Monkeypatch `ai_agent_council.llm.complete` and `agent.llm` to call `fake.complete`."""
    import ai_agent_council.agent as agent_mod
    import ai_agent_council.llm as llm_mod

    monkeypatch.setattr(llm_mod, "complete", fake.complete)
    # agent.py imports `llm` as a module reference, so we need to be sure the monkeypatched
    # attribute is visible from there. Since agent does `from . import llm`, setattr on the
    # llm module is sufficient — the attribute lookup goes through the module object.
    monkeypatch.setattr(agent_mod.llm, "complete", fake.complete, raising=True)
