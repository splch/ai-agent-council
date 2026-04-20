"""Agent behaviour."""

from __future__ import annotations

from typing import Any

import pytest

from ai_agent_council.agent import Agent
from ai_agent_council.config import AgentConfig
from ai_agent_council.exceptions import LLMError
from ai_agent_council.models import Phase, Role

from .fakes import FakeCall, FakeLLM


def _cfg(**overrides: Any) -> AgentConfig:
    fields: dict[str, Any] = {
        "name": "Muse",
        "role": Role.IDEATOR,
        "model": "ollama/gemma3:2b",
        "temperature": 0.9,
    }
    fields.update(overrides)
    return AgentConfig(**fields)


def test_agent_from_config_uses_default_prompt_for_role() -> None:
    agent = Agent.from_config(_cfg())
    assert "You are Muse," in agent.system_prompt
    assert "Ideator" in agent.system_prompt


def test_agent_from_config_respects_explicit_system_prompt() -> None:
    agent = Agent.from_config(_cfg(system_prompt="Custom instructions."))
    assert agent.system_prompt == "Custom instructions."


async def test_respond_returns_message_with_meta(fake_llm: FakeLLM) -> None:
    fake_llm.by_name = {"Muse": "hello world"}
    agent = Agent.from_config(_cfg())
    msg = await agent.respond("say hi", phase=Phase.DIVERGENT)
    assert msg.agent_name == "Muse"
    assert msg.phase is Phase.DIVERGENT
    assert msg.content == "hello world"
    assert msg.error is None
    assert msg.latency_ms == 1


async def test_failed_agent_captured_not_raised(fake_llm: FakeLLM) -> None:
    """If the LLM raises, Agent.respond returns a Message with `error` set — the council run
    continues. This is a design-doc contract."""

    def responder(call: FakeCall) -> str:
        raise LLMError("provider blew up")

    fake_llm.responder = responder
    agent = Agent.from_config(_cfg())
    msg = await agent.respond("...", phase=Phase.DIVERGENT)
    assert msg.content == ""
    assert msg.error is not None
    assert "provider blew up" in msg.error


@pytest.mark.parametrize("phase", list(Phase))
async def test_respond_preserves_phase_label(fake_llm: FakeLLM, phase: Phase) -> None:
    fake_llm.default = "ok"
    agent = Agent.from_config(_cfg())
    msg = await agent.respond("...", phase=phase)
    assert msg.phase is phase
