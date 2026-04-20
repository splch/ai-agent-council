"""End-to-end tests for the tool-calling plumbing.

The FakeLLM, when given `tools`, invokes each tool once with empty args and records the
results in the returned meta under `tool_calls_made`. Real model-driven tool-call tests
hit live providers under `COUNCIL_OLLAMA_IT=1`.
"""

from ai_agent_council.agent import Agent
from ai_agent_council.config import AgentConfig
from ai_agent_council.models import Phase, Role

from .fakes import FakeLLM


async def test_finisher_with_tools_records_tool_calls(fake_llm: FakeLLM) -> None:
    fake_llm.default = "polished"
    agent = Agent.from_config(
        AgentConfig(
            name="Polish",
            role=Role.FINISHER,
            model="ollama/mistral-small",
            temperature=0.1,
            tools=["current_time", "calculate"],
        )
    )
    msg = await agent.respond("finish this", phase=Phase.FINISHING)
    assert msg.content == "polished"
    assert [tc.name for tc in msg.tool_calls] == ["current_time", "calculate"]
    # calculate called with no args → TypeError → captured as a tool error
    calc_call = msg.tool_calls[1]
    assert calc_call.error is not None


async def test_agent_without_tools_has_empty_tool_calls(fake_llm: FakeLLM) -> None:
    fake_llm.default = "ok"
    agent = Agent.from_config(
        AgentConfig(
            name="Muse",
            role=Role.IDEATOR,
            model="ollama/gemma3:2b",
            temperature=0.9,
        )
    )
    msg = await agent.respond("think", phase=Phase.DIVERGENT)
    assert msg.tool_calls == []


async def test_agent_config_resolves_unknown_tool_at_construction(fake_llm: FakeLLM) -> None:
    import pytest

    with pytest.raises(KeyError, match="unknown tool"):
        Agent.from_config(
            AgentConfig(
                name="X",
                role=Role.FINISHER,
                model="ollama/mistral-small",
                temperature=0.1,
                tools=["no_such_tool"],
            )
        )
