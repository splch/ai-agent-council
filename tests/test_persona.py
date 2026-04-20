"""Big Five persona dial tests."""

from typing import Any

import pytest
from pydantic import ValidationError

from ai_agent_council.agent import Agent
from ai_agent_council.config import AgentConfig
from ai_agent_council.models import Role
from ai_agent_council.prompts import render_persona_block


def _cfg(**overrides: Any) -> AgentConfig:
    fields: dict[str, Any] = {
        "name": "Test",
        "role": Role.IDEATOR,
        "model": "ollama/gemma3:2b",
        "temperature": 0.5,
    }
    fields.update(overrides)
    return AgentConfig(**fields)


def test_persona_dict_includes_only_set_traits() -> None:
    cfg = _cfg(openness=0.9, agreeableness=0.2)
    assert cfg.persona_dict() == {"openness": 0.9, "agreeableness": 0.2}


def test_persona_dict_empty_when_no_traits_set() -> None:
    assert _cfg().persona_dict() == {}


def test_persona_trait_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        _cfg(openness=1.5)
    with pytest.raises(ValidationError):
        _cfg(agreeableness=-0.1)


def test_render_persona_block_empty_returns_empty_string() -> None:
    assert render_persona_block({}) == ""


def test_render_persona_block_lists_traits_numerically() -> None:
    out = render_persona_block({"openness": 0.9, "agreeableness": 0.2})
    assert "Openness: 0.90" in out
    assert "Agreeableness: 0.20" in out
    assert "0.0-1.0" in out  # scale hint


def test_agent_prompt_gains_persona_block_when_traits_set() -> None:
    agent = Agent.from_config(_cfg(openness=0.9, conscientiousness=0.3))
    assert "Personality dials" in agent.system_prompt
    assert "Openness: 0.90" in agent.system_prompt
    assert "Conscientiousness: 0.30" in agent.system_prompt


def test_agent_prompt_has_no_persona_block_when_no_traits() -> None:
    agent = Agent.from_config(_cfg())
    assert "Personality dials" not in agent.system_prompt


def test_persona_and_lessons_both_appended() -> None:
    agent = Agent.from_config(
        _cfg(openness=0.8), lessons="Recent lessons from prior runs:\n  * be careful"
    )
    assert "Personality dials" in agent.system_prompt
    assert "Recent lessons" in agent.system_prompt
    # Persona should come before lessons in the prompt.
    assert agent.system_prompt.index("Personality dials") < agent.system_prompt.index(
        "Recent lessons"
    )


def test_all_five_traits_settable() -> None:
    cfg = _cfg(
        openness=0.5,
        conscientiousness=0.6,
        extraversion=0.7,
        agreeableness=0.4,
        neuroticism=0.3,
    )
    d = cfg.persona_dict()
    assert len(d) == 5
    assert d["neuroticism"] == 0.3
