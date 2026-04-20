"""Sycophancy-prior surfacing tests."""

import pytest
from pydantic import ValidationError

from ai_agent_council.config import AgentConfig
from ai_agent_council.models import Message, Phase, Role
from ai_agent_council.prompts import render_synthesis_prompt


def _critic_msg(name: str, content: str = "critique") -> Message:
    return Message(
        role=Role.CRITIC,
        agent_name=name,
        phase=Phase.CRITIQUE,
        content=content,
        model="ollama/x",
        temperature=0.2,
    )


def _own_draft() -> Message:
    return Message(
        role=Role.IDEATOR,
        agent_name="Muse",
        phase=Phase.DIVERGENT,
        content="my answer",
        model="ollama/y",
        temperature=0.9,
    )


def test_sycophancy_prior_validated_range() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(
            name="X",
            role=Role.IDEATOR,
            model="ollama/x",
            temperature=0.5,
            sycophancy_prior=1.5,
        )
    with pytest.raises(ValidationError):
        AgentConfig(
            name="X",
            role=Role.IDEATOR,
            model="ollama/x",
            temperature=0.5,
            sycophancy_prior=-0.1,
        )


def test_sycophancy_prior_defaults_to_none() -> None:
    cfg = AgentConfig(
        name="X", role=Role.IDEATOR, model="ollama/x", temperature=0.5
    )
    assert cfg.sycophancy_prior is None


def test_synthesis_prompt_without_priors_omits_header() -> None:
    out = render_synthesis_prompt("task", _own_draft(), [_critic_msg("Hawk")])
    assert "sycophancy prior" not in out.lower()


def test_synthesis_prompt_with_priors_annotates_critics() -> None:
    out = render_synthesis_prompt(
        "task",
        _own_draft(),
        [_critic_msg("Hawk"), _critic_msg("Judge")],
        sycophancy_priors={"Hawk": 0.2, "Judge": 0.8},
    )
    assert "sycophancy prior: 0.20" in out
    assert "sycophancy prior: 0.80" in out
    # Guidance line should appear exactly once.
    assert out.count("rarely defers") == 1


def test_synthesis_prompt_partial_priors() -> None:
    """Only annotate critics who have a prior set; others render unchanged."""
    out = render_synthesis_prompt(
        "task",
        _own_draft(),
        [_critic_msg("Hawk"), _critic_msg("Judge")],
        sycophancy_priors={"Hawk": 0.3},
    )
    assert "From Hawk (sycophancy prior: 0.30)" in out
    assert "From Judge" in out
    # Judge gets no prior annotation.
    assert "From Judge (sycophancy" not in out


def test_synthesis_prompt_handles_failed_critique_with_prior() -> None:
    failed = Message(
        role=Role.CRITIC,
        agent_name="Hawk",
        phase=Phase.CRITIQUE,
        content="",
        model="ollama/x",
        temperature=0.2,
        error="boom",
    )
    out = render_synthesis_prompt(
        "task", _own_draft(), [failed], sycophancy_priors={"Hawk": 0.4}
    )
    assert "sycophancy prior: 0.40" in out
    assert "FAILED" in out
