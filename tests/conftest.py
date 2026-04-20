"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from ai_agent_council.config import AgentConfig, CouncilConfig
from ai_agent_council.models import Role

from .fakes import FakeLLM, install_fake_llm


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Monkeypatched FakeLLM that captures every call."""
    fake = FakeLLM()
    install_fake_llm(monkeypatch, fake)
    return fake


@pytest.fixture
def minimal_council_config() -> CouncilConfig:
    """A valid 4-agent roster (gemma / deepseek / phi / claude)."""
    return CouncilConfig(
        agents=[
            AgentConfig(name="Muse", role=Role.IDEATOR, model="ollama/gemma3:2b", temperature=0.9),
            AgentConfig(
                name="Judge",
                role=Role.REASONER,
                model="ollama/deepseek-r1:1.5b",
                temperature=0.4,
            ),
            AgentConfig(
                name="Hawk",
                role=Role.CRITIC,
                model="ollama/phi4-mini:3.8b",
                temperature=0.2,
            ),
            AgentConfig(
                name="Scribe",
                role=Role.ORCHESTRATOR,
                model="anthropic/claude-haiku-4-5-20251001",
                temperature=0.3,
            ),
        ],
    )


@pytest.fixture
def full_council_config() -> CouncilConfig:
    """A valid 6-agent Belbin roster, six distinct families."""
    return CouncilConfig(
        agents=[
            AgentConfig(name="Muse", role=Role.IDEATOR, model="ollama/gemma3:4b", temperature=0.9),
            AgentConfig(
                name="Judge",
                role=Role.REASONER,
                model="ollama/deepseek-r1:7b",
                temperature=0.4,
            ),
            AgentConfig(
                name="Adept",
                role=Role.SPECIALIST,
                model="ollama/qwen2.5-coder:7b",
                temperature=0.5,
            ),
            AgentConfig(
                name="Hawk",
                role=Role.CRITIC,
                model="ollama/phi4-mini:3.8b",
                temperature=0.2,
            ),
            AgentConfig(
                name="Polish",
                role=Role.FINISHER,
                model="ollama/mistral-small",
                temperature=0.1,
            ),
            AgentConfig(
                name="Scribe",
                role=Role.ORCHESTRATOR,
                model="anthropic/claude-haiku-4-5-20251001",
                temperature=0.3,
            ),
        ],
    )
