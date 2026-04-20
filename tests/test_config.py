"""Config validation — the design-doc invariants live here."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ai_agent_council.config import (
    AgentConfig,
    CouncilConfig,
    derive_family,
    load_council_config,
)
from ai_agent_council.exceptions import CouncilConfigError
from ai_agent_council.models import Role


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ollama/llama3.1:8b", "llama"),
        ("ollama/llama3.3:70b", "llama"),
        ("anthropic/claude-sonnet-4-6", "claude"),
        ("openai/gpt-5.1", "gpt"),
        ("ollama/qwen2.5-coder:7b", "qwen"),
        ("ollama/deepseek-r1:7b", "deepseek"),
        ("ollama/phi4-mini:3.8b", "phi"),
        ("ollama/gemma3:4b", "gemma"),
        ("ollama/mistral-small", "mistral"),
        ("gemini/gemini-2.0-pro", "gemini"),
    ],
)
def test_derive_family(model: str, expected: str) -> None:
    assert derive_family(model) == expected


def _agent(name: str, role: Role, model: str) -> AgentConfig:
    return AgentConfig(name=name, role=role, model=model, temperature=0.5)


def test_rejects_same_family_twice() -> None:
    with pytest.raises(ValidationError, match="cognitive-diversity"):
        CouncilConfig(
            agents=[
                _agent("A", Role.IDEATOR, "ollama/llama3.1:8b"),
                _agent("B", Role.CRITIC, "ollama/llama3.3:70b"),  # same family
                _agent("C", Role.REASONER, "ollama/deepseek-r1:7b"),
                _agent("D", Role.ORCHESTRATOR, "anthropic/claude-haiku-4-5-20251001"),
            ],
        )


def test_requires_orchestrator() -> None:
    with pytest.raises(ValidationError, match="orchestrator"):
        CouncilConfig(
            agents=[
                _agent("A", Role.IDEATOR, "ollama/llama3.1:8b"),
                _agent("B", Role.CRITIC, "ollama/phi4-mini:3.8b"),
                _agent("C", Role.REASONER, "ollama/deepseek-r1:7b"),
                _agent("D", Role.FINISHER, "ollama/mistral-small"),
            ],
        )


def test_rejects_two_orchestrators() -> None:
    with pytest.raises(ValidationError, match="exactly one orchestrator"):
        CouncilConfig(
            agents=[
                _agent("A", Role.IDEATOR, "ollama/llama3.1:8b"),
                _agent("B", Role.CRITIC, "ollama/phi4-mini:3.8b"),
                _agent("C", Role.ORCHESTRATOR, "ollama/deepseek-r1:7b"),
                _agent("D", Role.ORCHESTRATOR, "anthropic/claude-haiku-4-5-20251001"),
            ],
        )


def test_requires_ideator_and_critic() -> None:
    with pytest.raises(ValidationError, match="ideator"):
        CouncilConfig(
            agents=[
                _agent("A", Role.REASONER, "ollama/llama3.1:8b"),
                _agent("B", Role.CRITIC, "ollama/phi4-mini:3.8b"),
                _agent("C", Role.FINISHER, "ollama/mistral-small"),
                _agent("D", Role.ORCHESTRATOR, "anthropic/claude-haiku-4-5-20251001"),
            ],
        )
    with pytest.raises(ValidationError, match="critic"):
        CouncilConfig(
            agents=[
                _agent("A", Role.IDEATOR, "ollama/llama3.1:8b"),
                _agent("B", Role.REASONER, "ollama/phi4-mini:3.8b"),
                _agent("C", Role.FINISHER, "ollama/mistral-small"),
                _agent("D", Role.ORCHESTRATOR, "anthropic/claude-haiku-4-5-20251001"),
            ],
        )


def test_rejects_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        CouncilConfig(
            agents=[
                _agent("Dup", Role.IDEATOR, "ollama/llama3.1:8b"),
                _agent("Dup", Role.CRITIC, "ollama/phi4-mini:3.8b"),
                _agent("C", Role.REASONER, "ollama/deepseek-r1:7b"),
                _agent("D", Role.ORCHESTRATOR, "anthropic/claude-haiku-4-5-20251001"),
            ],
        )


def test_size_bounds_reject_three() -> None:
    with pytest.raises(ValidationError):
        CouncilConfig(
            agents=[
                _agent("A", Role.IDEATOR, "ollama/llama3.1:8b"),
                _agent("B", Role.CRITIC, "ollama/phi4-mini:3.8b"),
                _agent("C", Role.ORCHESTRATOR, "anthropic/claude-haiku-4-5-20251001"),
            ],
        )


def test_size_bounds_reject_nine() -> None:
    roles_cycle = [Role.IDEATOR, Role.REASONER, Role.CRITIC, Role.SPECIALIST, Role.FINISHER]
    models_cycle = [
        "ollama/gemma3:2b",
        "ollama/deepseek-r1:1.5b",
        "ollama/phi4-mini:3.8b",
        "ollama/qwen2.5:3b",
        "ollama/mistral-small",
        "ollama/llama3.2:3b",
        "anthropic/claude-haiku-4-5-20251001",
        "openai/gpt-5",
        "gemini/gemini-2.0-flash",
    ]
    with pytest.raises(ValidationError):
        CouncilConfig(
            agents=[
                AgentConfig(
                    name=f"A{i}",
                    role=roles_cycle[i % len(roles_cycle)],
                    model=models_cycle[i],
                    temperature=0.3,
                )
                for i in range(9)
            ],
        )


def test_load_council_config_from_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "council.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "agents": [
                    {
                        "name": "A",
                        "role": "ideator",
                        "model": "ollama/gemma3:2b",
                        "temperature": 0.9,
                    },
                    {
                        "name": "B",
                        "role": "reasoner",
                        "model": "ollama/deepseek-r1:1.5b",
                        "temperature": 0.4,
                    },
                    {
                        "name": "C",
                        "role": "critic",
                        "model": "ollama/phi4-mini:3.8b",
                        "temperature": 0.2,
                    },
                    {
                        "name": "D",
                        "role": "orchestrator",
                        "model": "anthropic/claude-haiku-4-5-20251001",
                        "temperature": 0.3,
                    },
                ],
            }
        )
    )
    cfg = load_council_config(cfg_path)
    assert len(cfg.agents) == 4
    assert cfg.agents[0].family == "gemma"


def test_load_council_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CouncilConfigError, match="config not found"):
        load_council_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "template",
    ["laptop-4agent.yaml", "workstation-4agent.yaml", "server-6agent.yaml"],
)
def test_shipped_templates_are_valid(template: str) -> None:
    """Every shipped template must validate — this is a smoke test for diversity rule too."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "src" / "ai_agent_council" / "templates" / template
    cfg = load_council_config(path)
    assert len(cfg.agents) >= 4
