"""Integration smoke test against a live Ollama instance.

Gated by the env var `COUNCIL_OLLAMA_IT=1` — the CI matrix does not run it by default.
"""

import os
from pathlib import Path

import pytest

from ai_agent_council.config import load_council_config
from ai_agent_council.council import Council

pytestmark = pytest.mark.integration

RUN_IT = os.environ.get("COUNCIL_OLLAMA_IT") == "1"


@pytest.mark.skipif(not RUN_IT, reason="set COUNCIL_OLLAMA_IT=1 to run")
async def test_laptop_config_runs_against_live_ollama() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "src" / "ai_agent_council" / "templates" / "laptop-4agent.yaml"
    cfg = load_council_config(cfg_path)
    council = Council(cfg)
    result = await council.run("Reply with the word 'ok' and nothing else.")
    assert result.final_answer
    assert len(result.phases) >= 4
