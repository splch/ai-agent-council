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
_MINIMAL_CFG = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "ai_agent_council"
    / "templates"
    / "minimal.yaml"
)


@pytest.mark.skipif(not RUN_IT, reason="set COUNCIL_OLLAMA_IT=1 to run")
async def test_minimal_config_runs_against_live_ollama() -> None:
    council = Council(load_council_config(_MINIMAL_CFG))
    result = await council.run("Reply with the word 'ok' and nothing else.")
    assert result.final_answer
    assert len(result.phases) >= 4
