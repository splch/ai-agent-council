"""CLI smoke tests via typer's CliRunner."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_agent_council.cli import AVAILABLE_TEMPLATES, app

runner = CliRunner()


def test_validate_ok(tmp_path: Path) -> None:
    import shutil

    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src" / "ai_agent_council" / "templates" / "workstation.yaml"
    dest = tmp_path / "c.yaml"
    shutil.copyfile(src, dest)
    result = runner.invoke(app, ["validate", str(dest)])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output.lower()


def test_validate_rejects_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "agents:\n"
        "  - {name: A, role: ideator, model: ollama/llama3.1:8b, temperature: 0.9}\n"
        "  - {name: B, role: critic,  model: ollama/llama3.3:70b, temperature: 0.2}\n"
        "  - {name: C, role: reasoner, model: ollama/deepseek-r1:7b, temperature: 0.4}\n"
        "  - {name: D, role: orchestrator, model: ollama/gpt-oss:20b, "
        "temperature: 0.3}\n"
    )
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code != 0


@pytest.mark.parametrize("template", AVAILABLE_TEMPLATES)
def test_init_creates_config_from_template(tmp_path: Path, template: str) -> None:
    dest = tmp_path / f"{template}.yaml"
    result = runner.invoke(app, ["init", str(dest), "--template", template])
    assert result.exit_code == 0, result.output
    assert dest.exists()
    # Validating the scaffolded copy should succeed.
    result2 = runner.invoke(app, ["validate", str(dest)])
    assert result2.exit_code == 0, result2.output


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    dest = tmp_path / "c.yaml"
    dest.write_text("existing content")
    result = runner.invoke(app, ["init", str(dest), "--template", "workstation"])
    assert result.exit_code != 0
    # Rich may wrap the output across a newline; match on a single stable token.
    assert "already" in result.output


def test_init_unknown_template(tmp_path: Path) -> None:
    dest = tmp_path / "c.yaml"
    result = runner.invoke(app, ["init", str(dest), "--template", "nonexistent"])
    assert result.exit_code != 0
