"""Eval harness tests — using the FakeLLM to run the full plumbing without LLM cost."""

from pathlib import Path

import pytest
import yaml

from ai_agent_council.config import AgentConfig, CouncilConfig
from ai_agent_council.evals import EvalReport, TaskSet, TaskSpec, run_eval
from ai_agent_council.models import Role

from .fakes import FakeLLM


def _minimal_config() -> CouncilConfig:
    return CouncilConfig(
        name="eval-test",
        agents=[
            AgentConfig(name="Muse", role=Role.IDEATOR, model="ollama/gemma3:2b", temperature=0.9),
            AgentConfig(
                name="Judge",
                role=Role.REASONER,
                model="ollama/deepseek-r1:1.5b",
                temperature=0.4,
            ),
            AgentConfig(
                name="Hawk", role=Role.CRITIC, model="ollama/phi4-mini:3.8b", temperature=0.2
            ),
            AgentConfig(
                name="Scribe",
                role=Role.ORCHESTRATOR,
                model="ollama/llama3.2:3b",
                temperature=0.3,
            ),
        ],
    )


def test_task_set_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "smoke",
                "tasks": [
                    {"id": "t1", "task": "what time is it?"},
                    {"id": "t2", "category": "code", "task": "fizzbuzz"},
                ],
            }
        )
    )
    ts = TaskSet.from_yaml(path)
    assert ts.name == "smoke"
    assert len(ts.tasks) == 2
    assert ts.tasks[0].category == "general"  # default
    assert ts.tasks[1].category == "code"


async def test_run_eval_council_only(fake_llm: FakeLLM) -> None:
    fake_llm.default = "council answer"
    ts = TaskSet(name="tiny", tasks=[TaskSpec(id="t1", task="hello?")])
    report = await run_eval(ts, _minimal_config())
    assert len(report.results) == 1
    assert report.results[0].mode == "council"
    assert report.results[0].answer == "council answer"
    assert report.baseline_model is None


async def test_run_eval_with_baseline(fake_llm: FakeLLM) -> None:
    # The FakeLLM's default is used for both council and baseline calls since it
    # intercepts llm.complete regardless of caller.
    fake_llm.default = "baseline-or-council answer"
    ts = TaskSet(
        name="tiny",
        tasks=[
            TaskSpec(id="t1", task="hello?"),
            TaskSpec(id="t2", task="world?"),
        ],
    )
    report = await run_eval(ts, _minimal_config(), baseline_model="ollama/phi4:14b")
    assert len(report.results) == 4  # 2 tasks x 2 modes
    modes = {r.mode for r in report.results}
    assert modes == {"council", "baseline"}
    # Grouped by-task should pair them up.
    for runs in report.by_task().values():
        assert len(runs) == 2
        assert {r.mode for r in runs} == {"council", "baseline"}


async def test_totals_aggregate_correctly(fake_llm: FakeLLM) -> None:
    fake_llm.default = "answer"
    fake_llm.cost_per_call = 0.005
    ts = TaskSet(
        name="tiny",
        tasks=[TaskSpec(id="t1", task="a"), TaskSpec(id="t2", task="b")],
    )
    report = await run_eval(ts, _minimal_config(), baseline_model="ollama/phi4:14b")
    totals = report.totals
    # Council makes many calls per task (divergent + critique + synthesis + orchestrate);
    # the baseline makes exactly one per task. Both non-zero at $0.005/call.
    assert totals["council"]["cost_usd"] > 0
    assert totals["baseline"]["cost_usd"] == pytest.approx(2 * 0.005)  # 1 call per task


async def test_report_is_json_serializable(fake_llm: FakeLLM) -> None:
    fake_llm.default = "x"
    ts = TaskSet(name="tiny", tasks=[TaskSpec(id="t1", task="?")])
    report = await run_eval(ts, _minimal_config())
    blob = report.model_dump_json()
    assert '"results"' in blob
    assert '"task_set"' in blob
    # Roundtrip
    roundtripped = EvalReport.model_validate_json(blob)
    assert roundtripped.task_set == report.task_set


def test_shipped_default_task_set_is_valid() -> None:
    """The evals/tasks.yaml shipped with the repo must parse."""
    path = Path(__file__).resolve().parent.parent / "evals" / "tasks.yaml"
    assert path.exists()
    ts = TaskSet.from_yaml(path)
    assert len(ts.tasks) >= 5
    # Every task has a category for grouping in analysis.
    assert all(t.category for t in ts.tasks)
