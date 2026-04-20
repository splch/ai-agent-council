"""Evaluation harness: council vs baseline.

The honest question the research keeps raising: does the council actually beat a
single well-prompted strong model on your tasks? A five-member council makes
15+ LLM calls plus tool calls per query; the baseline makes one. Without a
measurement path, every future "improvement" is faith-based.

This harness is deliberately small. It runs a fixed task set through both a
Council and a single-model baseline, capturing per-task outputs, timings,
token counts, costs, and the raw material for a human or LLM-judge scorecard
afterwards. Scoring quality is intentionally out-of-band — you or a judge
model grade the JSON after the fact. The harness's job is to produce apples-
to-apples data, not to score it.

Task file format (YAML):

    name: "my evaluation"
    tasks:
      - id: factual_1
        category: factual
        task: "In what year did DeepMind release AlphaFold 2?"
      - id: code_1
        category: code
        task: "Write a Python one-liner that …"

Run: `council eval --config council.yaml --tasks evals/tasks.yaml --baseline
ollama/phi4:14b --out /tmp/eval.json`
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import llm
from ._timing import elapsed_ms
from .config import CouncilConfig, load_council_config
from .council import Council


class TaskSpec(BaseModel):
    """One evaluation task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(default="general", min_length=1)
    task: str = Field(min_length=1)


class TaskSet(BaseModel):
    """Named collection of tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tasks: list[TaskSpec] = Field(min_length=1)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class RunResult(BaseModel):
    """One agent's result on one task (council or baseline)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    mode: str  # "council" | "baseline"
    answer: str
    elapsed_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str | None = None


class EvalReport(BaseModel):
    """Full side-by-side results for a task set."""

    model_config = ConfigDict(frozen=True)

    task_set: str
    council_name: str
    baseline_model: str | None
    started_at: datetime
    finished_at: datetime
    results: list[RunResult]

    def by_task(self) -> dict[str, list[RunResult]]:
        grouped: dict[str, list[RunResult]] = {}
        for r in self.results:
            grouped.setdefault(r.task_id, []).append(r)
        return grouped

    @property
    def totals(self) -> dict[str, dict[str, float]]:
        """Aggregate cost / tokens / time per mode across the whole set."""
        totals: dict[str, dict[str, float]] = {}
        for r in self.results:
            bucket = totals.setdefault(
                r.mode, {"elapsed_ms": 0.0, "tokens_in": 0.0, "tokens_out": 0.0, "cost_usd": 0.0}
            )
            bucket["elapsed_ms"] += r.elapsed_ms
            bucket["tokens_in"] += r.tokens_in
            bucket["tokens_out"] += r.tokens_out
            bucket["cost_usd"] += r.cost_usd
        return totals


async def _run_council(council: Council, task: TaskSpec) -> RunResult:
    with elapsed_ms() as ms:
        try:
            result = await council.run(task.task)
        except Exception as e:
            return RunResult(
                task_id=task.id,
                mode="council",
                answer="",
                elapsed_ms=ms(),
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                error=f"{type(e).__name__}: {e}",
            )
    tin, tout = result.total_tokens
    return RunResult(
        task_id=task.id,
        mode="council",
        answer=result.final_answer,
        elapsed_ms=ms(),
        tokens_in=tin,
        tokens_out=tout,
        cost_usd=result.total_cost_usd,
    )


_BASELINE_SYSTEM = (
    "You are answering a single-shot evaluation task. Respond directly and "
    "concisely. If you do not know the answer, say so."
)


async def _run_baseline(model: str, task: TaskSpec) -> RunResult:
    with elapsed_ms() as ms:
        try:
            content, meta = await llm.complete(
                model=model,
                system=_BASELINE_SYSTEM,
                user=task.task,
                temperature=0.3,
                max_tokens=2048,
                timeout_s=120.0,
            )
        except Exception as e:
            return RunResult(
                task_id=task.id,
                mode="baseline",
                answer="",
                elapsed_ms=ms(),
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                error=f"{type(e).__name__}: {e}",
            )
    return RunResult(
        task_id=task.id,
        mode="baseline",
        answer=content,
        elapsed_ms=ms(),
        tokens_in=meta.get("tokens_in") or 0,
        tokens_out=meta.get("tokens_out") or 0,
        cost_usd=meta.get("cost_usd") or 0.0,
    )


async def run_eval(
    task_set: TaskSet,
    config: CouncilConfig,
    *,
    baseline_model: str | None = None,
) -> EvalReport:
    """Run `task_set` through the council and (optionally) through a baseline model."""
    started = datetime.now(UTC)
    council = Council(config)
    results: list[RunResult] = []
    for task in task_set.tasks:
        results.append(await _run_council(council, task))
        if baseline_model is not None:
            results.append(await _run_baseline(baseline_model, task))
    return EvalReport(
        task_set=task_set.name,
        council_name=config.name,
        baseline_model=baseline_model,
        started_at=started,
        finished_at=datetime.now(UTC),
        results=results,
    )


def run_eval_from_paths(
    task_path: Path | str,
    config_path: Path | str,
    *,
    baseline_model: str | None = None,
) -> EvalReport:
    """Sync wrapper. Reads both YAMLs, runs end-to-end, returns the report."""
    task_set = TaskSet.from_yaml(task_path)
    config = load_council_config(config_path)
    return asyncio.run(run_eval(task_set, config, baseline_model=baseline_model))
