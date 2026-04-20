"""Council configuration: pydantic schema, YAML loader, cognitive-diversity validator."""

import hashlib
import itertools
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import CouncilConfigError
from .models import Role


def derive_family(model: str) -> str:
    """Derive a family tag from a LiteLLM-format model string.

    Examples:
        ollama/llama3.1:8b          -> llama
        anthropic/claude-sonnet-4-6 -> claude
        openai/gpt-5.1              -> gpt
        ollama/qwen2.5-coder:7b     -> qwen
        ollama/deepseek-r1:7b       -> deepseek
        ollama/phi4-mini:3.8b       -> phi
        ollama/gemma3:4b            -> gemma
        ollama/mistral-small        -> mistral

    Families are used to enforce the cognitive-diversity rule at config-load time.
    """
    if not model:
        raise CouncilConfigError("empty model string")
    head = model.split("/", 1)[-1].split(":", 1)[0].lower()
    family = "".join(itertools.takewhile(str.isalpha, head))
    if not family:
        raise CouncilConfigError(f"cannot derive family from model string: {model!r}")
    return family


class AgentConfig(BaseModel):
    """Per-agent config entry in a council YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=40)
    role: Role
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    timeout_s: float = Field(default=120.0, gt=0.0)
    system_prompt: str | None = None
    json_mode: bool = False
    # Names of tools from ai_agent_council.tools the agent may call during its turn. Only
    # the Finisher uses tools in the shipped templates, but any role can opt in.
    tools: list[str] = Field(default_factory=list)
    # Cap on the model's tool-call loop. After this many LLM rounds without a plain-content
    # answer, the loop exits with whatever content came last.
    max_tool_iterations: int = Field(default=5, ge=1, le=20)
    # Big Five personality dials. Each is optional (None = don't include in the prompt).
    # Research: prompt-based numeric trait scaling yields r > 0.85 correlation between
    # assigned levels and measured behavioral shifts. Range 0.0-1.0; 0.5 is average.
    openness: float | None = Field(default=None, ge=0.0, le=1.0)
    conscientiousness: float | None = Field(default=None, ge=0.0, le=1.0)
    extraversion: float | None = Field(default=None, ge=0.0, le=1.0)
    agreeableness: float | None = Field(default=None, ge=0.0, le=1.0)
    neuroticism: float | None = Field(default=None, ge=0.0, le=1.0)

    def persona_dict(self) -> dict[str, float]:
        """Return only the trait dials that are set — suitable for prompt rendering."""
        traits = {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }
        return {k: v for k, v in traits.items() if v is not None}

    @property
    def family(self) -> str:
        return derive_family(self.model)


class CouncilConfig(BaseModel):
    """Top-level YAML schema. Enforces the design-doc invariants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    name: str = "default-council"
    agents: list[AgentConfig] = Field(min_length=4, max_length=8)
    # When true, the Council runs a retrospective phase after delivery, appending lessons
    # to `<retrospective_dir>/<name>.jsonl`, and loads up to `retrospective_recall` recent
    # entries into each agent's system prompt on construction.
    retrospective: bool = False
    retrospective_dir: Path | None = None
    retrospective_recall: int = Field(default=5, ge=0, le=50)
    # When true, insert a restate phase BEFORE divergent: every agent independently
    # restates the task in their own words and proposes one alternative framing. If
    # the restatements materially diverge, the question itself was probably ambiguous —
    # surfaced via the transcript for human review. Research catches ~20% of tasks
    # where the user asked the wrong question.
    restate: bool = False
    # Minimum fraction of critics who must find substantive issues. If the first
    # critique round comes back with too few real objections (below this threshold),
    # a steelman round is triggered — critics are required to articulate the
    # strongest possible objection they can, even if they initially found the
    # proposal acceptable. Set 0 (default) to disable. The research maps this
    # to the "if >70% agree too early, force steelman" dissent quota.
    min_dissent: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_roster(self) -> Self:
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate agent names: {dupes}")

        roles = [a.role for a in self.agents]
        orchestrators = [a for a in self.agents if a.role is Role.ORCHESTRATOR]
        if len(orchestrators) != 1:
            raise ValueError(f"exactly one orchestrator required; found {len(orchestrators)}")
        if Role.IDEATOR not in roles:
            raise ValueError("roster must include at least one ideator")
        # The critique phase enlists Critic + Reasoner as reviewers. Allow rosters that
        # only have a Reasoner — matches the design brief's "Minimal (laptop)" where the
        # Reasoner doubles as Critic for compute reasons.
        if Role.CRITIC not in roles and Role.REASONER not in roles:
            raise ValueError("roster must include at least one critic or reasoner")

        families: dict[str, str] = {}
        for a in self.agents:
            fam = a.family
            if fam in families:
                raise ValueError(
                    "cognitive-diversity rule violated: "
                    f"{families[fam]!r} and {a.name!r} both belong to family {fam!r}"
                )
            families[fam] = a.name
        return self


def load_council_config(path: Path | str) -> CouncilConfig:
    """Parse and validate a council YAML."""
    p = Path(path)
    if not p.exists():
        raise CouncilConfigError(f"config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CouncilConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(raw, dict):
        raise CouncilConfigError(f"top-level of {p} must be a mapping")
    try:
        return CouncilConfig.model_validate(raw)
    except Exception as e:
        raise CouncilConfigError(f"invalid council config {p}: {e}") from e


def hash_config(cfg: CouncilConfig) -> str:
    """Digest of a config — used to correlate transcripts with rosters."""
    canonical = cfg.model_dump_json(exclude_none=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]
