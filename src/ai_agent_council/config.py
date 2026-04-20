"""Council configuration: pydantic schema, YAML loader, cognitive-diversity validator."""

import hashlib
import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import CouncilConfigError
from .models import Role

_PROVIDER_STRIP = re.compile(r"^[^/]+/")
_TAG_STRIP = re.compile(r":.*$")
_FAMILY_CAPTURE = re.compile(r"^([a-z]+)")


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
    without_provider = _PROVIDER_STRIP.sub("", model, count=1)
    without_tag = _TAG_STRIP.sub("", without_provider)
    m = _FAMILY_CAPTURE.match(without_tag.lower())
    if not m:
        raise CouncilConfigError(f"cannot derive family from model string: {model!r}")
    return m.group(1)


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

    @property
    def family(self) -> str:
        return derive_family(self.model)


class CouncilConfig(BaseModel):
    """Top-level YAML schema. Enforces the design-doc invariants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    name: str = "default-council"
    agents: list[AgentConfig] = Field(min_length=4, max_length=8)

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
        if Role.CRITIC not in roles:
            raise ValueError("roster must include at least one critic")

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
