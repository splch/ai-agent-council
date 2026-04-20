"""Exceptions for the council."""

from __future__ import annotations


class CouncilError(Exception):
    """Base for all council errors."""


class CouncilConfigError(CouncilError):
    """Invalid roster or config."""


class AgentError(CouncilError):
    """Unrecoverable agent failure."""


class PhaseError(CouncilError):
    """A phase could not be completed."""


class LLMError(CouncilError):
    """Base for LLM provider errors."""


class LLMTimeoutError(LLMError):
    """The provider timed out."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited us."""
