"""Exceptions for the council."""


class CouncilError(Exception):
    """Base for all council errors."""


class CouncilConfigError(CouncilError):
    """Invalid roster or config."""


class LLMError(CouncilError):
    """Base for LLM provider errors."""


class LLMTimeoutError(LLMError):
    """The provider timed out."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited us."""
