"""Stdlib structured logging setup."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, ClassVar

LOGGER_NAME = "ai_agent_council"

_configured = False


class JsonLineFormatter(logging.Formatter):
    """One-JSON-object-per-line formatter. Extras merged via `extra=` get first-class keys."""

    _RESERVED: ClassVar[frozenset[str]] = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure (idempotently) the package logger. Level via `COUNCIL_LOG_LEVEL` env var."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if _configured:
        return logger
    resolved_level = level or os.environ.get("COUNCIL_LOG_LEVEL", "INFO")
    logger.setLevel(resolved_level.upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True
    return logger
