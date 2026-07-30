"""Small, synchronization-free helpers for opt-in diagnostic telemetry."""

from __future__ import annotations

import logging
import threading
from typing import Any, Hashable

import torch


def _format_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        shape = ",".join(str(dim) for dim in value.shape)
        return f"tensor(shape=[{shape}],dtype={value.dtype},device={value.device})"
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_format_value(item) for item in value) + "]"
    return str(value).replace(" ", "_")


def format_telemetry_fields(**fields: Any) -> str:
    """Format host metadata without inspecting tensor contents."""
    return " ".join(f"{name}={_format_value(value)}" for name, value in fields.items())


def is_rank_zero() -> bool:
    """Return rank-zero status without initializing or synchronizing distributed."""
    try:
        return (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
    except (AttributeError, RuntimeError):
        return True


class BoundedTelemetryLogger:
    """Log each key once, up to a process-local hard limit."""

    def __init__(
        self,
        logger: logging.Logger,
        signature: str,
        *,
        enabled: bool,
        max_events: int = 64,
        rank_zero_only: bool = True,
    ) -> None:
        self._logger = logger
        self._signature = signature
        self._enabled = enabled
        self._max_events = max_events
        self._rank_zero_only = rank_zero_only
        self._keys: set[Hashable] = set()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, key: Hashable, event: str, **fields: Any) -> bool:
        if not self._enabled or (self._rank_zero_only and not is_rank_zero()):
            return False
        with self._lock:
            if key in self._keys or len(self._keys) >= self._max_events:
                return False
            self._keys.add(key)
        suffix = format_telemetry_fields(event=event, **fields)
        self._logger.info("%s %s", self._signature, suffix)
        return True
