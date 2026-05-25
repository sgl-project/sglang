from __future__ import annotations

import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


def normalize_metric_label(value: Any, default: str = "unknown") -> str:
    value = str(value or default).split(":", 1)[0].strip().lower()
    chars = [ch if ch.isalnum() or ch == "_" else "_" for ch in value]
    value = "".join(chars).strip("_")
    return (value or default)[:80]


def observe_reuse(
    metrics_collector,
    *,
    backend: str,
    outcome: str,
    reason: str,
    tokens: int = 0,
    wait_ms: Optional[float] = None,
    insert_ms: Optional[float] = None,
    transfer_bytes: Optional[int] = None,
) -> None:
    observe = getattr(metrics_collector, "observe_shared_hicache", None)
    if observe is None:
        return
    try:
        observe(
            backend=normalize_metric_label(backend),
            outcome=normalize_metric_label(outcome),
            reason=normalize_metric_label(reason),
            tokens=max(0, int(tokens)),
            wait_ms=wait_ms,
            insert_ms=insert_ms,
            transfer_bytes=transfer_bytes,
        )
    except Exception:
        logger.debug("Failed to record SharedHiCache metrics", exc_info=True)


def observe_quarantine(
    metrics_collector,
    *,
    backend: str,
    reason: str,
    tokens: int,
    current_tokens: int,
) -> None:
    observe = getattr(metrics_collector, "observe_shared_hicache_quarantine", None)
    if observe is None:
        return
    try:
        observe(
            backend=normalize_metric_label(backend),
            reason=normalize_metric_label(reason),
            tokens=max(0, int(tokens)),
            current_tokens=max(0, int(current_tokens)),
        )
    except Exception:
        logger.debug("Failed to record SharedHiCache quarantine metrics", exc_info=True)
