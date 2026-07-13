# SPDX-License-Identifier: Apache-2.0
"""Rate-limited fallback diagnostics for continuous-batching hot paths.

Packed-step and batched-solver fallbacks are correct but slower; operators
need visibility into how often and *why* they happen without log spam. Every
occurrence is counted and logged at DEBUG; the first occurrence of each
(kind, key, reason) is logged at INFO and then re-logged at most once per
interval with the accumulated count.
"""

from __future__ import annotations

import threading
import time
from collections import Counter

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_INTERVAL_S = 600.0


class FallbackDiagnostics:
    """Counts fallback reasons and rate-limits operator-facing logs."""

    def __init__(self, interval_s: float = _DEFAULT_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._counts: Counter[tuple[str, str, str]] = Counter()
        self._last_logged_s: dict[tuple[str, str, str], float] = {}

    def record(self, kind: str, key: str, reason: str) -> None:
        """Count one fallback and emit a rate-limited log."""
        entry = (kind, key, reason)
        now = time.monotonic()
        with self._lock:
            self._counts[entry] += 1
            count = self._counts[entry]
            last = self._last_logged_s.get(entry)
            should_log_info = last is None or (now - last) >= self._interval_s
            if should_log_info:
                self._last_logged_s[entry] = now
        if should_log_info:
            logger.info(
                "%s fallback for %s: %s (seen %d time(s); "
                "running the slower per-request path)",
                kind,
                key,
                reason,
                count,
            )
        else:
            logger.debug("%s fallback for %s: %s", kind, key, reason)

    def snapshot(self) -> dict[tuple[str, str, str], int]:
        """Return a copy of the fallback counters (for tests/metrics)."""
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._last_logged_s.clear()


# Process-wide diagnostics shared by the denoising stage and batched solver.
fallback_diagnostics = FallbackDiagnostics()
