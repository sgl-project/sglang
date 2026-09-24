"""Circuit breaker that isolates HiCache from a failing L3 storage backend.

The storage threads in ``HiCacheController`` must never die and must always
emit their acks: acks unpin host pages and feed cross-rank collectives, so a
single unhandled storage error would otherwise strand every rank. Catching the
error keeps the threads alive; the breaker additionally stops issuing *new* L3
I/O for a cool-down after repeated consecutive failures, so a hung or broken
store does not keep burning I/O time on every request.

Reads and writes use separate breakers: a store that is readable but no longer
writable (ENOSPC, lost write permission) keeps serving hits, and successes in
one direction never mask failures in the other.

In-flight operations are never abandoned by the breaker; it only gates the
start of new ones.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class StorageCircuitBreaker:
    """Consecutive-failure circuit breaker with a timed half-open probe.

    * closed: operations are allowed; ``failure_threshold`` consecutive
      failures open the breaker.
    * open: ``allow()`` returns False until ``cooldown_s`` has elapsed.
    * half-open: after the cool-down, operations are allowed again; the first
      success closes the breaker, the first failure re-opens it immediately.

    ``failure_threshold <= 0`` disables the breaker (it never opens).
    Thread-safe; ``allow`` and ``record_*`` may be called from several storage
    threads concurrently.
    """

    def __init__(
        self,
        direction: str,
        failure_threshold: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
        log_interval_s: float = 30.0,
    ):
        self.direction = direction
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._log_interval_s = log_interval_s
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = None  # None: closed; float: open (or half-open)
        self._last_log = float("-inf")
        # Counters, for logs/metrics and tests.
        self.num_failures = 0
        self.num_opens = 0
        self.num_rejected = 0

    @property
    def enabled(self) -> bool:
        return self.failure_threshold > 0

    def allow(self) -> bool:
        """Whether a new L3 operation in this direction may be issued."""
        with self._lock:
            if self._open_until is None or self._clock() >= self._open_until:
                return True
            self.num_rejected += 1
            return False

    def is_open(self) -> bool:
        with self._lock:
            return self._open_until is not None and self._clock() < self._open_until

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            if self._open_until is not None:
                self._open_until = None
                logger.warning(
                    "HiCache L3 %s circuit closed: storage %s succeeded again.",
                    self.direction,
                    self.direction,
                )

    def record_failure(self, where: str, error: object) -> None:
        """Record one failed operation (an exception or a failed/short result)."""
        with self._lock:
            now = self._clock()
            self.num_failures += 1
            self._consecutive_failures += 1
            if now - self._last_log >= self._log_interval_s:
                self._last_log = now
                logger.error(
                    "HiCache L3 %s failed (%d consecutive %s failures): %r",
                    where,
                    self._consecutive_failures,
                    self.direction,
                    error,
                )
            if not self.enabled:
                return
            half_open = self._open_until is not None and now >= self._open_until
            if half_open or self._consecutive_failures >= self.failure_threshold:
                self._open_until = now + self.cooldown_s
                self.num_opens += 1
                logger.error(
                    "HiCache L3 %s circuit OPEN for %.0fs after %d consecutive "
                    "failures; serving continues without L3 %s.",
                    self.direction,
                    self.cooldown_s,
                    self._consecutive_failures,
                    self.direction,
                )
                self._consecutive_failures = 0
