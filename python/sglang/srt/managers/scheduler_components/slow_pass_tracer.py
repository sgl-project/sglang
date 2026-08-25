"""Slow-pass stack dumps for scheduler stall attribution.

When ``SGLANG_DEBUG_SLOW_SCHEDULER_PASS_MS`` > 0, wrapped scheduler passes arm
a ``faulthandler`` watchdog: if the pass runs longer than the threshold, the
watchdog dumps every python thread's stack (file:line) to stderr — i.e. into
the scheduler's log — catching stalls that are invisible to torch profiler
traces (pure-python holes) without external py-spy timing luck. A follow-up
log line names the pass and its duration so dumps are attributable.
"""

from __future__ import annotations

import faulthandler
import logging
import time
from contextlib import contextmanager

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class SlowPassTracer:
    def __init__(self) -> None:
        ms = envs.SGLANG_DEBUG_SLOW_SCHEDULER_PASS_MS.get() or 0.0
        self.timeout_s: float = ms / 1000.0 if ms > 0 else 0.0

    @property
    def enabled(self) -> bool:
        return self.timeout_s > 0.0

    @contextmanager
    def trace(self, label: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        faulthandler.dump_traceback_later(self.timeout_s, repeat=False, exit=False)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            elapsed = time.perf_counter() - start
            if elapsed >= self.timeout_s:
                logger.warning(
                    "slow scheduler pass: %s took %.1fms (>= %.1fms; thread "
                    "stacks were dumped to stderr by the faulthandler watchdog)",
                    label,
                    elapsed * 1e3,
                    self.timeout_s * 1e3,
                )
