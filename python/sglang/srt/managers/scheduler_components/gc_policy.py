"""Scheduler GC policy: periodic manual collects instead of automatic GC.

At large live sets (batch ~1500, 65k-token outputs) CPython's automatic gen2
collections become 100-350ms stop-the-world pauses every few decode steps.
With SGLANG_OPT_SCHEDULER_GC_COLLECT_INTERVAL_S > 0: one collect + gc.freeze
of startup objects on activation, automatic GC disabled, then one full
collect per interval at a loop boundary. Refcounting still reclaims acyclic
per-request state immediately; the periodic pass handles cycles. Caveat: a
startup-era object that is cyclic and dies later leaks (rare, small).
"""

from __future__ import annotations

import gc
import logging
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class SchedulerGcPolicy:
    def __init__(self) -> None:
        self.interval_s: float = (
            envs.SGLANG_OPT_SCHEDULER_GC_COLLECT_INTERVAL_S.get() or 0.0
        )
        self._activated = False
        self._next_collect_ts = 0.0

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0.0

    def _activate(self, now: float) -> None:
        gc.collect()
        gc.freeze()
        gc.disable()
        self._activated = True
        self._next_collect_ts = now + self.interval_s
        logger.info(
            "SchedulerGcPolicy active: automatic GC disabled, %d startup objects "
            "frozen, manual collect every %.1fs.",
            gc.get_freeze_count(),
            self.interval_s,
        )

    def maybe_run(self) -> None:
        """Cheap per-iteration hook: activates on first call, then runs one
        full collection whenever the interval elapsed."""
        if not self.enabled:
            return
        now = time.monotonic()
        if not self._activated:
            self._activate(now)
            return
        if now < self._next_collect_ts:
            return
        start = time.perf_counter()
        unreachable = gc.collect()
        elapsed = time.perf_counter() - start
        self._next_collect_ts = now + self.interval_s
        logger.debug(
            "SchedulerGcPolicy collect: %.1fms, %d unreachable objects.",
            elapsed * 1e3,
            unreachable,
        )
