import collections
import gc
import logging
import math
import time
from typing import Callable, Optional

from sglang.srt.utils.gc_utils import freeze_gc, refreeze_gc

logger = logging.getLogger(__name__)

GC_CHECK_INTERVAL_SECS = 600.0
HEAP_GROWTH_WARN_RATIO = 1.5
# A third GC threshold that no count of generation-1 collections reaches, so
# the collector never starts a full collection on its own.
NO_AUTOMATIC_FULL_GC = 10**9


class GCFreezer:
    def __init__(self, *, freeze_on_idle_secs: float, dp_lockstep_steps: int = 0):
        self._freeze_on_idle_secs = freeze_on_idle_secs
        self._dp_lockstep_steps = dp_lockstep_steps
        self._dp_gathers = 0
        # Run before every attention-DP batch gather; None unless
        # --gc-dp-lockstep-steps is set.
        self.before_dp_gather: Optional[Callable[[], None]] = (
            self._collect_in_lockstep if dp_lockstep_steps > 0 else None
        )
        self._seen_forward_ct = -1
        self._next_freeze = math.inf
        now = time.monotonic()
        self._next_heap_check = now + GC_CHECK_INTERVAL_SECS
        self._next_garbage_check = now + GC_CHECK_INTERVAL_SECS
        self._heap_warn_above = None

    def on_cache_reset(self) -> None:
        if self._freeze_on_idle_secs > 0:
            gc.unfreeze()
            self._next_freeze = time.monotonic()

    def on_idle(self, forward_ct: int) -> None:
        now = time.monotonic()
        if now >= self._next_heap_check:
            self._next_heap_check = now + GC_CHECK_INTERVAL_SECS
            self._check_heap_growth()
        if forward_ct != self._seen_forward_ct:
            self._seen_forward_ct = forward_ct
            if self._freeze_on_idle_secs > 0:
                self._next_freeze = now + self._freeze_on_idle_secs
        elif now >= self._next_freeze:
            self._next_freeze = math.inf
            self._freeze(now)

    def _collect_in_lockstep(self) -> None:
        # Each gather is a collective, so every attention-DP rank has run the
        # same number of them: the count names the same step on every rank,
        # and the ranks pause for a full collection together instead of each
        # one stalling all the others at the next gather.
        if self._dp_gathers == 0:
            self._start_lockstep()
        self._dp_gathers += 1
        if self._dp_gathers % self._dp_lockstep_steps == 0:
            gc.collect()

    def _start_lockstep(self) -> None:
        # Freeze what startup built (weights, CUDA graphs, caches), so each
        # lockstep collection scans only the objects created since.
        freeze_gc("Scheduler")
        threshold0, threshold1, _ = gc.get_threshold()
        gc.set_threshold(threshold0, threshold1, NO_AUTOMATIC_FULL_GC)
        if gc.get_threshold()[2] != NO_AUTOMATIC_FULL_GC:
            logger.warning(
                "This Python's garbage collector has no third threshold, so "
                "--gc-dp-lockstep-steps cannot turn off its automatic full "
                "collections; it only adds the lockstep ones."
            )
        logger.info(
            f"Froze the heap and turned off automatic full GC; every attention-DP "
            f"rank runs a full collection every {self._dp_lockstep_steps} batch gathers."
        )

    def _freeze(self, now: float) -> None:
        if now >= self._next_garbage_check:
            self._next_garbage_check = now + GC_CHECK_INTERVAL_SECS
            self._warn_frozen_garbage(refreeze_gc("Scheduler"))
        else:
            freeze_gc("Scheduler")

    def _warn_frozen_garbage(self, garbage: collections.Counter) -> None:
        if not garbage:
            return
        logger.warning(
            f"{sum(garbage.values())} frozen objects are cyclic garbage, "
            f"which a frozen heap never frees; collected them now. "
            f"Most common: {garbage.most_common(5)}"
        )

    def _check_heap_growth(self) -> None:
        unfrozen = gc.get_objects(2)
        frozen = gc.get_freeze_count()
        long_lived = len(unfrozen) + frozen
        if self._heap_warn_above is None:
            self._heap_warn_above = long_lived * HEAP_GROWTH_WARN_RATIO
        elif long_lived >= self._heap_warn_above:
            types = collections.Counter(type(obj).__qualname__ for obj in unfrozen)
            logger.warning(
                f"Long-lived Python objects in the scheduler grew to {long_lived} "
                f"({frozen} frozen). Each full GC pass scans the unfrozen ones, and "
                f"frozen ones that become garbage are not freed until unfrozen. "
                f"Most common unfrozen: {types.most_common(5)}"
            )
            self._heap_warn_above = long_lived * HEAP_GROWTH_WARN_RATIO
