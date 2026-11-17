import collections
import gc
import logging
import math
import time

from sglang.srt.managers.scheduler_components.state_check import (
    FailureReporter,
    Severity,
)
from sglang.srt.utils.gc_utils import freeze_gc, refreeze_gc

logger = logging.getLogger(__name__)

GC_CHECK_INTERVAL_SECS = 600.0
HEAP_GROWTH_WARN_RATIO = 1.5


class GCFreezer:
    def __init__(
        self,
        *,
        freeze_on_idle_secs: float,
        enable_frozen_garbage_check: bool,
    ):
        self._freeze_on_idle_secs = freeze_on_idle_secs
        self._enable_frozen_garbage_check = enable_frozen_garbage_check
        self._reporter = FailureReporter()
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

    def _freeze(self, now: float) -> None:
        if self._enable_frozen_garbage_check or now >= self._next_garbage_check:
            self._next_garbage_check = now + GC_CHECK_INTERVAL_SECS
            self._report_frozen_garbage(refreeze_gc("Scheduler"))
        else:
            freeze_gc("Scheduler")

    def _report_frozen_garbage(self, garbage: collections.Counter) -> None:
        if not garbage:
            return
        message = (
            f"{sum(garbage.values())} frozen objects are cyclic garbage, "
            f"which a frozen heap never frees; collected them now. "
            f"Most common: {garbage.most_common(5)}"
        )
        if self._enable_frozen_garbage_check:
            self._reporter.report(Severity.LEAK, message)
        else:
            logger.warning(message)

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
