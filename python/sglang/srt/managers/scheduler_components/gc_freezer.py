import collections
import gc
import logging
import time
from typing import Callable, List

from sglang.srt.managers.scheduler_components.state_check import (
    CheckPhase,
    FailureReporter,
    Severity,
    StateCheck,
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
        check_frozen_garbage: bool,
        get_forward_ct: Callable[[], int],
    ):
        self._freeze_on_idle_secs = freeze_on_idle_secs
        self._freezes = freeze_on_idle_secs > 0
        self._check_frozen_garbage = check_frozen_garbage
        self._get_forward_ct = get_forward_ct
        self._reporter = FailureReporter()
        self._seen_forward_ct = -1
        self._idle_since = 0.0
        self._frozen = False
        now = time.monotonic()
        self._next_heap_check = now + GC_CHECK_INTERVAL_SECS
        self._next_garbage_check = now + GC_CHECK_INTERVAL_SECS
        self._heap_warn_above = None

    def checks(self) -> List[StateCheck]:
        return [
            StateCheck(phase=CheckPhase.IDLE, when=lambda: True, fn=self._on_idle),
            StateCheck(
                phase=CheckPhase.FLUSH, when=lambda: True, fn=self.on_cache_reset
            ),
        ]

    def on_cache_reset(self) -> None:
        if self._freezes:
            gc.unfreeze()
            gc.collect()
            freeze_gc("Scheduler")

    def _on_idle(self) -> None:
        now = time.monotonic()
        if now >= self._next_heap_check:
            self._next_heap_check = now + GC_CHECK_INTERVAL_SECS
            self._check_heap_growth()
        forward_ct = self._get_forward_ct()
        if forward_ct != self._seen_forward_ct:
            self._seen_forward_ct = forward_ct
            self._idle_since = now
            self._frozen = False
        elif (
            self._freezes
            and not self._frozen
            and now - self._idle_since >= self._freeze_on_idle_secs
        ):
            self._freeze(now)
            self._frozen = True

    def _freeze(self, now: float) -> None:
        if self._check_frozen_garbage or now >= self._next_garbage_check:
            self._next_garbage_check = now + GC_CHECK_INTERVAL_SECS
            self._report_frozen_garbage(refreeze_gc("Scheduler"))
        else:
            gc.collect()
            freeze_gc("Scheduler")

    def _report_frozen_garbage(self, garbage: collections.Counter) -> None:
        if not garbage:
            return
        message = (
            f"{sum(garbage.values())} frozen objects had become cyclic garbage, "
            f"which a frozen heap never frees; collected them now. "
            f"Most common: {garbage.most_common(5)}"
        )
        if self._check_frozen_garbage:
            self._reporter.report(Severity.LEAK, message)
        else:
            logger.warning(message)

    def _check_heap_growth(self) -> None:
        unfrozen = gc.get_objects(2)
        long_lived = len(unfrozen) + gc.get_freeze_count()
        if self._heap_warn_above is not None and long_lived >= self._heap_warn_above:
            types = collections.Counter(type(obj).__qualname__ for obj in unfrozen)
            logger.warning(
                f"Long-lived Python objects in the scheduler grew to {long_lived} "
                f"({long_lived - len(unfrozen)} frozen). Each full GC pass scans the "
                f"unfrozen ones, and frozen ones that become garbage are never freed. "
                f"Most common unfrozen: {types.most_common(5)}"
            )
        if self._heap_warn_above is None or long_lived >= self._heap_warn_above:
            self._heap_warn_above = long_lived * HEAP_GROWTH_WARN_RATIO
