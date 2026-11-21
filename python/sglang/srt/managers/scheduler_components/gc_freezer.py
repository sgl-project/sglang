import collections
import gc
import logging
import math
import time

from sglang.srt.utils.gc_utils import freeze_gc, refreeze_gc

logger = logging.getLogger(__name__)

GC_CHECK_INTERVAL_SECS = 600.0
HEAP_GROWTH_WARN_RATIO = 1.5


class GCFreezer:
    def __init__(
        self,
        *,
        freeze_on_idle_secs: float,
        freeze_after_batches: int = 0,
        freeze_every_batches: int = 0,
    ):
        self._freeze_on_idle_secs = freeze_on_idle_secs
        self._freeze_after_batches = freeze_after_batches
        self._freeze_every_batches = freeze_every_batches
        self._next_batch_freeze = freeze_after_batches or math.inf
        self._batch_froze = False
        self._seen_forward_ct = -1
        self._next_freeze = math.inf
        now = time.monotonic()
        self._next_heap_check = now + GC_CHECK_INTERVAL_SECS
        self._next_garbage_check = now + GC_CHECK_INTERVAL_SECS
        self._heap_warn_above = None

    def on_cache_reset(self) -> None:
        if self._freeze_on_idle_secs > 0 or self._freeze_after_batches > 0:
            gc.unfreeze()
        if self._freeze_on_idle_secs > 0:
            self._next_freeze = time.monotonic()
        if self._batch_froze:
            self._next_batch_freeze = 0

    def on_batch(self, forward_ct: int) -> None:
        if forward_ct < self._next_batch_freeze:
            return
        self._next_batch_freeze = (
            forward_ct + self._freeze_every_batches
            if self._freeze_every_batches > 0
            else math.inf
        )
        start = time.monotonic()
        # Young generations only: a full collection on a busy scheduler is the
        # pause this freeze exists to avoid.
        gc.collect(1)
        gc.freeze()
        elapsed_ms = (time.monotonic() - start) * 1e3
        if self._batch_froze:
            logger.debug(
                f"Froze the scheduler heap at batch {forward_ct} in {elapsed_ms:.1f} ms"
            )
            return
        self._batch_froze = True
        # gc.get_freeze_count() walks the frozen list: only at the first freeze.
        logger.info(
            f"Froze the scheduler heap at batch {forward_ct} in {elapsed_ms:.1f} ms: "
            f"{gc.get_freeze_count()} objects; next every "
            f"{self._freeze_every_batches or 'never'} batches"
        )

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
