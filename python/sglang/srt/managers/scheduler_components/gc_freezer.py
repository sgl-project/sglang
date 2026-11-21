import collections
import gc
import logging
import math
import time
from typing import Optional, Sequence

from sglang.srt.utils.gc_utils import freeze_gc, refreeze_gc

logger = logging.getLogger(__name__)

GC_CHECK_INTERVAL_SECS = 600.0
HEAP_GROWTH_WARN_RATIO = 1.5
# A third GC threshold that no count of generation-1 collections reaches, so
# the collector never starts a full collection on its own.
NO_AUTOMATIC_FULL_GC = 10**9

# Lockstep full GC across attention-DP ranks. Every rank pauses for each pass
# at once, so the interval between passes (in batch gathers, one or two per
# scheduler step) is sized to keep a pass near this pause: well inside a decode
# step, and about what passes took (2-6.6 ms) on a DEP32 decode deployment
# that collected every 30000 gathers.
LOCKSTEP_PAUSE_BUDGET_US = 5_000
# The first pass comes early, so the interval adapts from a measured pause.
LOCKSTEP_INITIAL_GATHERS = 1_000
LOCKSTEP_MIN_GATHERS = 16
# Bounds how long cyclic garbage waits for a pass (~17 min of 10 ms steps).
LOCKSTEP_MAX_GATHERS = 100_000
# An interval grows at most this much per pass, so a pause overshoots the
# budget by at most this factor; it shrinks to its target at once.
LOCKSTEP_MAX_GROWTH = 2
# The pause per gather is a least-squares fit over this many recent passes,
# refit only when their intervals differ by at least this ratio, so noise
# between nearby intervals does not set the slope.
LOCKSTEP_FIT_PASSES = 8
LOCKSTEP_MIN_FIT_RATIO = 1.25


def third_threshold_works() -> bool:
    """Whether the collector keeps a third threshold, which lockstep raises so
    it never starts a full collection on its own."""
    saved = gc.get_threshold()
    try:
        gc.set_threshold(saved[0], saved[1], NO_AUTOMATIC_FULL_GC)
        return gc.get_threshold()[2] == NO_AUTOMATIC_FULL_GC
    finally:
        gc.set_threshold(*saved)


class GCFreezer:
    def __init__(
        self,
        *,
        freeze_on_idle_secs: float,
        dp_lockstep: bool = False,
        gc_threshold: Optional[Sequence[int]] = None,
    ):
        self._freeze_on_idle_secs = freeze_on_idle_secs
        self._dp_lockstep = dp_lockstep and _lockstep_allowed(gc_threshold)
        self._dp_gathers = 0
        self._next_collect = LOCKSTEP_INITIAL_GATHERS
        self._interval = LOCKSTEP_INITIAL_GATHERS
        self._collected_at: Optional[int] = None
        self._pause_us = 0
        self._passes: collections.deque = collections.deque(maxlen=LOCKSTEP_FIT_PASSES)
        self._per_gather_us: Optional[float] = None
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

    def dp_gather_callbacks(self) -> dict:
        """SchedulerDPAttnAdapter callbacks for lockstep GC; none when it is off."""
        if not self._dp_lockstep:
            return {}
        return dict(
            before_gather=self._before_dp_gather,
            get_gc_pause_us=lambda: self._pause_us,
            on_gc_pause=self._on_dp_gc_pause,
        )

    def _before_dp_gather(self) -> None:
        # Each gather is a collective, so every attention-DP rank has run the
        # same number of them: the count names the same step on every rank,
        # and the ranks pause for a full collection together instead of each
        # one stalling all the others at the next gather.
        if self._dp_gathers == 0:
            self._start_lockstep()
        self._dp_gathers += 1
        if self._dp_gathers < self._next_collect:
            return
        start = time.perf_counter()
        gc.collect()
        self._pause_us = max(1, round((time.perf_counter() - start) * 1e6))
        self._collected_at = self._dp_gathers
        # Kept if no gather reports the pause back.
        self._next_collect = self._dp_gathers + self._interval

    def _on_dp_gc_pause(self, max_pause_us: int) -> None:
        # The longest pause across ranks is the same on every rank, and so is
        # every interval derived from it: the ranks stay in lockstep.
        if max_pause_us <= 0 or self._collected_at is None:
            return
        self._interval = self._next_interval(max_pause_us)
        self._next_collect = self._collected_at + self._interval
        self._collected_at = None
        self._pause_us = 0

    def _next_interval(self, pause_us: int) -> int:
        # A pass scans the live objects created since the freeze (a fixed part)
        # and the garbage of its interval (a part per gather). Fit the latter
        # over the recent passes once their intervals differ (until then,
        # assume the whole pause is garbage), and take the fixed part from the
        # latest pass so it follows the live heap. Every rank has the same
        # inputs in the same order, so it computes the same interval.
        n = self._interval
        self._passes.append((n, pause_us))
        ns = [x for x, _ in self._passes]
        if max(ns) >= LOCKSTEP_MIN_FIT_RATIO * min(ns):
            mean_n = sum(ns) / len(ns)
            mean_p = sum(p for _, p in self._passes) / len(ns)
            slope = sum((x - mean_n) * (p - mean_p) for x, p in self._passes) / sum(
                (x - mean_n) ** 2 for x in ns
            )
            self._per_gather_us = max(slope, 0.0)
        if self._per_gather_us is None:
            fixed, per_gather = 0.0, pause_us / n
        else:
            per_gather = self._per_gather_us
            fixed = max(pause_us - per_gather * n, 0.0)
        # No interval makes a pass shorter than its fixed part: past the budget,
        # allow twice that part and spread it over a longer interval.
        budget = max(LOCKSTEP_PAUSE_BUDGET_US, 2 * fixed)
        target = (budget - fixed) / per_gather if per_gather > 0 else math.inf
        target = min(target, n * LOCKSTEP_MAX_GROWTH)
        return int(min(max(target, LOCKSTEP_MIN_GATHERS), LOCKSTEP_MAX_GATHERS))

    def _start_lockstep(self) -> None:
        # Freeze what startup built (weights, CUDA graphs, caches), so each
        # lockstep collection scans only the objects created since.
        freeze_gc("Scheduler")
        threshold0, threshold1, _ = gc.get_threshold()
        gc.set_threshold(threshold0, threshold1, NO_AUTOMATIC_FULL_GC)
        logger.info(
            "Froze the heap and turned off automatic full GC; attention-DP ranks "
            "run full collections in lockstep, sized to "
            f"~{LOCKSTEP_PAUSE_BUDGET_US / 1000:g} ms per pass."
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


def _lockstep_allowed(gc_threshold: Optional[Sequence[int]]) -> bool:
    if gc_threshold and len(gc_threshold) == 3:
        # The post-warmup /freeze_gc applies --gc-threshold in the scheduler and
        # would turn automatic full collections back on.
        logger.info(
            "Lockstep full GC across attention-DP ranks is off: --gc-threshold "
            "sets the third threshold."
        )
        return False
    if not third_threshold_works():
        logger.info(
            "Lockstep full GC across attention-DP ranks is off: this Python's "
            "collector has no third threshold to turn off automatic full "
            "collections."
        )
        return False
    return True
