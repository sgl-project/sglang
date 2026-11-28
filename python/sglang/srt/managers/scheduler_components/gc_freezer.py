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
NO_AUTOMATIC_FULL_GC = 10**9

# About what lockstep passes took (2-6.6 ms) on a DEP32 decode deployment.
LOCKSTEP_PAUSE_BUDGET_US = 5_000
# Intervals count loaded gathers (some rank has tokens);
# idle gathers get a pass every this many instead.
LOCKSTEP_IDLE_GATHERS = 10_000
LOCKSTEP_INITIAL_GATHERS = 1_000
LOCKSTEP_MIN_GATHERS = 16
LOCKSTEP_MAX_GATHERS = 100_000
LOCKSTEP_MAX_GROWTH = 2
LOCKSTEP_FIT_PASSES = 8
LOCKSTEP_MIN_FIT_RATIO = 1.25


def third_threshold_works() -> bool:
    """Whether the third GC threshold holds; CPython 3.14.0-3.14.4 ignores it."""
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
        self._lockstep_started = False
        self._loaded_gathers = 0
        self._idle_gathers = 0  # since the last pass
        self._fit_pass = False
        self._next_collect = LOCKSTEP_INITIAL_GATHERS
        self._interval = LOCKSTEP_INITIAL_GATHERS
        self._collected_at: Optional[int] = None
        self._pause_us = 0
        self._passes: collections.deque = collections.deque(maxlen=LOCKSTEP_FIT_PASSES)
        self._per_gather_us: Optional[float] = None
        self._fixed_us: Optional[float] = None
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
            after_gather=self._after_dp_gather,
        )

    def _before_dp_gather(self) -> None:
        if not self._lockstep_started:
            self._lockstep_started = True
            self._start_lockstep()
        self._fit_pass = self._loaded_gathers >= self._next_collect
        if not self._fit_pass and self._idle_gathers < LOCKSTEP_IDLE_GATHERS:
            return
        start = time.perf_counter()
        gc.collect()
        self._pause_us = max(1, round((time.perf_counter() - start) * 1e6))
        self._collected_at = self._loaded_gathers
        self._idle_gathers = 0
        # Kept if no gather reports the pause back.
        self._next_collect = self._loaded_gathers + self._interval

    def _after_dp_gather(self, max_pause_us: int, any_tokens: bool) -> None:
        # Both values are gathered, so every rank derives the same counts and intervals.
        if any_tokens:
            self._loaded_gathers += 1
        else:
            self._idle_gathers += 1
        if max_pause_us <= 0 or self._collected_at is None:
            return
        # An idle pass restarts the interval but does not size it.
        if self._fit_pass:
            self._interval = self._next_interval(max_pause_us)
        self._next_collect = self._collected_at + self._interval
        logger.debug(
            f"Lockstep full GC ({'loaded' if self._fit_pass else 'idle'}) at "
            f"loaded gather {self._collected_at}: longest pause {max_pause_us} "
            f"us across ranks; next pass in {self._interval} loaded gathers."
        )
        self._collected_at = None
        self._pause_us = 0

    def _next_interval(self, pause_us: int) -> int:
        # pause = live objects since the freeze (fixed) + interval garbage (per gather).
        n = self._interval
        if self._per_gather_us is not None:
            predicted = self._fixed_us + self._per_gather_us * n
            if pause_us > LOCKSTEP_MAX_GROWTH * max(
                predicted, LOCKSTEP_PAUSE_BUDGET_US
            ):
                self._passes.clear()
                self._per_gather_us = None
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
        self._fixed_us = fixed
        # No interval beats the fixed part; past the budget, allow twice it.
        budget = max(LOCKSTEP_PAUSE_BUDGET_US, 2 * fixed)
        target = (budget - fixed) / per_gather if per_gather > 0 else math.inf
        target = min(target, n * LOCKSTEP_MAX_GROWTH)
        return int(min(max(target, LOCKSTEP_MIN_GATHERS), LOCKSTEP_MAX_GATHERS))

    def _start_lockstep(self) -> None:
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
        # The post-warmup /freeze_gc would reapply it and re-enable automatic full GC.
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
