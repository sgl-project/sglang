from typing import Optional


def resolve_auto_min_free_slots(request_target: int) -> Optional[int]:
    """Scale a refill threshold from the observed request target."""
    request_target = max(0, int(request_target))
    if request_target < 8:
        return None
    return min(4, max(2, (request_target + 5) // 6))


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
) -> Optional[int]:
    """Resolve the min-free-slots threshold (None = disabled).

    An explicit user value always wins, capped by max_running_requests
    (<= 1 disables). When unset, configured DFlash capacity decides whether the
    delay is enabled; the effective automatic threshold is rescaled per pass
    from observed request demand. Other workloads stay disabled.
    """
    max_running_requests = max(0, int(max_running_requests))
    if user_value is not None:
        threshold = min(user_value, max_running_requests)
        return threshold if threshold > 1 else None
    if is_dflash_family:
        return resolve_auto_min_free_slots(max_running_requests)
    return None


class MinFreeSlotsDelayer:
    """Batch replacement prefills after enough active requests complete.

    ``num_allocatable_reqs`` includes request-pool capacity that may never have
    been occupied. Counting it directly makes the delay ineffective whenever
    configured capacity is larger than workload concurrency. Track the request
    target established by actual admissions instead, while allowing a growing
    workload to use idle capacity immediately. Unless explicitly overridden,
    that observed target also bounds the delay in scheduler passes.

    Per-rank local: running-batch slots are private to each DP rank, so a rank
    with free slots does not wait for a congested peer.
    """

    def __init__(
        self,
        min_free_slots: int,
        *,
        scale_with_observed_target: bool = False,
        max_delay_passes: Optional[int] = None,
    ):
        if max_delay_passes is not None and max_delay_passes < 0:
            raise ValueError("max_delay_passes must be non-negative")
        self._min_free_slots = min_free_slots
        self._scale_with_observed_target = scale_with_observed_target
        self._max_delay_passes = max_delay_passes
        self._delay_passes = 0
        self._target_running_bs = 0

    def should_delay(
        self,
        *,
        running_bs: int,
        num_allocatable_reqs: int,
        waiting_bs: int,
        active_running_bs: Optional[int] = None,
    ) -> bool:
        """Return whether this prefill attempt should wait for a larger batch.

        ``running_bs`` is raw scheduler occupancy, including requests that
        finished since the last decode update. ``active_running_bs`` excludes
        those finished requests. The former determines which slots are already
        reusable; the latter distinguishes replacements from real demand
        growth. Callers without deferred completion state may omit the latter.
        """
        running_bs = max(0, int(running_bs))
        if running_bs == 0:
            self._target_running_bs = 0
            self._delay_passes = 0
            return False

        if active_running_bs is None:
            active_running_bs = running_bs
        active_running_bs = min(running_bs, max(0, int(active_running_bs)))
        self._target_running_bs = max(self._target_running_bs, active_running_bs)
        waiting_bs = max(0, int(waiting_bs))
        if waiting_bs == 0:
            self._delay_passes = 0
            return False
        refillable_bs = min(max(0, int(num_allocatable_reqs)), waiting_bs)
        if refillable_bs == 0:
            # No admission is possible on this pass. Preserve a finite delay
            # budget so an alternating capacity signal cannot restart it.
            return False

        active_demand = active_running_bs + refillable_bs
        if active_demand > self._target_running_bs:
            # Do not delay a real increase in workload concurrency.
            self._delay_passes = 0
            return False

        min_free_slots = self._resolve_threshold()
        if min_free_slots is None:
            self._delay_passes = 0
            return False

        # Adapt after a real workload contraction without mistaking a staggered
        # replacement for a lower target. Equality proves that the decode update
        # already removed finished requests; the threshold-sized hysteresis
        # leaves one refill batch of room for replacements still in transit.
        if (
            running_bs == active_running_bs
            and self._target_running_bs - active_demand >= min_free_slots
        ):
            self._target_running_bs = active_demand
            self._delay_passes = 0
            # The waiting work is genuine demand relative to the contracted
            # target. Admit it now instead of immediately delaying against the
            # target that this pass just established.
            return False

        # Finished requests remain in the raw running batch until the next
        # decode update filters them. Do not treat those slots as reusable yet:
        # delaying this pass lets filtering happen and gives replacement
        # requests a chance to accumulate into one prefill batch.
        num_freed_slots = max(0, self._target_running_bs - running_bs)
        if num_freed_slots >= min_free_slots:
            self._delay_passes = 0
            return False
        max_delay_passes = (
            self._target_running_bs
            if self._max_delay_passes is None
            else self._max_delay_passes
        )
        if self._delay_passes >= max_delay_passes:
            self._delay_passes = 0
            return False

        self._delay_passes += 1
        return True

    def _resolve_threshold(self) -> Optional[int]:
        if self._scale_with_observed_target:
            # The constructor threshold enabled automatic batching from
            # configured capacity; observed demand owns the per-pass value.
            return resolve_auto_min_free_slots(self._target_running_bs)
        threshold = min(self._min_free_slots, self._target_running_bs)
        return threshold if threshold > 1 else None

    def on_prefill_admitted(self, *, active_running_bs: int, admitted_bs: int) -> None:
        """Record the request target established by an actual prefill batch."""
        self._target_running_bs = max(
            self._target_running_bs,
            max(0, int(active_running_bs)) + max(0, int(admitted_bs)),
        )
        self._delay_passes = 0
