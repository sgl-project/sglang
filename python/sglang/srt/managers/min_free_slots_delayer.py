from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash: bool = False,
) -> Optional[int]:
    """Resolve the min-free-slots threshold (None = disabled).

    A user value (>1) is capped to the DFlash formula so the trigger never
    delays more aggressively than the legacy heuristic. When unset, DFlash
    workloads fall back to the formula (preserving the always-on behavior);
    other workloads stay disabled. Also disabled when max_running_requests < 8.
    """
    max_running_requests = max(0, int(max_running_requests))
    formula = min(4, max(2, (max_running_requests + 5) // 6))
    if user_value is None:
        user_value = formula if is_dflash else None

    if user_value is None or user_value <= 1:
        return None
    if max_running_requests < 8:
        return None
    return min(user_value, formula)


class MinFreeSlotsDelayer:
    """Delay fresh prefill admissions until at least ``min_free_slots`` running-
    request slots free up, batching them into one admission instead of one at a
    time. Useful when each admission is expensive (e.g. DFlash's draft prefill).

    Per-rank local: running-batch slots are private to each DP rank, so a rank
    with free slots does not wait for a congested peer.
    """

    def __init__(
        self,
        min_free_slots: int,
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
    ):
        self._min_free_slots = min_free_slots
        self._metrics_collector = metrics_collector

    def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
        delayed = running_bs > 0 and num_allocatable_reqs < self._min_free_slots
        if self._metrics_collector is not None:
            self._metrics_collector.observe_min_free_slots_check(
                running_bs=running_bs,
                num_allocatable_reqs=num_allocatable_reqs,
                delayed=delayed,
            )
        return delayed
