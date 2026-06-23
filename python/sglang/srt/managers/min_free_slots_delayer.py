from typing import Optional


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash: bool = False,
) -> Optional[int]:
    """Resolve the min-free-slots threshold.

    A user value (>1) is honored but capped to the DFlash formula so the
    trigger never delays more aggressively than the historical heuristic:
        min(user_value, min(4, max(2, (max_running_requests + 5) // 6)))

    When unset, DFlash workloads fall back to the formula automatically
    (matching the legacy always-on behavior); other workloads stay disabled.

    Returns None (disabled) when the resolved value is <= 1 or
    max_running_requests < 8.
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
    request slots have freed up, so they are admitted as one batch instead of
    one request at a time.

    Useful when each admission is disproportionately expensive (e.g. DFlash's
    separate draft prefill pass). The decision is per-rank local: running-batch
    slots are private to each DP rank, so a rank with enough free slots does not
    wait for a congested peer.
    """

    def __init__(self, min_free_slots: int):
        self._min_free_slots = min_free_slots

    def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
        return running_bs > 0 and num_allocatable_reqs < self._min_free_slots
