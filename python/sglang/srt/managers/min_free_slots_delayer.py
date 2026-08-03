from typing import Optional


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
) -> Optional[int]:
    """Resolve the min-free-slots threshold (None = disabled).

    DFlash workloads use the legacy formula (preserving the always-on
    behavior). Other workloads use an explicit user value when provided.
    Also disabled when max_running_requests < 8.
    """
    max_running_requests = max(0, int(max_running_requests))
    formula = min(4, max(2, (max_running_requests + 5) // 6))
    if is_dflash_family:
        min_free_slots = formula
    elif user_value is not None:
        min_free_slots = min(user_value, max_running_requests)
    else:
        min_free_slots = None

    if min_free_slots is None or min_free_slots <= 1:
        return None
    if max_running_requests < 8:
        return None
    return min_free_slots


class MinFreeSlotsDelayer:
    """Delay fresh prefill admissions until at least ``min_free_slots`` running-
    request slots free up, batching them into one admission instead of one at a
    time. Useful when each admission is expensive (e.g. DFlash's draft prefill).

    Per-rank local: running-batch slots are private to each DP rank, so a rank
    with free slots does not wait for a congested peer.
    """

    def __init__(self, min_free_slots: int):
        self._min_free_slots = min_free_slots

    def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
        return running_bs > 0 and num_allocatable_reqs < self._min_free_slots
