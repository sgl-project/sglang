from typing import Optional


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
    admission_cap: Optional[int] = None,
) -> Optional[int]:
    """Resolve the min-free-slots threshold (None = disabled).

    An explicit user value always wins, capped by the running-request limit
    (<= 1 disables). When unset, DFlash workloads fall back to the legacy
    formula (preserving the always-on behavior, disabled when the limit is
    < 8); other workloads stay disabled.

    The limit is ``max_running_requests``, lowered to ``admission_cap`` (the
    per-batch admission budget, ``--pp-max-micro-batch-size``) when that is
    smaller: a threshold above the slots that can ever be free while a request
    runs would hold every admission until the running batch drains.
    """
    max_running_requests = max(0, int(max_running_requests))
    if admission_cap is not None:
        max_running_requests = min(max_running_requests, max(0, int(admission_cap)))
    if user_value is not None:
        threshold = min(user_value, max_running_requests)
        return threshold if threshold > 1 else None
    if is_dflash_family and max_running_requests >= 8:
        return min(4, max(2, (max_running_requests + 5) // 6))
    return None


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
