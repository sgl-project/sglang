from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def select_queue_evictions(
    *,
    waiting_queue: list[Req],
    incoming: Req,
    max_requests: int | None,
    max_tokens: int | None,
    enable_priority: bool,
    low_priority_values_first: bool,
) -> list[Req] | None:
    """Return evictions needed for admission, or None if admission is impossible."""
    if max_requests is None and max_tokens is None:
        return []

    num_requests = len(waiting_queue) + 1
    num_tokens = (
        sum(len(req.origin_input_ids) for req in waiting_queue)
        + len(incoming.origin_input_ids)
        if max_tokens is not None
        else 0
    )

    def fits() -> bool:
        return (max_requests is None or num_requests <= max_requests) and (
            max_tokens is None or num_tokens <= max_tokens
        )

    if fits():
        return []
    if not enable_priority:
        return None
    if max_tokens is not None and len(incoming.origin_input_ids) > max_tokens:
        return None

    direction = 1 if low_priority_values_first else -1
    candidates = sorted(
        waiting_queue,
        key=lambda req: (
            direction * req.priority,
            req.time_stats.wait_queue_entry_time,
        ),
        reverse=True,
    )
    evictions = []
    for candidate in candidates:
        if direction * incoming.priority >= direction * candidate.priority:
            break
        evictions.append(candidate)
        num_requests -= 1
        if max_tokens is not None:
            num_tokens -= len(candidate.origin_input_ids)
        if fits():
            return evictions
    return None
