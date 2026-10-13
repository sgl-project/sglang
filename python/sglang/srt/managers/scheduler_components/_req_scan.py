"""A temporary module that reads the holders of running and in-flight requests,
and goes away once callers read the components that own them."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


def running_and_inflight_batches(
    scheduler: Scheduler,
) -> List[Optional[ScheduleBatch]]:
    """The running batch, then the launched batches whose requests no other
    holder lists yet: the steps in flight, or under a pipeline loop the step
    the current slot launched last, until the slot launches again. Any may be
    None; under PP the running batch is the current slot's rows."""
    if not scheduler.inflight_steps.ring:
        return [
            scheduler.running_batch,
            *(record.batch for record in scheduler.inflight_steps),
        ]
    slots = scheduler.inflight_steps.slot_batches()
    return [
        scheduler.running_batch,
        slots[scheduler.inflight_steps.step % len(slots)],
    ]
