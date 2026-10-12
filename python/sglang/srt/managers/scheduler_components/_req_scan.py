"""A temporary module that reads the holders of running and in-flight requests,
and goes away once callers read the components that own them."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


def running_and_inflight_batches(
    scheduler: Scheduler,
) -> List[Optional[ScheduleBatch]]:
    """The running batch, then the launched batches whose requests no other
    holder lists yet: the batch launched last, or when another holder lists
    the steps launched, the steps in flight. Any may be None, and the running and last
    batch may be the same. Under PP they are the current slot's rows and the
    step that slot launched last."""
    if launched_reqs_held(scheduler):
        return [
            scheduler.running_batch,
            *(record.batch for record in scheduler.inflight_steps),
        ]
    if scheduler.inflight_steps.ring:
        # The step the current pipeline slot launched last, until the slot
        # launches again.
        slots = scheduler.inflight_steps.slot_batches()
        return [
            scheduler.running_batch,
            slots[scheduler.inflight_steps.step % len(slots)],
        ]
    return [scheduler.running_batch, scheduler.last_batch]


def launched_reqs_held(scheduler: Scheduler) -> bool:
    """Whether another holder lists every unfinished request of the steps
    launched, except the steps in flight and the request still chunking: the
    rows in the normal and overlap loops, which join before the plan, and in
    the PD decode loops, whose step is a copy of the rows; the KV transfer
    and waiting queues in the PD prefill loops, which take the step's
    requests at its commit. Not the pipeline loops."""
    return (
        scheduler.rows_join_before_plan
        or scheduler.disaggregation_mode
        in (DisaggregationMode.DECODE, DisaggregationMode.PREFILL)
    ) and not scheduler.inflight_steps.ring
