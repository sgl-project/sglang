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
    """The running batch, then the launched batches whose requests it does not
    hold yet: the batch launched last, or when the rows hold the steps
    launched, the steps in flight. Any may be None, and the running and last
    batch may be the same. Under PP they are the current micro-batch's."""
    if rows_hold_launched(scheduler):
        return [
            scheduler.running_batch,
            *(record.batch for record in scheduler.inflight_steps),
        ]
    return [scheduler.running_batch, scheduler.last_batch]


def rows_hold_launched(scheduler: Scheduler) -> bool:
    """Whether the rows hold every unfinished request of the steps launched,
    except the steps in flight and the request still chunking: the loops that
    join at launch, and the PD decode loops, whose step is a copy of the rows."""
    return scheduler.rows_join_before_plan or scheduler.disagg_decode_launches_copy
