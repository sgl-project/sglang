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
    hold yet: the batch launched last, or under a loop that joins its rows at
    launch, the steps in flight. Any may be None, and the running and last
    batch may be the same. Under PP they are the current micro-batch's."""
    if scheduler.rows_join_at_launch:
        return [
            scheduler.running_batch,
            *(record.batch for record in scheduler.inflight_steps),
        ]
    return [scheduler.running_batch, scheduler.last_batch]
