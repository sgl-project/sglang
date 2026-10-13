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
    """The running batch, then the batch launched last; either may be None,
    and they may be the same batch. Under PP they are the current
    micro-batch's. Callers that also read the overlap result queue or the PP
    micro-batch lists read those themselves."""
    return [scheduler.running_batch, scheduler.last_batch]
