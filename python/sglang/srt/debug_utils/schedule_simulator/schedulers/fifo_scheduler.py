from typing import TYPE_CHECKING, Optional

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.schedulers.base import (
    ScheduleDecision,
    SchedulerPolicy,
)

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class FIFOScheduler(SchedulerPolicy):
    """FIFO scheduler: runs pending requests in order, evicts in LIFO order."""

    def schedule(self, gpu_state: "GPUState") -> ScheduleDecision:
        # Run all pending requests (Simulator handles token limits via eviction)
        return ScheduleDecision(to_run=list(gpu_state.pending_requests))

    def select_victim(self, gpu_state: "GPUState") -> Optional[SimRequest]:
        # LIFO eviction: evict the last running request
        if gpu_state.running_requests:
            return gpu_state.running_requests[-1]
        return None
