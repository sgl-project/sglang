from typing import TYPE_CHECKING

from sglang.srt.debug_utils.schedule_simulator.schedulers.base import (
    ScheduleDecision,
    SchedulerPolicy,
)

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class FIFOScheduler(SchedulerPolicy):
    def __init__(self, max_running_requests: int = 256):
        self.max_running_requests = max_running_requests

    def schedule(self, gpu_state: "GPUState") -> ScheduleDecision:
        to_run = []
        available_slots = self.max_running_requests - len(gpu_state.running_requests)
        for req in gpu_state.pending_requests[:available_slots]:
            to_run.append(req)
        return ScheduleDecision(to_run=to_run, to_preempt=[])
