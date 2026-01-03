from typing import TYPE_CHECKING

from sglang.srt.debug_utils.schedule_simulator.schedulers.base import (
    ScheduleDecision,
    SchedulerPolicy,
)

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class FIFOScheduler(SchedulerPolicy):
    def __init__(self, max_total_tokens: int = 100000):
        self.max_total_tokens = max_total_tokens

    def schedule(self, gpu_state: "GPUState") -> ScheduleDecision:
        to_run = []
        to_preempt = []

        current_tokens = gpu_state.total_seq_len()

        # If over budget, preempt requests (LIFO order - last added first)
        running = list(gpu_state.running_requests)
        while current_tokens > self.max_total_tokens and running:
            req = running.pop()
            to_preempt.append(req)
            current_tokens -= req.seq_len()

        # Try to add pending requests
        for req in gpu_state.pending_requests:
            req_tokens = req.seq_len()
            if current_tokens + req_tokens <= self.max_total_tokens:
                to_run.append(req)
                current_tokens += req_tokens

        return ScheduleDecision(to_run=to_run, to_preempt=to_preempt)
