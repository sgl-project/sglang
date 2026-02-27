from typing import TYPE_CHECKING

from sglang.srt.debug_utils.schedule_simulator.schedulers.base import SchedulerPolicy

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class FIFOScheduler(SchedulerPolicy):
    def schedule(self, gpu_state: "GPUState") -> None:
        while not gpu_state.is_valid() and gpu_state.running_requests:
            gpu_state.evict_request(gpu_state.running_requests[-1])

        for req in list(gpu_state.pending_requests):
            if gpu_state.total_seq_len(extra_reqs=[req]) <= gpu_state.max_total_tokens:
                gpu_state.start_request(req)
