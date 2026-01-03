from typing import TYPE_CHECKING

from sglang.srt.debug_utils.schedule_simulator.schedulers.base import SchedulerPolicy

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class FIFOScheduler(SchedulerPolicy):
    """FIFO scheduler: runs pending requests in order, evicts in LIFO order."""

    def schedule(self, gpu_state: "GPUState", max_total_tokens: int) -> None:
        # Evict running requests if over budget (LIFO order)
        while not gpu_state.is_valid(max_total_tokens) and gpu_state.running_requests:
            victim = gpu_state.running_requests[-1]
            gpu_state.evict_request(victim)

        # Run pending requests that fit (FIFO order)
        for req in list(gpu_state.pending_requests):
            if gpu_state.total_seq_len() + req.seq_len() <= max_total_tokens:
                gpu_state.run_request(req)
