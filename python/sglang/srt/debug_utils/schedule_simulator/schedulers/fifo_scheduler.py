from typing import TYPE_CHECKING

from sglang.srt.debug_utils.schedule_simulator.schedulers.base import SchedulerPolicy

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class FIFOScheduler(SchedulerPolicy):
    def schedule(self, gpu_state: "GPUState") -> None:
        while not gpu_state.is_valid() and gpu_state.running_requests:
            gpu_state.evict_request(gpu_state.running_requests[-1])

        if not gpu_state.pending_requests:
            return

        seen_groups = set()
        cur_tokens = 0
        for req in gpu_state.running_requests:
            is_shared = req.group_id is not None and req.group_id in seen_groups
            cur_tokens += req.seq_len() - (req.prefix_len if is_shared else 0)
            if req.group_id is not None:
                seen_groups.add(req.group_id)

        max_tokens = gpu_state.max_total_tokens
        remaining_pending = []
        admitted_any = False
        for req in gpu_state.pending_requests:
            is_shared = req.group_id is not None and req.group_id in seen_groups
            req_tokens = req.seq_len() - (req.prefix_len if is_shared else 0)
            if cur_tokens + req_tokens <= max_tokens:
                cur_tokens += req_tokens
                if req.group_id is not None:
                    seen_groups.add(req.group_id)
                gpu_state.running_requests.append(req)
                admitted_any = True
            else:
                remaining_pending.append(req)

        if admitted_any:
            gpu_state.pending_requests[:] = remaining_pending
