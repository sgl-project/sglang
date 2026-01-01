"""Prefill Delayer for DP Attention scenarios.

This module delays prefill requests when DP ranks have imbalanced load to reduce
idle time during prefill/decode phases. It looks at the waiting queue across
all DP ranks to make scheduling decisions.
"""

import torch


class PrefillDelayer:
    """Delays prefill requests to reduce idle time in DP attention scenarios.

    In DP attention with mixed configurations, when all DP ranks are at full capacity
    and one rank finishes a request, starting a prefill causes other ranks to idle.
    This class delays prefill until more DP ranks have available slots, reducing the
    prefill/idle scenario.
    """

    def __init__(
        self, dp_size, attn_tp_size, tp_worker, max_running_requests, server_args
    ):
        self.dp_size = dp_size
        self.attn_tp_size = attn_tp_size
        self.global_waiting_queue_len = torch.empty(
            (self.dp_size, self.attn_tp_size, 1),
            dtype=torch.int64,
            device="cpu",
        )
        self.cpu_group = tp_worker.get_tp_group().cpu_group
        self.max_running_requests = max_running_requests
        self.stable_count = 0
        self.max_stable_count = 30
        assert (
            server_args.schedule_policy == "fcfs"
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, schedule_policy must be 'fcfs'. '{server_args.schedule_policy}' is not supported."
        assert (
            server_args.enable_dp_attention == True
        ), "To use SCHEDULER_DECREASE_PREFILL_IDLE, enable_dp_attention must be enabled."
        assert (
            server_args.disaggregation_mode == "null"
        ), "To use SCHEDULER_DECREASE_PREFILL_IDLE, disaggregation_mode must be null."
        assert (
            server_args.disable_overlap_schedule == False
        ), "To use SCHEDULER_DECREASE_PREFILL_IDLE, disable_overlap_schedule must be False."

    def _gather_waiting_queue_info(self, waiting_queue_len: int):
        """Gather waiting queue length from all DP ranks."""
        local_queue_len = torch.tensor(
            [waiting_queue_len],
            device="cpu",
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self.global_waiting_queue_len.flatten(),
            local_queue_len,
            group=self.cpu_group,
        )
        tp0_info = self.global_waiting_queue_len[:, 0, :]
        return tp0_info

    def should_allow_prefill(self, waiting_queue_len: int) -> bool:
        """Determine if prefill should be allowed based on waiting queue state.

        Returns True if prefill is allowed, False if it should be delayed.

        The heuristic: delay prefill when some DP ranks have empty waiting queues
        while others have requests waiting. This indicates an imbalance that could
        cause idle time if we start prefill. After max_stable_count consecutive
        delays, we allow prefill anyway to avoid excessive TTFT.
        """
        tp0_info = self._gather_waiting_queue_info(waiting_queue_len)
        min_queue_len = int(tp0_info[:, 0].min().item())
        max_queue_len = int(tp0_info[:, 0].max().item())

        if min_queue_len == 0 and max_queue_len > 0:
            self.stable_count += 1
            if self.stable_count < self.max_stable_count:
                return False
        self.stable_count = 0
        return True


# Backward compatibility alias
SchedulerEnhancer = PrefillDelayer
