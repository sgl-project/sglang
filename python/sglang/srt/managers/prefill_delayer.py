import torch

from sglang.srt.environ import envs


class PrefillDelayer:
    def __init__(
        self, dp_size, attn_tp_size, tp_worker, max_running_requests, server_args
    ):
        self.dp_size = dp_size
        self.attn_tp_size = attn_tp_size
        self.global_waiting_queue_len = torch.empty(
            (self.dp_size, self.attn_tp_size, 2),
            dtype=torch.int64,
            device="cpu",
        )
        self.cpu_group = tp_worker.get_tp_group().cpu_group
        self.max_running_requests = max_running_requests
        self.delayed_count = 0
        self.max_delay_passes = envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.get()
        assert (
            server_args.schedule_policy == "fcfs"
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, schedule_policy must be 'fcfs'. '{server_args.schedule_policy}' is not supported."
        assert (
            server_args.enable_dp_attention
        ), "To use SCHEDULER_DECREASE_PREFILL_IDLE, enable_dp_attention must be enabled."
        assert (
            server_args.disaggregation_mode == "null"
        ), "To use SCHEDULER_DECREASE_PREFILL_IDLE, disaggregation_mode must be null."
        assert (
            not server_args.disable_overlap_schedule
        ), "To use SCHEDULER_DECREASE_PREFILL_IDLE, disable_overlap_schedule must be False."

    def _gather_info(self, waiting_queue_len: int, is_local_idle: bool):
        local_queue_len = torch.tensor(
            [waiting_queue_len, int(is_local_idle)],
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

    def should_allow_prefill(self, waiting_queue_len: int, is_local_idle: bool) -> bool:
        tp0_info = self._gather_info(waiting_queue_len=waiting_queue_len, is_local_idle=is_local_idle)
        min_queue_len = int(tp0_info[:, 0].min().item())
        max_queue_len = int(tp0_info[:, 0].max().item())
        has_idle = bool(tp0_info[:, 1].max().item())

        if has_idle:
            return True

        if min_queue_len == 0 and max_queue_len > 0:
            self.delayed_count += 1
            if self.delayed_count < self.max_delay_passes:
                return False

        self.delayed_count = 0
        return True
