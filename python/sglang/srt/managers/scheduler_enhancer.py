import torch
from sglang.srt.environ import envs


class SchedulerEnhancer:
    def __init__(
        self, dp_size, attn_tp_size, tp_worker, max_running_requests, server_args
    ):
        self.dp_size = dp_size
        self.attn_tp_size = attn_tp_size
        self.tp_worker = tp_worker
        self.disable_overlap_schedule = server_args.disable_overlap_schedule
        self.group = None
        if server_args.disable_overlap_schedule:
            self.device = tp_worker.get_tp_group().device
        else:
            self.device = "cpu"

        self.global_batch_size = torch.empty(
            (self.dp_size, self.attn_tp_size, 2),
            dtype=torch.int64,
            device=self.device,
        )

        self.max_running_requests = max_running_requests
        self.stable_count = 0
        # If scheduling is performed 200 times and some dp units are still at full load, the prefill-prioritized scheduling strategy will still be used.
        self.max_stable_count = 200
        self.server_args = server_args
        assert (
            server_args.schedule_policy == "fcfs"
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, schedule_policy must be 'fcfs'. '{self.schedule_policy}' is not supported."
        assert (
            server_args.disaggregation_mode == "null"
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, disaggregation_mode must be null."

    def get_schedule_info(self, running_batch, max_prefill_bs):
        if self.group is None:
            if self.disable_overlap_schedule:
                self.group = self.tp_worker.get_tp_group().device_group
            else:
                self.group = self.tp_worker.get_tp_group().cpu_group
        local_batch_size = torch.tensor(
            [
                running_batch.batch_size(),
                max_prefill_bs,
            ],
            device=self.device,
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self.global_batch_size.flatten(),
            local_batch_size,
            group=self.group,
        )
        tp0_info = self.global_batch_size[:, 0, :]
        return tp0_info

    def get_schedule_decision(self, running_batch, max_prefill_bs):
        if self.server_args.enable_dp_attention:
            tp0_info = self.get_schedule_info(running_batch, max_prefill_bs)
            prefill_delay_level1 = (
                envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.get() == 1
                and self.max_running_requests - int(tp0_info[:, 0].max().item())
                < int(tp0_info[:, 1].max().item())
            )
            prefill_delay_level2 = (
                envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.get() == 2
                and int(tp0_info[:, 0].min().item()) < self.max_running_requests
                and int(tp0_info[:, 0].max().item()) == self.max_running_requests
            )
            if prefill_delay_level1 or prefill_delay_level2:
                self.stable_count += 1
                if self.stable_count < self.max_stable_count:
                    return False
        elif self.max_running_requests - running_batch.batch_size() < max_prefill_bs:
            self.stable_count += 1
            if self.stable_count < self.max_stable_count:
                return False

        self.stable_count = 0
        return True
