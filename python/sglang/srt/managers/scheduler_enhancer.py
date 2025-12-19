import torch


class SchedulerEnhancer:
    def __init__(
        self, dp_size, attn_tp_size, tp_worker, max_running_requests, server_args
    ):
        self.dp_size = dp_size
        self.attn_tp_size = attn_tp_size
        self.global_batch_size = torch.empty(
            (self.dp_size, self.attn_tp_size, 1),
            dtype=torch.int64,
            device="cpu",
        )
        self.cpu_group = tp_worker.get_tp_group().cpu_group
        self.max_running_requests = max_running_requests
        self.stable_count = 0
        # If scheduling is performed 30 times and some dp units are still at full load, the prefill-prioritized scheduling strategy will still be used.
        self.max_stable_count = 30
        assert (
            server_args.schedule_policy == "fcfs"
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, schedule_policy must be 'fcfs'. '{self.schedule_policy}' is not supported."
        assert (
            server_args.enable_dp_attention == True
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, enable_dp_attention must be enable."
        assert (
            server_args.disaggregation_mode == "null"
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, disaggregation_mode must be null."
        assert (
            server_args.disable_overlap_schedule == False
        ), f"To use SCHEDULER_DECREASE_PREFILL_IDLE, disable_overlap_schedule must be False."

    def get_schedule_info(self, running_batch):
        local_batch_size = torch.tensor(
            [
                running_batch.batch_size(),
            ],
            device="cpu",
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self.global_batch_size.flatten(),
            local_batch_size,
            group=self.cpu_group,
        )
        tp0_info = self.global_batch_size[:, 0, :]
        return tp0_info

    def get_schedule_decision(self, running_batch):
        tp0_info = self.get_schedule_info(running_batch)
        if (
            int(tp0_info[:, 0].min().item()) < self.max_running_requests
            and int(tp0_info[:, 0].max().item()) == self.max_running_requests
        ):
            self.stable_count += 1
            if self.stable_count < self.max_stable_count:
                return False
        self.stable_count = 0
        return True
