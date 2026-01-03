import torch

from sglang.srt.environ import envs


class PrefillDelayer:
    def __init__(self, dp_size, attn_tp_size, tp_worker, server_args):
        self.global_info = torch.empty(
            (dp_size, attn_tp_size, 1),
            dtype=torch.int64,
            device="cpu",
        )
        self.cpu_group = tp_worker.get_tp_group().cpu_group

        self.curr_delayed_count = 0
        self.max_delay_passes = envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.get()

        assert (
            server_args.schedule_policy == "fcfs"
        ), f"To use PrefillDelayer, schedule_policy must be 'fcfs'. '{server_args.schedule_policy}' is not supported."
        assert (
            server_args.enable_dp_attention
        ), "To use PrefillDelayer, enable_dp_attention must be enabled."
        assert (
            server_args.disaggregation_mode == "null"
        ), "To use PrefillDelayer, disaggregation_mode must be null."
        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."

    def _gather_info(self, local_can_prefill: int):
        local_info = torch.tensor(
            [local_can_prefill],
            device="cpu",
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self.global_info.flatten(),
            local_info,
            group=self.cpu_group,
        )
        tp0_info = self.global_info[:, 0, :]
        return tp0_info

    def should_allow_prefill(self, local_can_prefill: int) -> bool:
        tp0_info = self._gather_info(local_can_prefill=local_can_prefill)
        global_can_prefill = tp0_info[:, 0]
        global_exists_cannot_prefill = global_can_prefill.min().item() == 0
        global_exists_can_prefill = global_can_prefill.max().item() > 0
        global_exists_idle = bool(tp0_info[:, 1].max().item())

        if (
            (not global_exists_idle)
            and global_exists_cannot_prefill
            and global_exists_can_prefill
        ):
            self.curr_delayed_count += 1
            if self.curr_delayed_count < self.max_delay_passes:
                print("hi branch: delay prefill")
                return False
            print("hi branch: cannot delay prefill since timeout")

        self.curr_delayed_count = 0
        return True
