import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


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

    def _negotiate_should_allow_prefill(self, local_prefillable: bool) -> bool:
        tp0_info = self._gather_info(local_prefillable=local_prefillable)
        global_prefillable = tp0_info[:, 0]
        global_exists_not_prefillable = global_prefillable.min().item() == 0
        global_exists_prefillable = global_prefillable.max().item() > 0
        global_mixed_prefillable = (
            global_exists_not_prefillable and global_exists_prefillable
        )

        if global_mixed_prefillable:
            self.curr_delayed_count += 1
            if self.curr_delayed_count < self.max_delay_passes:
                return False

        if _DEBUG_LOG and global_mixed_prefillable:
            logger.info(
                f"PrefillDelayer timeout thus not forbid prefill (prefillable: {global_prefillable.sum()})"
            )

        self.curr_delayed_count = 0
        return True

    def _gather_info(self, local_prefillable: bool):
        local_info = torch.tensor(
            [int(local_prefillable)],
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


class PrefillDelayerSinglePassExecutor:
    def __init__(self, prefill_delayer: PrefillDelayer):
        self._prefill_delayer = prefill_delayer
        self._result: Optional[bool] = None

    @property
    def _called(self) -> bool:
        return self._result is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)
        return False

    def negotiate_should_allow_prefill(self, local_prefillable: bool) -> bool:
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable
            )
        return self._result
