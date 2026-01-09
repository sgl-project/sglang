import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.metrics.collector import SchedulerMetricsCollector

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


@dataclass
class _DelayInfo:
    delayed_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)


class PrefillDelayer:
    def __init__(
        self,
        dp_size,
        attn_tp_size,
        tp_worker,
        server_args,
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
    ):
        self._max_delay_passes = envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.get()
        self._token_usage_low_watermark = (
            envs.SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK.get()
        )

        self._global_info_buffer = torch.empty(
            (dp_size, attn_tp_size, 2),
            dtype=torch.int64,
            device="cpu",
        )
        self.cpu_group = tp_worker.get_tp_group().cpu_group

        self._metrics_collector = metrics_collector

        self._curr_delay_info: Optional[_DelayInfo] = None

        assert (
            server_args.enable_dp_attention
        ), "To use PrefillDelayer, enable_dp_attention must be enabled."
        assert (
            server_args.disaggregation_mode == "null"
        ), "To use PrefillDelayer, disaggregation_mode must be null."
        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."

    def _negotiate_should_allow_prefill(
        self, local_prefillable: bool, token_usage: bool
    ) -> bool:
        local_force_allow = (
            local_prefillable
            and ((x := self._token_usage_low_watermark) is not None)
            and (token_usage < x)
        )

        global_prefillable, global_force_allow = self._gather_info(
            local_prefillable=local_prefillable,
            local_force_allow=local_force_allow,
        )
        num_prefillable = global_prefillable.sum().item()
        num_force_allow = global_force_allow.sum().item()

        if global_force_allow.max().item() > 0:
            self._reset(
                outcome="token_usage_watermark",
                debug_num_prefillable=num_prefillable,
                debug_num_force_allow=num_force_allow,
            )
            return True

        global_exists_not_prefillable = global_prefillable.min().item() == 0
        global_exists_prefillable = global_prefillable.max().item() > 0
        global_mixed_prefillable = (
            global_exists_not_prefillable and global_exists_prefillable
        )

        if global_mixed_prefillable:
            if self._curr_delay_info is None:
                self._curr_delay_info = _DelayInfo()
            self._curr_delay_info.delayed_count += 1
            if self._curr_delay_info.delayed_count < self._max_delay_passes:
                return False

        is_timeout = global_mixed_prefillable
        self._reset(
            outcome="timeout" if is_timeout else TODO,
            debug_num_prefillable=num_prefillable,
            debug_num_force_allow=num_force_allow,
        )
        return True

    def _reset(self, outcome: str, debug_num_prefillable: int, debug_num_force_allow: int) -> None:
        if _DEBUG_LOG:
            if outcome == "timeout":
                logger.info(
                    f"PrefillDelayer timeout thus not forbid prefill (num_prefillable={debug_num_prefillable})"
                )
            elif outcome == "token_usage_watermark":
                logger.info(
                    f"PrefillDelayer force allow prefill due to low watermark. "
                    f"(num_prefillable={debug_num_prefillable}, "
                    f"num_force_allow={debug_num_force_allow})"
                )

        TODO_should_this_info_is_not_none
        if self._curr_delay_info is not None and self._metrics_collector is not None:
            wait_seconds = time.perf_counter() - self._curr_delay_info.start_time
            self._metrics_collector.observe_prefill_delayer_wait(
                forward_passes=self._curr_delay_info.delayed_count,
                wait_seconds=wait_seconds,
                outcome=outcome,
            )

        self._curr_delay_info = None

    def _gather_info(self, local_prefillable: bool, local_force_allow: bool):
        local_info = torch.tensor(
            [int(local_prefillable), int(local_force_allow)],
            device="cpu",
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self.cpu_group,
        )
        tp0_info = self._global_info_buffer[:, 0, :]
        return tp0_info[:, 0], tp0_info[:, 1]


class PrefillDelayerSinglePassExecutor:
    def __init__(self, prefill_delayer: PrefillDelayer, token_usage: float):
        self._prefill_delayer = prefill_delayer
        self._token_usage = token_usage
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
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
            )
        return self._result
