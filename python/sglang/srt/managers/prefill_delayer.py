import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple, NamedTuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.metrics.collector import SchedulerMetricsCollector

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _State:
    delayed_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)


class _NegotiateOutput(NamedTuple):
    allow_prefill: bool


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
        logger.info(
            f"PrefillDelayer initialized with "
            f"max_delay_passes={self._max_delay_passes} "
            f"token_usage_low_watermark={self._token_usage_low_watermark}"
        )

        self._global_info_buffer = torch.empty(
            (dp_size, attn_tp_size, 2),
            dtype=torch.int64,
            device="cpu",
        )
        self.cpu_group = tp_worker.get_tp_group().cpu_group

        self._metrics_collector = metrics_collector

        self._curr_state: Optional[_State] = None

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
            self, local_prefillable: bool, token_usage: float
    ) -> _NegotiateOutput:
        self._curr_state, out = self._negotiate_should_allow_prefill_pure(
            prev_state=self._curr_state,
            local_prefillable=local_prefillable,
            token_usage=token_usage,
        )
        return out

    # (Almost) pure function
    def _negotiate_should_allow_prefill_pure(
        self,
        prev_state: _State,
        local_prefillable: bool,
        token_usage: float,
    ) -> Tuple[_State, _NegotiateOutput]:
        # Compute local states
        local_token_watermark_force_allow = (
            local_prefillable
            and ((x := self._token_usage_low_watermark) is not None)
            and (token_usage < x)
        )

        # Gather global states
        global_prefillable, global_token_watermark_force_allow = self._gather_info(
            local_prefillable=local_prefillable,
            local_token_watermark_force_allow=local_token_watermark_force_allow,
        )

        # Compute derived global states
        num_prefillable = global_prefillable.sum().item()
        num_token_watermark_force_allow = global_token_watermark_force_allow.sum().item()
        global_exists_force_allow = global_token_watermark_force_allow.max().item() > 0
        global_exists_not_prefillable = global_prefillable.min().item() == 0
        global_exists_prefillable = global_prefillable.max().item() > 0
        global_mixed_prefillable = (
            global_exists_not_prefillable and global_exists_prefillable
        )

        if global_exists_force_allow:
            self._record_outcome_and_reset(
                debug_outcome="token_watermark_force_allow",
                debug_num_prefillable=num_prefillable,
                debug_num_token_watermark_force_allow=num_token_watermark_force_allow,
            )
            return _NegotiateOutput(allow_prefill=True)

        prev_delayed_count = prev_state.delayed_count if prev_state else 0
        if global_mixed_prefillable and prev_delayed_count < self._max_delay_passes - 1:
            next_state = _State() or prev_state
            next_state = dataclasses.replace(
                next_state,
                delayed_count=next_state.delayed_count + 1,
            )
            return _NegotiateOutput(allow_prefill=False)

        is_timeout = global_mixed_prefillable
        exist_previous_wait = prev_state is not None
        self._record_outcome_and_reset(
            debug_outcome=(
                "wait_timeout"
                if is_timeout
                else (
                    "wait_success_all_prefillable"
                    if exist_previous_wait
                    else "no_wait_all_prefillable"
                )
            ),
            debug_num_prefillable=num_prefillable,
            debug_num_token_watermark_force_allow=num_token_watermark_force_allow,
        )
        return _NegotiateOutput(allow_prefill=True)

    def _gather_info(self, local_prefillable: bool, local_token_watermark_force_allow: bool):
        local_info = torch.tensor(
            [int(local_prefillable), int(local_token_watermark_force_allow)],
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
        self._result: Optional[_NegotiateOutput] = None

    @property
    def _called(self) -> bool:
        return self._result is not None

    def finalize(self, *, actual_prefill: bool):
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)

        TODO_report_metrics

    def negotiate_should_allow_prefill(self, local_prefillable: bool) -> bool:
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
            )
        return self._result.allow_prefill


def _record_outcome(
    debug_outcome: str, debug_num_prefillable: int, debug_num_token_watermark_force_allow: int
) -> None:
    if _DEBUG_LOG:
        if debug_outcome == "wait_timeout":
            logger.info(
                f"PrefillDelayer timeout thus not forbid prefill "
                f"(num_prefillable={debug_num_prefillable})"
            )
        elif debug_outcome == "token_watermark_force_allow":
            logger.info(
                f"PrefillDelayer force allow prefill due to low watermark. "
                f"(num_prefillable={debug_num_prefillable}, "
                f"num_token_watermark_force_allow={debug_num_token_watermark_force_allow})"
            )
        else:
            assert debug_outcome in {
                "wait_success_all_prefillable",
                "no_wait_all_prefillable",
            }

    if (collector := _metrics_collector) is not None:
        if (x := _curr_state) is not None:
            wait_seconds = time.perf_counter() - x.start_time
            forward_passes = x.delayed_count
        else:
            wait_seconds = forward_passes = 0
        collector.observe_prefill_delayer_wait(
            forward_passes=forward_passes,
            wait_seconds=wait_seconds,
            outcome=debug_outcome,
        )

