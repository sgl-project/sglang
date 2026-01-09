import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

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
    next_state: Optional[_State]
    prefillable: str
    allow: bool
    reason: str
    num_prefillable: int
    num_token_watermark_force_allow: int


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
        self._cpu_group = tp_worker.get_tp_group().cpu_group

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
        out = self._negotiate_should_allow_prefill_pure(
            prev_state=self._curr_state,
            local_prefillable=local_prefillable,
            token_usage=token_usage,
        )
        self._curr_state = out.next_state
        return out

    # (Almost) pure function, do not modify self state
    def _negotiate_should_allow_prefill_pure(
        self,
        prev_state: Optional[_State],
        local_prefillable: bool,
        token_usage: float,
    ) -> _NegotiateOutput:
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
        if global_prefillable.min().item() > 0:
            prefillable_status = "all"
        elif global_prefillable.max().item() == 0:
            prefillable_status = "none"
        else:
            prefillable_status = "mixed"
        global_exists_token_watermark_force_allow = (
            global_token_watermark_force_allow.max().item() > 0
        )
        debug_info = dict(
            prefillable_status=prefillable_status,
            num_prefillable=global_prefillable.sum().item(),
            num_token_watermark_force_allow=global_token_watermark_force_allow.sum().item(),
        )

        # Compute outputs
        if prefillable_status == "all":
            exist_previous_wait = prev_state is not None
            return _NegotiateOutput(
                next_state=None,
                allow=True,
                reason="wait_success" if exist_previous_wait else "no_wait",
                **debug_info,
            )
        elif prefillable_status == "none":
            return _NegotiateOutput(
                next_state=None,
                # It does not matter whether we allow or not, thus we allow for simplicity
                allow=True,
                reason="",
                **debug_info,
            )
        elif prefillable_status == "mixed":
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    allow=True,
                    reason="token_watermark",
                    **debug_info,
                )

            prev_delayed_count = prev_state.delayed_count if prev_state else 0
            if prev_delayed_count < self._max_delay_passes - 1:
                next_state = dataclasses.replace(
                    prev_state,
                    delayed_count=prev_state.delayed_count + 1,
                )
                return _NegotiateOutput(
                    next_state=next_state,
                    allow=False,
                    reason="delay",
                    **debug_info,
                )
            else:
                return _NegotiateOutput(
                    next_state=None,
                    allow=True,
                    reason="wait_timeout",
                    **debug_info,
                )
        else:
            raise NotImplementedError

    def _gather_info(
        self, local_prefillable: bool, local_token_watermark_force_allow: bool
    ):
        local_info = torch.tensor(
            [int(local_prefillable), int(local_token_watermark_force_allow)],
            device="cpu",
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self._cpu_group,
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

        _record_single_pass_result(
            actual_prefill=actual_prefill,
            negotiate_output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
        )

    def negotiate_should_allow_prefill(self, local_prefillable: bool) -> bool:
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
            )
        return self._result.allow


def _record_single_pass_result(
    actual_prefill: bool,
    output: _NegotiateOutput,
    metrics_collector: Optional["SchedulerMetricsCollector"],
) -> None:
    if _DEBUG_LOG:
        if output.allow and (output.reason == "wait_timeout"):
            logger.info(
                f"PrefillDelayer timeout thus not forbid prefill "
                f"(num_prefillable={output.num_prefillable})"
            )
        elif output.allow and (output.reason == "token_watermark"):
            logger.info(
                f"PrefillDelayer force allow prefill due to low watermark. "
                f"(num_prefillable={output.num_prefillable}, "
                f"num_token_watermark_force_allow={output.num_token_watermark_force_allow})"
            )
        else:
            assert output.reason in {
                "",
                "wait_success",
                "no_wait",
            }

    if metrics_collector is not None:
        if (s := output.next_state) is not None:
            wait_seconds = time.perf_counter() - s.start_time
            forward_passes = s.delayed_count
        else:
            wait_seconds = forward_passes = 0
        metrics_collector.observe_prefill_delayer_wait(
            forward_passes=forward_passes,
            wait_seconds=wait_seconds,
            prefillable=output.prefillable,
            allow=output.allow,
            reason=output.reason,
            actual_prefill=actual_prefill,
        )
