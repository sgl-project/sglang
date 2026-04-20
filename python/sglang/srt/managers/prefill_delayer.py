import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _State:
    delayed_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def bump_delayed_count(self) -> "_State":
        return dataclasses.replace(self, delayed_count=self.delayed_count + 1)


class _NegotiateOutput(NamedTuple):
    next_state: Optional[_State]
    input_estimation: str
    output_allow: bool
    output_reason: str
    num_prefillable: int
    num_token_watermark_force_allow: int


class PrefillDelayer:
    # Number of int64 fields packed into the cross-rank all-gather tensor.
    # Fields (in order): prefillable, token_watermark_force_allow, running_batch,
    # max_prefill_bs, waiting_queue_len.
    _GATHER_FIELDS = 5

    def __init__(
        self,
        dp_size: int,
        attn_tp_size: int,
        cpu_group,
        server_args,
        max_delay_passes: int,
        token_usage_low_watermark: Optional[float],
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
        device: Optional["torch.device"] = "cpu",
    ):
        self._max_delay_passes = max_delay_passes
        self._token_usage_low_watermark = token_usage_low_watermark
        # Adaptive queue-based trigger (additive on top of slot_trigger).
        # Opt-in: activates only when queue_min_ratio is explicitly set.
        # max_delay_ms gets a default from ServerArgs so users only need to
        # pass --prefill-delayer-queue-min-ratio to enable the new trigger.
        self._queue_min_ratio = getattr(
            server_args, "prefill_delayer_queue_min_ratio", None
        )
        self._max_delay_ms = getattr(server_args, "prefill_delayer_max_delay_ms", None)
        self._queue_trigger_enabled = (
            self._queue_min_ratio is not None and self._max_delay_ms is not None
        )
        logger.info(
            f"PrefillDelayer initialized with "
            f"max_delay_passes={self._max_delay_passes} "
            f"token_usage_low_watermark={self._token_usage_low_watermark} "
            f"queue_min_ratio={self._queue_min_ratio} "
            f"max_delay_ms={self._max_delay_ms} "
            f"queue_trigger_enabled={self._queue_trigger_enabled}"
        )
        self.dp_size = dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        dp_size_dim = dp_size if self.enable_dp_attention else 1
        self._global_info_buffer = torch.empty(
            (dp_size_dim, attn_tp_size, self._GATHER_FIELDS),
            dtype=torch.int64,
            device=device,
        )
        self._cpu_group = cpu_group

        self._metrics_collector = metrics_collector

        self._curr_state: Optional[_State] = None
        self.skip_first_delayer = True

        assert (
            server_args.disaggregation_mode == "null"
        ), "To use PrefillDelayer, disaggregation_mode must be null."
        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."

    def _negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        token_usage: float,
        **kwargs,
    ) -> _NegotiateOutput:
        out = self._negotiate_should_allow_prefill_pure(
            prev_state=self._curr_state,
            local_prefillable=local_prefillable,
            token_usage=token_usage,
            **kwargs,
        )
        self._curr_state = out.next_state
        return out

    # (Almost) pure function, do not modify self state
    def _negotiate_should_allow_prefill_pure(
        self,
        prev_state: Optional[_State],
        local_prefillable: bool,
        token_usage: float,
        **kwargs,
    ) -> _NegotiateOutput:
        # Compute local states
        local_token_watermark_force_allow = (
            local_prefillable
            and ((x := self._token_usage_low_watermark) is not None)
            and (token_usage < x)
        )

        # Gather global states
        tp0_info = self._gather_info(
            local_prefillable=local_prefillable,
            local_token_watermark_force_allow=local_token_watermark_force_allow,
            **kwargs,
        )
        global_prefillable = tp0_info[:, 0]
        global_token_watermark_force_allow = tp0_info[:, 1]
        global_running_batch = tp0_info[:, 2]
        global_max_prefill_bs = tp0_info[:, 3]
        global_waiting_queue_len = tp0_info[:, 4]

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
            input_estimation=prefillable_status,
            num_prefillable=global_prefillable.sum().item(),
            num_token_watermark_force_allow=global_token_watermark_force_allow.sum().item(),
        )

        # Compute outputs
        if prefillable_status == "all":
            if kwargs is None:
                exist_previous_wait = prev_state is not None
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="wait_success" if exist_previous_wait else "no_wait",
                    **debug_info,
                )

            # Safety valve: low KV usage means GPU is underutilized, skip
            # delay. Mirrors the check in the "mixed" branch.
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                )

            max_running_requests = kwargs.get("max_running_requests", 0)
            if not self.enable_dp_attention:
                max_running_requests = (
                    max_running_requests + self.dp_size - 1
                ) // self.dp_size

            # Aggregate across ranks; .max() is conservative (w.r.t.
            # over-delaying) and consistent with the original slot_trigger.
            global_running_batch_max = int(global_running_batch.max().item())
            global_max_prefill_bs_max = int(global_max_prefill_bs.max().item())
            global_waiting_queue_max = int(global_waiting_queue_len.max().item())

            slot_trigger = (
                max_running_requests - global_running_batch_max
                < global_max_prefill_bs_max
            )

            # Queue-based trigger: delay prefill until the waiting queue
            # reaches queue_min = min(running_req * ratio, max_prefill_bs).
            # Scales with load (no-op when running_req is small) and is capped
            # by max_prefill_bs (batching larger than one chunked prefill step
            # gives no extra benefit). Targets workloads where decode trickles
            # finish one-at-a-time and fragment prefill (e.g. large
            # random_range_ratio).
            queue_trigger = False
            if self._queue_trigger_enabled and global_running_batch_max > 0:
                queue_min_effective = min(
                    int(global_running_batch_max * self._queue_min_ratio),
                    global_max_prefill_bs_max,
                )
                queue_trigger = (
                    queue_min_effective > 0
                    and global_waiting_queue_max < queue_min_effective
                )

            # Wall-clock cap bounds worst-case TTFT for queue_trigger only;
            # slot_trigger is a hard capacity constraint (prefill would
            # immediately over-subscribe running slots) and must not be
            # overridden by the timeout.
            queue_trigger_timed_out = False
            if queue_trigger and prev_state is not None:
                elapsed_ms = (time.perf_counter() - prev_state.start_time) * 1000.0
                queue_trigger_timed_out = elapsed_ms >= self._max_delay_ms

            effective_queue_trigger = queue_trigger and not queue_trigger_timed_out
            if slot_trigger or effective_queue_trigger:
                if self.skip_first_delayer:
                    self.skip_first_delayer = False
                else:
                    next_state = prev_state or _State()
                    next_state = next_state.bump_delayed_count()
                    return _NegotiateOutput(
                        next_state=next_state,
                        output_allow=False,
                        output_reason="delay",
                        **debug_info,
                    )
            exist_previous_wait = prev_state is not None
            return _NegotiateOutput(
                next_state=None,
                output_allow=True,
                output_reason="wait_success" if exist_previous_wait else "no_wait",
                **debug_info,
            )
        elif prefillable_status == "none":
            return _NegotiateOutput(
                next_state=None,
                # It does not matter whether we allow or not, thus we allow for simplicity
                output_allow=True,
                output_reason="",
                **debug_info,
            )
        elif prefillable_status == "mixed":
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                )

            prev_delayed_count = prev_state.delayed_count if prev_state else 0
            if prev_delayed_count < self._max_delay_passes - 1:
                next_state = prev_state or _State()
                next_state = next_state.bump_delayed_count()
                return _NegotiateOutput(
                    next_state=next_state,
                    output_allow=False,
                    output_reason="delay",
                    **debug_info,
                )
            else:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="wait_timeout",
                    **debug_info,
                )
        else:
            raise NotImplementedError

    def _gather_info(
        self, local_prefillable: bool, local_token_watermark_force_allow: bool, **kwargs
    ):
        local_info = torch.tensor(
            [
                int(local_prefillable),
                int(local_token_watermark_force_allow),
                kwargs.get("running_batch", 0),
                kwargs.get("max_prefill_bs", 0),
                kwargs.get("waiting_queue_len", 0),
            ],
            device="cpu",
            dtype=torch.int64,
        )
        assert local_info.shape[0] == self._GATHER_FIELDS
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self._cpu_group,
        )
        tp0_info = self._global_info_buffer[:, 0, :]
        return tp0_info


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
            actual_execution=actual_prefill,
            output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
        )

    def negotiate_should_allow_prefill(self, local_prefillable: bool, **kwargs) -> bool:
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
                **kwargs,
            )
        return self._result.output_allow


def _record_single_pass_result(
    actual_execution: bool,
    output: _NegotiateOutput,
    metrics_collector: Optional["SchedulerMetricsCollector"],
) -> None:
    if _DEBUG_LOG:
        if output.output_allow and (output.output_reason == "wait_timeout"):
            logger.info(
                f"PrefillDelayer timeout thus not forbid prefill "
                f"(num_prefillable={output.num_prefillable}, "
                f"actual_execution={actual_execution})"
            )
        elif output.output_allow and (output.output_reason == "token_watermark"):
            logger.info(
                f"PrefillDelayer force allow prefill due to low watermark. "
                f"(num_prefillable={output.num_prefillable}, "
                f"num_token_watermark_force_allow={output.num_token_watermark_force_allow}, "
                f"actual_execution={actual_execution})"
            )
        else:
            assert output.output_reason in {
                "",
                "wait_success",
                "no_wait",
                "delay",
            }

    if metrics_collector is not None:
        if (s := output.next_state) is not None:
            wait_seconds = time.perf_counter() - s.start_time
            forward_passes = s.delayed_count
        else:
            wait_seconds = forward_passes = 0
        metrics_collector.observe_prefill_delayer_outcome(
            forward_passes=forward_passes,
            wait_seconds=wait_seconds,
            input_estimation=output.input_estimation,
            output_allow=output.output_allow,
            output_reason=output.output_reason,
            actual_execution=actual_execution,
        )
