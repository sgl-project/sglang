from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.srt.kv_canary.runner.enable_warner import _CanaryEnableWarner
from sglang.srt.kv_canary.runner.kernel_launch import (
    invoke_plan,
    launch_endpoints_per_forward,
)
from sglang.srt.kv_canary.runner.swa_divergence import SwaDivergenceReport
from sglang.srt.kv_canary.single_forward_manager.data import (
    PostOpsInsideGraphOutputBuffer,
)
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
from sglang.srt.utils.phase_checker import SimplePhaseChecker

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class _SingleForwardPhase(IntEnum):
    IDLE = 0
    AFTER_PRE_OUT = 1
    AFTER_PRE_MAYBE_IN = 2
    AFTER_POST_MAYBE_IN = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreOpsMaybeInsideGraphOutput:
    # Per-group VerifyPlan and WritePlan: FULL and SWA groups write different slot-translated
    # indices into their plans (SWA goes through full_to_swa_index_mapping; FULL doesn't),
    # and TAIL endpoints in post_ops re-read these plans. A shared plan would let SWA's
    # invoke_plan in pre_ops overwrite FULL's plan, so TAIL_K_FULL in post_ops reads SWA-translated
    # slot indices and looks them up in the FULL canary_buf — silently reading garbage / a
    # different req's old data, surfacing as TAIL_K_FULL position FPs whenever the SWA mapping
    # is non-identity (i.e. SWA pool < FULL pool).
    verify_plans: tuple[VerifyPlan, ...]
    write_plans: tuple[WritePlan, ...]
    expected_inputs: ExpectedInputs


class SingleForwardManager:
    """Owns the state of one inner ``model.forward`` invocation."""

    def __init__(
        self,
        *,
        config: CanaryConfig,
        device: torch.device,
        device_state: CanaryDeviceState,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        endpoints: tuple[CanaryEndpoint, ...],
        req_to_token_pool: "ReqToTokenPool",
        swa_window_size: int,
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        per_forward_write_entry_capacity: int,
        d2h_stream: torch.cuda.Stream,
        token_oracle_manager: Optional[TokenOracleManager],
        swa_divergence_report: Optional[SwaDivergenceReport],
        is_eagle_draft_decode: bool,
    ) -> None:
        self._config = config
        self._device = device
        self._device_state = device_state
        self._buffer_groups = buffer_groups
        self._endpoints = endpoints
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._d2h_stream = d2h_stream
        self._token_oracle_manager: Optional[TokenOracleManager] = token_oracle_manager
        self._swa_divergence_report: Optional[SwaDivergenceReport] = (
            swa_divergence_report
        )
        self._is_eagle_draft_decode: bool = is_eagle_draft_decode

        self._write_req_capacity = per_forward_write_req_capacity
        self._write_entry_capacity = per_forward_write_entry_capacity
        self._verify_capacity = per_forward_verify_capacity

        self._enable_warner = _CanaryEnableWarner(
            verify_capacity=self._verify_capacity,
            d2h_stream=d2h_stream,
        )

        self._phase_checker = SimplePhaseChecker(
            initial_phase=_SingleForwardPhase.IDLE, device=device
        )

        self._output_buffer = PostOpsInsideGraphOutputBuffer.allocate(
            num_kernel_tags=int(device_state.kernel_run_counters.shape[0]),
            num_slot_tags=int(device_state.slot_run_counters.shape[0]),
            swa_verify_total_count_shape=(
                None
                if swa_divergence_report is None
                else tuple(swa_divergence_report.verify_total_count_device.shape)
            ),
            device=device,
        )

    @property
    def phase_checker(self) -> SimplePhaseChecker:
        return self._phase_checker

    def pre_ops_outside_graph(
        self, *, maybe_inaccurate_forward_batch: "ForwardBatch"
    ) -> None:
        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.IDLE,
            next_phase=_SingleForwardPhase.AFTER_PRE_OUT,
            caller_name="SingleForwardManager.pre_ops_outside_graph",
        )

        bs = int(maybe_inaccurate_forward_batch.batch_size)
        num_tokens = int(maybe_inaccurate_forward_batch.positions.shape[0])
        if bs > self._write_req_capacity:
            raise RuntimeError(
                f"kv-canary: forward_batch.batch_size={bs} exceeds pre-allocated "
                f"write_req_capacity={self._write_req_capacity}; raise --cuda-graph-max-bs "
                f"or check CanaryLaunchCapacities.from_args"
            )
        if num_tokens > self._write_entry_capacity:
            raise RuntimeError(
                f"kv-canary: forward_batch token count={num_tokens} exceeds pre-allocated "
                f"write_entry_capacity={self._write_entry_capacity}; raise "
                f"--chunked-prefill-size / --max-prefill-tokens or check "
                f"CanaryLaunchCapacities.from_args"
            )

    def pre_ops_maybe_inside_graph(
        self, forward_batch: "ForwardBatch"
    ) -> "_PreOpsMaybeInsideGraphOutput":
        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.AFTER_PRE_OUT,
            next_phase=_SingleForwardPhase.AFTER_PRE_MAYBE_IN,
            caller_name="SingleForwardManager.pre_ops_maybe_inside_graph",
        )

        # One VerifyPlan / WritePlan per group: FULL and SWA write group-specific slot indices
        # (FULL slots vs SWA-mapped slots). A shared plan would have SWA's invoke_plan overwrite
        # FULL's, then TAIL_K_FULL in post_ops would read SWA-mapped slot indices but look them
        # up in FULL canary_buf — the source of the SWA-mode TAIL_K_FULL position FP.
        verify_plans = tuple(
            VerifyPlan.allocate(
                verify_capacity=self._verify_capacity, device=self._device
            )
            for _ in self._buffer_groups
        )
        write_plans = tuple(
            WritePlan.allocate(
                write_req_capacity=self._write_req_capacity, device=self._device
            )
            for _ in self._buffer_groups
        )
        expected_inputs = ExpectedInputs.allocate(
            capacity=self._write_entry_capacity, device=self._device
        )
        plan_input = PlanInput.allocate(
            bs_capacity=self._write_req_capacity, device=self._device
        )

        input_check_mode = self._should_enable_input_check_for_launch(forward_batch)
        if input_check_mode:
            manager = self._token_oracle_manager
            if manager is None:
                raise RuntimeError(
                    "kv-canary: input_check_mode=True requires a TokenOracleManager; pass "
                    "token_oracle_manager=install_oracle_sampler(oracle=...) into "
                    "install_canary(...)"
                )
            manager.fill_expected_inputs(
                forward_batch=forward_batch,
                expected_inputs_out=expected_inputs,
            )

        plan_input.fill_from_forward_batch(forward_batch=forward_batch)

        violation_log = self._device_state.violation_log
        num_tokens = int(forward_batch.positions.shape[0])
        expected_inputs_slice = expected_inputs.slice(num_tokens)
        for group_idx, group in enumerate(self._buffer_groups):
            verify_plan = verify_plans[group_idx]
            write_plan = write_plans[group_idx]
            invoke_plan(
                plan_input=plan_input,
                verify_plan=verify_plan,
                write_plan=write_plan,
                group=group,
                req_to_token=self._req_to_token_pool.req_to_token,
                swa_window_size=self._swa_window_size,
            )
            if self._swa_divergence_report is not None:
                self._swa_divergence_report.observe_after_invoke_plan(
                    group=group,
                    verify_plan=verify_plan,
                )
            launch_endpoints_per_forward(
                endpoints=self._endpoints,
                group=group,
                tag_filter=_is_head_tag,
                verify_plan=verify_plan,
                write_plan=write_plan,
                forward_batch=forward_batch,
                expected_inputs=expected_inputs_slice,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=input_check_mode,
            )

        return _PreOpsMaybeInsideGraphOutput(
            verify_plans=verify_plans,
            write_plans=write_plans,
            expected_inputs=expected_inputs,
        )

    def post_ops_maybe_inside_graph(
        self,
        forward_batch: "ForwardBatch",
        pre_ops_output: "_PreOpsMaybeInsideGraphOutput",
    ) -> None:
        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.AFTER_PRE_MAYBE_IN,
            next_phase=_SingleForwardPhase.AFTER_POST_MAYBE_IN,
            caller_name="SingleForwardManager.post_ops_maybe_inside_graph",
        )

        violation_log = self._device_state.violation_log
        num_tokens = int(forward_batch.positions.shape[0])
        expected_inputs_slice = pre_ops_output.expected_inputs.slice(num_tokens)
        input_check_mode = self._should_enable_input_check_for_launch(forward_batch)
        for group_idx, group in enumerate(self._buffer_groups):
            launch_endpoints_per_forward(
                endpoints=self._endpoints,
                group=group,
                tag_filter=_is_tail_tag,
                verify_plan=pre_ops_output.verify_plans[group_idx],
                write_plan=pre_ops_output.write_plans[group_idx],
                forward_batch=forward_batch,
                expected_inputs=expected_inputs_slice,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=input_check_mode,
            )

        # AND across all groups' enable flags: the warner should fire when ANY group
        # skipped verify (overflow), not just FULL group's view.
        verify_plan_enable_combined = pre_ops_output.verify_plans[0].enable
        for vp in pre_ops_output.verify_plans[1:]:
            verify_plan_enable_combined = torch.minimum(verify_plan_enable_combined, vp.enable)
        self._output_buffer.copy_from(
            verify_plan_enable=verify_plan_enable_combined,
            kernel_run_counters=self._device_state.kernel_run_counters,
            slot_run_counters=self._device_state.slot_run_counters,
            violation_write_index=self._device_state.violation_log.violation_write_index,
            swa_verify_total_count=(
                None
                if self._swa_divergence_report is None
                else self._swa_divergence_report.verify_total_count_device
            ),
        )

    def post_ops_outside_graph(self) -> None:
        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.AFTER_POST_MAYBE_IN,
            next_phase=_SingleForwardPhase.IDLE,
            caller_name="SingleForwardManager.post_ops_outside_graph",
        )

        self._enable_warner.tick(self._output_buffer.verify_plan_enable)

    def _should_enable_input_check_for_launch(
        self, forward_batch: "ForwardBatch"
    ) -> bool:
        if not self._config.input_check_mode:
            return False
        forward_mode = forward_batch.forward_mode
        if (
            self._is_eagle_draft_decode
            and forward_mode is not None
            and forward_mode.is_decode()
        ):
            return False
        return True


def _is_head_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.HEAD_K_FULL,
        CanaryLaunchTag.HEAD_V_FULL,
        CanaryLaunchTag.HEAD_K_SWA,
        CanaryLaunchTag.HEAD_V_SWA,
    )


def _is_tail_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.TAIL_K_FULL,
        CanaryLaunchTag.TAIL_V_FULL,
        CanaryLaunchTag.TAIL_K_SWA,
        CanaryLaunchTag.TAIL_V_SWA,
    )
