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


def _torch_reduce_minimum(tensors: list[torch.Tensor]) -> torch.Tensor:
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.minimum(out, t)
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreOpsMaybeInsideGraphOutput:
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

        self._populate_expected_token_pool(forward_batch=maybe_inaccurate_forward_batch)

    def _populate_expected_token_pool(self, *, forward_batch: "ForwardBatch") -> None:
        """H2D-copy the per-req source-of-truth token snapshot into the device-side pool.

        Only runs when the real-model token-id validator is enabled. The pool/valid_lens
        tensors are device-resident and shared across forward steps; the inside-graph plan
        kernel gathers from them later. Host writes happen outside the cuda graph so
        non_blocking H2D copies are safe.

        ``forward_batch.req_truth_seqs is None`` happens during cuda-graph capture's
        synchronous-batch dry run. Bail out then; ``valid_lens`` stays at its zero init for
        the affected rows, the gather kernel produces all -1 sentinels, and the WRITE
        kernel skips the token check accordingly.
        """
        if not self._config.enable_req_token_ids_check:
            return
        req_truth_seqs = forward_batch.req_truth_seqs
        if req_truth_seqs is None:
            return
        pool = self._device_state.req_to_expected_token_ids_pool
        valid_lens = self._device_state.req_to_expected_token_ids_valid_lens
        if pool is None or valid_lens is None:
            return

        max_context_len = int(pool.shape[1])
        req_pool_indices_host = forward_batch.req_pool_indices.tolist()
        seq_lens_host: list[int] = []
        for req_pool_idx, seq in zip(req_pool_indices_host, req_truth_seqs):
            seq_len = len(seq)
            if seq_len > max_context_len:
                raise RuntimeError(
                    f"kv-canary: req sequence length {seq_len} exceeds canary pool "
                    f"max_context_len {max_context_len}; raise --context-length or "
                    f"audit ReqToTokenPool sizing"
                )
            if seq_len > 0:
                seq_tensor = torch.tensor(seq, dtype=torch.int32, pin_memory=True)
                pool[req_pool_idx, :seq_len].copy_(seq_tensor, non_blocking=True)
            seq_lens_host.append(seq_len)

        # Batch the valid_lens update: build a pinned int32 vector and index_copy_ in one shot
        # instead of N separate scalar D2H/H2D copies.
        index_t = torch.tensor(
            req_pool_indices_host, dtype=torch.int64, pin_memory=True
        )
        vlen_t = torch.tensor(seq_lens_host, dtype=torch.int32, pin_memory=True)
        valid_lens.index_copy_(
            0,
            index_t.to(valid_lens.device, non_blocking=True),
            vlen_t.to(valid_lens.device, non_blocking=True),
        )

    def pre_ops_maybe_inside_graph(
        self, forward_batch: "ForwardBatch"
    ) -> "_PreOpsMaybeInsideGraphOutput":
        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.AFTER_PRE_OUT,
            next_phase=_SingleForwardPhase.AFTER_PRE_MAYBE_IN,
            caller_name="SingleForwardManager.pre_ops_maybe_inside_graph",
        )

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
            self._fill_expected_inputs_host_side(
                forward_batch=forward_batch,
                out_expected_inputs=expected_inputs,
            )

        plan_input.fill_from_forward_batch(forward_batch=forward_batch)

        violation_log = self._device_state.violation_log
        num_tokens = int(forward_batch.positions.shape[0])
        expected_inputs_slice = expected_inputs.slice(num_tokens)

        # Real-model validator: gather the source-of-truth tokens into expected_inputs.tokens via
        # the plan-side kernel (cuda-graph safe). Positions are copied 1:1 from forward_batch.
        # When the validator is off the gather kernel is not invoked; tokens stay at their -1
        # sentinel allocate default which the WRITE kernel treats as "skip token check".
        use_kernel_side_token_gather = (
            input_check_mode and self._config.enable_req_token_ids_check
        )
        if use_kernel_side_token_gather:
            expected_inputs.positions[:num_tokens].copy_(
                forward_batch.positions.to(torch.int64)
            )

        for group_idx, group in enumerate(self._buffer_groups):
            verify_plan = verify_plans[group_idx]
            write_plan = write_plans[group_idx]
            # Plan kernel runs the expected-token gather once on group 0; subsequent groups would
            # produce an identical buffer (write_offsets are SWA-independent) so we skip the call.
            do_gather = use_kernel_side_token_gather and group_idx == 0
            invoke_plan(
                plan_input=plan_input,
                verify_plan=verify_plan,
                write_plan=write_plan,
                group=group,
                req_to_token=self._req_to_token_pool.req_to_token,
                swa_window_size=self._swa_window_size,
                expected_token_pool=(
                    self._device_state.req_to_expected_token_ids_pool
                    if do_gather
                    else None
                ),
                expected_token_valid_lens=(
                    self._device_state.req_to_expected_token_ids_valid_lens
                    if do_gather
                    else None
                ),
                out_expected_input_tokens=(
                    expected_inputs.tokens if do_gather else None
                ),
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

        verify_plan_enable_combined = _torch_reduce_minimum(
            [x.enable for x in pre_ops_output.verify_plans]
        )
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
        if self._config.enable_req_token_ids_check:
            return True
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

    def _fill_expected_inputs_host_side(
        self,
        *,
        forward_batch: "ForwardBatch",
        out_expected_inputs: ExpectedInputs,
    ) -> None:
        """Host-side fill for the mock-model (TokenOracleManager) path only.

        The real-model validator runs its expected-token gather inside the cuda graph via the
        plan-side kernel; positions are also copied inside-graph by the caller. So when
        ``enable_req_token_ids_check`` is on this method is a no-op.
        """
        if self._config.enable_req_token_ids_check:
            return

        manager = self._token_oracle_manager
        if manager is None:
            raise RuntimeError(
                "kv-canary: input_check_mode=True requires a TokenOracleManager; pass "
                "token_oracle_manager=install_oracle_sampler(oracle=...) into "
                "install_canary(...)"
            )
        manager.fill_expected_inputs(
            forward_batch=forward_batch,
            out_expected_inputs=out_expected_inputs,
        )


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
