"""SingleForwardManager (SFM) and its per-step snapshot dataclass.

One SFM owns the per-step state of one inner ``model.forward`` invocation
inside an outer canary cycle. The outer ``CanaryManager`` holds a static
list of SFMs (length ``max(1, speculative_num_steps - 1)``) and dispatches
the monkey-patched ``model.forward`` wrap to the active SFM through a
context manager.

Lifecycle (per SFM, enforced by ``SimplePhaseChecker``):

    IDLE
      ── pre_ops_outside_graph(maybe_non_mature_forward_batch)
      → AFTER_PRE_OUT
      ── pre_ops_maybe_inside_graph(forward_batch)
      → AFTER_PRE_MAYBE_IN
      ── (original model.forward runs)
      ── post_ops_maybe_inside_graph(forward_batch)
      → AFTER_POST_MAYBE_IN
      ── post_ops_outside_graph(snapshot=self.snapshot)
      → IDLE

Phase 1 and 4 are host-side outside any cuda graph; phase 2 and 3 are
"maybe inside graph" — captured on the DECODE path, eager on EXTEND
fallback. The source code in phase 2/3 is the same regardless: every op
must be capture-safe.
"""

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
from sglang.srt.kv_canary.perturb.manager import PerturbManager
from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.srt.kv_canary.runner.enable_warner import _CanaryEnableWarner
from sglang.srt.kv_canary.runner.kernel_launch import (
    invoke_plan,
    launch_endpoints_per_forward,
)
from sglang.srt.kv_canary.runner.swa_divergence import SwaDivergenceReport
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
from sglang.srt.utils.phase_checker import SimplePhaseChecker

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class _SingleForwardPhase(IntEnum):
    """Per-SFM 4-state lifecycle used with :class:`SimplePhaseChecker`.

    Enforced order::

        IDLE
          -> AFTER_PRE_OUT       (pre_ops_outside_graph)
          -> AFTER_PRE_MAYBE_IN  (pre_ops_maybe_inside_graph)
          -> AFTER_POST_MAYBE_IN (post_ops_maybe_inside_graph)
          -> IDLE                (post_ops_outside_graph)
    """

    IDLE = 0
    AFTER_PRE_OUT = 1
    AFTER_PRE_MAYBE_IN = 2
    AFTER_POST_MAYBE_IN = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class PostOpsInsideGraphOutputSnapshot:
    """Per-SFM cloned view of the tensors produced by phases 2-3.

    The snapshot is written by phase 3 (``post_ops_maybe_inside_graph``)
    and read by phase 4 (``post_ops_outside_graph``). Phase 4 must NOT
    read ``ForwardBatch`` directly — by then the outer cycle may have
    advanced the batch to the next inner step, mutating shared fields
    (seq_lens, out_cache_loc, positions).

    All fields are device tensors holding immutable cloned snapshots of
    the per-step output. We prefer over-cloning to guarantee phase 4 sees
    a dead snapshot of "what this SFM produced", not a live view that
    later steps might overwrite.
    """

    verify_plan_enable: torch.Tensor
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    violation_write_index: torch.Tensor
    swa_verify_total_count: torch.Tensor | None
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor


class SingleForwardManager:
    """Owns the per-step state of one inner ``model.forward`` invocation.

    All static buffers (VerifyPlan, WritePlan, ExpectedInputs, PlanInput,
    snapshot tensors) are allocated once at construction. Phase 1 fills
    them outside the graph; phase 2/3 read/write them inside the captured
    region; phase 4 consumes the snapshot outside the graph.

    KV-cache / device-state / endpoints / buffer groups / perturb manager
    are shared with the owning :class:`CanaryManager`.
    """

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
        perturb_manager: PerturbManager,
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        per_forward_write_entry_capacity: int,
        d2h_stream: torch.cuda.Stream,
        token_oracle_manager: Optional[TokenOracleManager],
        swa_divergence_report: Optional[SwaDivergenceReport],
    ) -> None:
        self._config = config
        self._device = device
        self._device_state = device_state
        self._buffer_groups = buffer_groups
        self._endpoints = endpoints
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._perturb_manager = perturb_manager
        self._d2h_stream = d2h_stream
        self._token_oracle_manager: Optional[TokenOracleManager] = token_oracle_manager
        self._swa_divergence_report: Optional[SwaDivergenceReport] = (
            swa_divergence_report
        )

        self._write_req_capacity = per_forward_write_req_capacity
        self._write_entry_capacity = per_forward_write_entry_capacity
        self._verify_capacity = per_forward_verify_capacity

        self._verify_plan = VerifyPlan.allocate(
            verify_capacity=per_forward_verify_capacity, device=device
        )
        self._write_plan = WritePlan.allocate(
            write_req_capacity=per_forward_write_req_capacity, device=device
        )
        self._expected_inputs = ExpectedInputs.allocate(
            capacity=per_forward_write_entry_capacity, device=device
        )
        self._plan_input = PlanInput.allocate(
            bs_capacity=per_forward_write_req_capacity, device=device
        )

        self._enable_warner = _CanaryEnableWarner(
            verify_capacity=self._verify_capacity,
            d2h_stream=d2h_stream,
        )

        self._phase_checker = SimplePhaseChecker(
            initial_phase=_SingleForwardPhase.IDLE, device=device
        )

        # Pre-allocated snapshot buffers populated by ``post_ops_maybe_inside_graph``.
        # We over-clone (per the design: "宁可多 clone 也要保证 immutable") so phase 4
        # never reads a live ForwardBatch / device_state tensor.
        self._snapshot_buffers = _allocate_snapshot_buffers(
            verify_capacity=per_forward_verify_capacity,
            write_entry_capacity=per_forward_write_entry_capacity,
            write_req_capacity=per_forward_write_req_capacity,
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
    def snapshot(self) -> PostOpsInsideGraphOutputSnapshot:
        return self._snapshot_buffers.as_snapshot()

    @property
    def phase_checker(self) -> SimplePhaseChecker:
        return self._phase_checker

    def pre_ops_outside_graph(
        self, *, maybe_non_mature_forward_batch: "ForwardBatch"
    ) -> None:
        """Phase 1. Host-side outside any cuda graph.

        The input ``maybe_non_mature_forward_batch`` may be the OUTER batch
        for an EAGLE draft step — its step-specific fields (seq_lens after
        increment, out_cache_loc slice for step i, ...) are NOT yet mature.
        Callers in this phase must only consume batch-level fields
        (``req_pool_indices``, batch_size, base out_cache_loc).
        """
        if self._config.mode == "off":
            return

        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.IDLE,
            next_phase=_SingleForwardPhase.AFTER_PRE_OUT,
            caller_name="SingleForwardManager.pre_ops_outside_graph",
        )

        bs = int(maybe_non_mature_forward_batch.batch_size)
        num_tokens = int(maybe_non_mature_forward_batch.positions.shape[0])
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

        self._perturb_manager.perturb(
            maybe_non_mature_forward_batch=maybe_non_mature_forward_batch
        )

    def pre_ops_maybe_inside_graph(self, forward_batch: "ForwardBatch") -> None:
        """Phase 2. Capture-safe ops only (DECODE path is inside cuda graph;
        EXTEND / eager fallback is outside). Fired by monkey-patched
        ``model.forward`` wrap BEFORE the original forward."""
        if self._config.mode == "off":
            return

        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.AFTER_PRE_OUT,
            next_phase=_SingleForwardPhase.AFTER_PRE_MAYBE_IN,
            caller_name="SingleForwardManager.pre_ops_maybe_inside_graph",
        )

        if self._config.input_check_mode:
            manager = self._token_oracle_manager
            if manager is None:
                raise RuntimeError(
                    "kv-canary: input_check_mode=True requires a TokenOracleManager; pass "
                    "token_oracle_manager=install_oracle_sampler(oracle=...) into "
                    "install_canary(...)"
                )
            manager.fill_expected_inputs(
                forward_batch=forward_batch,
                expected_inputs_out=self._expected_inputs,
            )

        self._plan_input.fill_from_forward_batch(forward_batch=forward_batch)

        violation_log = self._device_state.violation_log
        num_tokens = int(forward_batch.positions.shape[0])
        expected_inputs_slice = self._expected_inputs.slice(num_tokens)
        input_check_mode = self._config.input_check_mode
        for group in self._buffer_groups:
            invoke_plan(
                plan_input=self._plan_input,
                verify_plan=self._verify_plan,
                write_plan=self._write_plan,
                group=group,
                req_to_token=self._req_to_token_pool.req_to_token,
                swa_window_size=self._swa_window_size,
            )
            if self._swa_divergence_report is not None:
                self._swa_divergence_report.observe_after_invoke_plan(
                    group=group,
                    verify_plan=self._verify_plan,
                )
            launch_endpoints_per_forward(
                endpoints=self._endpoints,
                group=group,
                tag_filter=_is_head_tag,
                verify_plan=self._verify_plan,
                write_plan=self._write_plan,
                forward_batch=forward_batch,
                expected_inputs=expected_inputs_slice,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=input_check_mode,
            )

    def post_ops_maybe_inside_graph(self, forward_batch: "ForwardBatch") -> None:
        """Phase 3. Same capture regime as phase 2. Fired by monkey-patched
        ``model.forward`` wrap AFTER the original forward.

        Launches TAIL kernels reusing the plan staged in phase 2, then copies
        every observable into the per-SFM snapshot so phase 4 sees a dead
        view immune to later step mutation.
        """
        if self._config.mode == "off":
            return

        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.AFTER_PRE_MAYBE_IN,
            next_phase=_SingleForwardPhase.AFTER_POST_MAYBE_IN,
            caller_name="SingleForwardManager.post_ops_maybe_inside_graph",
        )

        violation_log = self._device_state.violation_log
        num_tokens = int(forward_batch.positions.shape[0])
        expected_inputs_slice = self._expected_inputs.slice(num_tokens)
        input_check_mode = self._config.input_check_mode
        for group in self._buffer_groups:
            launch_endpoints_per_forward(
                endpoints=self._endpoints,
                group=group,
                tag_filter=_is_tail_tag,
                verify_plan=self._verify_plan,
                write_plan=self._write_plan,
                forward_batch=forward_batch,
                expected_inputs=expected_inputs_slice,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=input_check_mode,
            )

        self._snapshot_buffers.copy_from(
            verify_plan=self._verify_plan,
            device_state=self._device_state,
            forward_batch=forward_batch,
            swa_divergence_report=self._swa_divergence_report,
        )

    def post_ops_outside_graph(
        self, *, snapshot: PostOpsInsideGraphOutputSnapshot
    ) -> None:
        """Phase 4. Host-side outside cuda graph. Reads ONLY ``snapshot``,
        NEVER the live ForwardBatch."""
        if self._config.mode == "off":
            return

        self._phase_checker.update(
            expect_phase=_SingleForwardPhase.AFTER_POST_MAYBE_IN,
            next_phase=_SingleForwardPhase.IDLE,
            caller_name="SingleForwardManager.post_ops_outside_graph",
        )

        self._perturb_manager.consume_snapshot(snapshot=snapshot)
        self._enable_warner.tick(snapshot.verify_plan_enable)


@dataclass(slots=True, kw_only=True)
class _SnapshotBuffers:
    """Mutable storage for the per-SFM snapshot. ``copy_from`` is called
    from phase 3 (inside cuda graph capture on DECODE), so every write
    must be an in-place ``copy_`` into pre-allocated tensors — no
    allocation, no shape change.
    """

    verify_plan_enable: torch.Tensor
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    violation_write_index: torch.Tensor
    swa_verify_total_count: torch.Tensor | None
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor

    def as_snapshot(self) -> PostOpsInsideGraphOutputSnapshot:
        return PostOpsInsideGraphOutputSnapshot(
            verify_plan_enable=self.verify_plan_enable,
            kernel_run_counters=self.kernel_run_counters,
            slot_run_counters=self.slot_run_counters,
            violation_write_index=self.violation_write_index,
            swa_verify_total_count=self.swa_verify_total_count,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            out_cache_loc=self.out_cache_loc,
            positions=self.positions,
        )

    def copy_from(
        self,
        *,
        verify_plan: VerifyPlan,
        device_state: CanaryDeviceState,
        forward_batch: "ForwardBatch",
        swa_divergence_report: Optional[SwaDivergenceReport],
    ) -> None:
        self.verify_plan_enable.copy_(verify_plan.enable)
        self.kernel_run_counters.copy_(device_state.kernel_run_counters)
        self.slot_run_counters.copy_(device_state.slot_run_counters)
        self.violation_write_index.copy_(device_state.violation_log.violation_write_index)
        if (
            self.swa_verify_total_count is not None
            and swa_divergence_report is not None
        ):
            self.swa_verify_total_count.copy_(
                swa_divergence_report.verify_total_count_device
            )

        # ForwardBatch slices: bs and num_tokens are capture-time constants
        # for any given graph, so the static-sized clones below are valid.
        bs = int(forward_batch.req_pool_indices.shape[0])
        self.req_pool_indices[:bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:bs].copy_(forward_batch.seq_lens)
        num_tokens = int(forward_batch.positions.shape[0])
        self.positions[:num_tokens].copy_(forward_batch.positions)
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)


def _allocate_snapshot_buffers(
    *,
    verify_capacity: int,
    write_entry_capacity: int,
    write_req_capacity: int,
    num_kernel_tags: int,
    num_slot_tags: int,
    swa_verify_total_count_shape: tuple[int, ...] | None,
    device: torch.device,
) -> _SnapshotBuffers:
    return _SnapshotBuffers(
        verify_plan_enable=torch.zeros(1, dtype=torch.int32, device=device),
        kernel_run_counters=torch.zeros(
            num_kernel_tags, dtype=torch.int64, device=device
        ),
        slot_run_counters=torch.zeros(
            num_slot_tags, dtype=torch.int64, device=device
        ),
        violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
        swa_verify_total_count=(
            None
            if swa_verify_total_count_shape is None
            else torch.zeros(
                swa_verify_total_count_shape, dtype=torch.int32, device=device
            )
        ),
        req_pool_indices=torch.zeros(
            write_req_capacity, dtype=torch.int64, device=device
        ),
        seq_lens=torch.zeros(write_req_capacity, dtype=torch.int64, device=device),
        out_cache_loc=torch.zeros(
            write_entry_capacity, dtype=torch.int64, device=device
        ),
        positions=torch.zeros(
            write_entry_capacity, dtype=torch.int64, device=device
        ),
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
