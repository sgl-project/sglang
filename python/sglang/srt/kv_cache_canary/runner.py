from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_SLOT_BYTES,
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    canary_step,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.host_state import (
    BatchPlan,
    CanaryDeviceState,
    CanaryHostState,
)
from sglang.srt.kv_cache_canary.pool_patch import attach_shadow_buffers, get_shadow_buffers

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CanaryRunner:
    """Top-level orchestrator for KV cache canary.

    One instance lives on each rank. Owns the host-side request state, the
    GPU-side violation buffer + counters, the side stream that polls the
    ``is_errored`` byte, and the log/raise policy.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        pool: "MHATokenToKVPool",
        num_req_slots: int,
        device: torch.device,
    ) -> None:
        self._config = config
        self._pool = pool
        self._device = device

        attach_shadow_buffers(pool)
        self._slot_stride_bytes = pool.canary_slot_stride_bytes
        self._k_head, self._k_tail, _, _ = get_shadow_buffers(pool)

        self._host_state = CanaryHostState(config=config, num_req_slots=num_req_slots)
        self._device_state = CanaryDeviceState.allocate(
            device=device, ring_capacity=config.violation_ring_capacity
        )
        self._side_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        )
        self._is_errored_host = torch.zeros(1, dtype=torch.uint8, pin_memory=torch.cuda.is_available())

        self._forward_pass_id = 0
        self._violation_already_handled = False
        self._lock = threading.Lock()

    @property
    def config(self) -> CanaryConfig:
        return self._config

    @property
    def device_state(self) -> CanaryDeviceState:
        return self._device_state

    def commit_batch(self, plan: BatchPlan) -> None:
        self._host_state.commit_plan(plan)

    def plan_forward(
        self,
        *,
        req_pool_indices: List[int],
        req_token_counts: List[int],
        req_start_positions: List[int],
        input_tokens_per_req: List[List[int]],
    ) -> BatchPlan:
        return self._host_state.plan_batch(
            req_pool_indices=req_pool_indices,
            req_token_counts=req_token_counts,
            req_start_positions=req_start_positions,
            input_tokens_per_req=input_tokens_per_req,
        )

    def reset_request(self, req_pool_idx: int) -> None:
        self._host_state.reset_request(req_pool_idx)

    def run_head(
        self,
        *,
        plan: BatchPlan,
        slot_indices: torch.Tensor,
    ) -> None:
        self._run_kernel(plan=plan, slot_indices=slot_indices, kernel_kind=KERNEL_KIND_HEAD)

    def run_tail(
        self,
        *,
        plan: BatchPlan,
        slot_indices: torch.Tensor,
    ) -> None:
        self._run_kernel(plan=plan, slot_indices=slot_indices, kernel_kind=KERNEL_KIND_TAIL)
        self._forward_pass_id += 1

    def poll_violations(self) -> None:
        """Check is_errored on the side stream and handle log/raise policy."""
        if not self._config.enabled:
            return
        if self._side_stream is None:
            return
        with torch.cuda.stream(self._side_stream):
            self._side_stream.wait_stream(torch.cuda.current_stream(self._device))
            self._is_errored_host.copy_(self._device_state.is_errored, non_blocking=True)
        self._side_stream.synchronize()
        if int(self._is_errored_host.item()) == 0:
            return
        with self._lock:
            if self._violation_already_handled:
                return
            self._violation_already_handled = True
        self._handle_violation()

    def _run_kernel(
        self,
        *,
        plan: BatchPlan,
        slot_indices: torch.Tensor,
        kernel_kind: int,
    ) -> None:
        if not self._config.enabled:
            return
        n = slot_indices.numel()
        if n == 0:
            return
        if kernel_kind == KERNEL_KIND_HEAD:
            src_buf, dst_buf = self._k_tail, self._k_head
            slot_run_counter = self._device_state.slot_run_counter_head
            kernel_run_counter = self._device_state.kernel_run_counter_head
        else:
            src_buf, dst_buf = self._k_head, self._k_tail
            slot_run_counter = self._device_state.slot_run_counter_tail
            kernel_run_counter = self._device_state.kernel_run_counter_tail

        expected_req_ids = torch.tensor(plan.expected_req_ids, dtype=torch.int64, device=self._device)
        expected_token_ids = torch.tensor(plan.expected_token_ids, dtype=torch.int64, device=self._device)
        expected_positions = torch.tensor(plan.expected_positions, dtype=torch.int64, device=self._device)
        expected_prev_hashes = torch.tensor(plan.expected_prev_hashes, dtype=torch.int64, device=self._device)
        verify_mask = torch.tensor(plan.verify_mask, dtype=torch.int32, device=self._device)

        canary_step(
            src_buf=src_buf.view(torch.uint8).flatten(),
            dst_buf=dst_buf.view(torch.uint8).flatten(),
            slot_stride_bytes=self._slot_stride_bytes,
            slot_indices=slot_indices.to(device=self._device, dtype=torch.int64),
            expected_req_ids=expected_req_ids,
            expected_token_ids=expected_token_ids,
            expected_positions=expected_positions,
            expected_prev_hashes=expected_prev_hashes,
            verify_mask=verify_mask,
            violation_ring=self._device_state.violation_ring,
            violation_write_index=self._device_state.violation_write_index,
            first_violation=self._device_state.first_violation,
            first_violation_set=self._device_state.first_violation_set,
            is_errored=self._device_state.is_errored,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            kernel_kind=kernel_kind,
        )

    def _handle_violation(self) -> None:
        first_violation = self._device_state.first_violation.cpu().tolist()
        write_index = int(self._device_state.violation_write_index.cpu().item())
        kernel_kind, fail_reason, slot_idx, req_id, token_id, position, expected_hash, actual_hash = first_violation

        message = (
            "kv-canary violation: kernel_kind={kk} fail_reason={fr} slot_idx={sl} "
            "req_id={ri} token_id={ti} position={po} expected_hash={eh:#x} actual_hash={ah:#x} "
            "(total violations recorded: {wi})"
        ).format(
            kk=int(kernel_kind),
            fr=int(fail_reason),
            sl=int(slot_idx),
            ri=int(req_id),
            ti=int(token_id),
            po=int(position),
            eh=int(expected_hash) & ((1 << 64) - 1),
            ah=int(actual_hash) & ((1 << 64) - 1),
            wi=write_index,
        )

        if self._config.mode is CanaryMode.LOG:
            logger.error(message)
            return

        if self._config.mode is CanaryMode.RAISE:
            self._raise_with_allreduce(message)

    def _raise_with_allreduce(self, message: str) -> None:
        # Always allreduce the flag so every rank in the TP group decides the
        # same way; raising on a single rank would deadlock the next NCCL op.
        flag = self._device_state.is_errored.to(torch.int32).clone()
        try:
            from sglang.srt.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
            torch.distributed.all_reduce(flag, group=tp_group.device_group)
        except Exception:
            logger.exception("kv-canary: failed to allreduce violation flag; raising locally")
        any_rank_errored = int(flag.cpu().item()) > 0
        if any_rank_errored:
            raise RuntimeError(message)
