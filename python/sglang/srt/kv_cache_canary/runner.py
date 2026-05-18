from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_cache_canary import (
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
from sglang.srt.kv_cache_canary.pool_patch import (
    attach_shadow_buffers,
    get_shadow_buffers,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

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
        self._device = device

        attach_shadow_buffers(pool)
        self._slot_stride_bytes = pool.canary_slot_stride_bytes
        self._k_head, self._k_tail, _, _ = get_shadow_buffers(pool)

        self.host_state = CanaryHostState(config=config, num_req_slots=num_req_slots)
        self._device_state = CanaryDeviceState.allocate(
            device=device, ring_capacity=config.violation_ring_capacity
        )
        self._side_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        )
        self._is_errored_host = torch.zeros(
            1, dtype=torch.uint8, pin_memory=torch.cuda.is_available()
        )
        self._violation_handled = False

    @property
    def config(self) -> CanaryConfig:
        return self._config

    def run_head(self, *, plan: BatchPlan, slot_indices: torch.Tensor) -> None:
        self._run_kernel(
            plan=plan, slot_indices=slot_indices, kernel_kind=KERNEL_KIND_HEAD
        )

    def run_tail(self, *, plan: BatchPlan, slot_indices: torch.Tensor) -> None:
        self._run_kernel(
            plan=plan, slot_indices=slot_indices, kernel_kind=KERNEL_KIND_TAIL
        )

    def poll_violations(self) -> None:
        if not self._config.enabled or self._side_stream is None:
            return
        if self._violation_handled:
            return

        with torch.cuda.stream(self._side_stream):
            self._side_stream.wait_stream(torch.cuda.current_stream(self._device))
            self._is_errored_host.copy_(
                self._device_state.is_errored, non_blocking=True
            )
        self._side_stream.synchronize()
        if int(self._is_errored_host.item()) == 0:
            return

        self._violation_handled = True
        self._handle_violation()

    def _run_kernel(
        self, *, plan: BatchPlan, slot_indices: torch.Tensor, kernel_kind: int
    ) -> None:
        if not self._config.enabled or slot_indices.numel() == 0:
            return
        if kernel_kind == KERNEL_KIND_HEAD:
            src_buf, dst_buf = self._k_tail, self._k_head
            slot_run_counter = self._device_state.slot_run_counter_head
            kernel_run_counter = self._device_state.kernel_run_counter_head
        else:
            src_buf, dst_buf = self._k_head, self._k_tail
            slot_run_counter = self._device_state.slot_run_counter_tail
            kernel_run_counter = self._device_state.kernel_run_counter_tail

        canary_step(
            src_buf=src_buf.view(torch.uint8).flatten(),
            dst_buf=dst_buf.view(torch.uint8).flatten(),
            slot_stride_bytes=self._slot_stride_bytes,
            slot_indices=slot_indices.to(device=self._device, dtype=torch.int64),
            expected_req_ids=_to_int64(plan.expected_req_ids, self._device),
            expected_token_ids=_to_int64(plan.expected_token_ids, self._device),
            expected_positions=_to_int64(plan.expected_positions, self._device),
            expected_prev_hashes=_to_int64(plan.expected_prev_hashes, self._device),
            verify_mask=_to_int32(plan.verify_mask, self._device),
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
        (
            kernel_kind,
            fail_reason,
            slot_idx,
            req_id,
            token_id,
            position,
            expected_hash,
            actual_hash,
        ) = first_violation

        u64_mask = (1 << 64) - 1
        message = (
            f"kv-canary violation: kernel_kind={int(kernel_kind)} "
            f"fail_reason={int(fail_reason)} slot_idx={int(slot_idx)} "
            f"req_id={int(req_id)} token_id={int(token_id)} position={int(position)} "
            f"expected_hash={int(expected_hash) & u64_mask:#x} "
            f"actual_hash={int(actual_hash) & u64_mask:#x} "
            f"(total violations recorded: {write_index})"
        )

        if self._config.mode is CanaryMode.LOG:
            logger.error(message)
            return
        if self._config.mode is CanaryMode.RAISE:
            self._raise_with_allreduce(message)

    def _raise_with_allreduce(self, message: str) -> None:
        # Raising on a single rank would leave peer ranks stuck in the next
        # NCCL collective; allreduce the flag so every rank exits together.
        from sglang.srt.distributed.parallel_state import get_tp_group

        flag = self._device_state.is_errored.to(torch.int32).clone()
        try:
            tp_group = get_tp_group()
            torch.distributed.all_reduce(flag, group=tp_group.device_group)
        except Exception:
            logger.exception(
                "kv-canary: failed to allreduce violation flag; raising locally"
            )
        if int(flag.cpu().item()) > 0:
            raise RuntimeError(message)


def _to_int64(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def _to_int32(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)
