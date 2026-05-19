from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import canary_step
from sglang.jit_kernel.kv_cache_canary_plan_ref import BatchPlanGpu
from sglang.srt.kv_cache_canary.host_state import (
    CanaryViolationSlot,
)


def _empty_real_kv_buf(device: torch.device) -> torch.Tensor:
    """2D 1x1 placeholder ``real_kv_buf`` for OFF-mode kernel launches.

    The kernel takes a non-null tensor handle but never dereferences it
    in OFF mode (``real_kv_hash_mode == 0`` -> early-out). A 2D uint8
    placeholder satisfies the type / shape constraint without allocating
    real KV memory.
    """
    return torch.zeros(1, 1, dtype=torch.uint8, device=device)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryEndpoint:
    """One end (head OR tail) of the head/tail canary pair.

    Owns the canary tensors that this endpoint both writes into and verifies
    against, the violation buckets the launches feed into, and the per-endpoint
    counters. Each endpoint is self-verifying — the splitmix64 chain over the
    endpoint's own buffer is enough to detect any corruption of the slots this
    endpoint has touched, so there is no cross-shadow read.

    One pair (``head_endpoint``, ``tail_endpoint``) lives on each
    ``CanaryRunner`` instance. The dataclass is frozen so the runner cannot
    silently swap canary buffer handles between forwards; per-launch mutable state
    (violation buffer + counters) is held by the GPU tensor handles.
    """

    kernel_kind: int
    k_canary_buf: torch.Tensor
    v_canary_buf: Optional[torch.Tensor]
    k_violation: CanaryViolationSlot
    v_violation: Optional[CanaryViolationSlot]
    slot_run_counter: torch.Tensor
    kernel_run_counter: torch.Tensor
    real_kv_buf: torch.Tensor
    real_kv_read_bytes: int
    real_kv_hash_mode: int

    @property
    def has_v_half(self) -> bool:
        return self.v_canary_buf is not None

    def launch(
        self,
        *,
        plan: BatchPlanGpu,
        seed: int,
    ) -> None:
        """Launch the head|tail kernel pair against this endpoint's own buffers.

        Iterates over (K-half) and (V-half, if present); each half launches
        an independent ``canary_step`` against its own
        ``CanaryViolationSlot`` so K-half and V-half mismatches never
        cross-latch into one another's first-violation row.
        """
        buf_specs: List[Tuple[torch.Tensor, CanaryViolationSlot]] = [
            (self.k_canary_buf, self.k_violation),
        ]
        if (
            self.has_v_half
            and self.v_violation is not None
            and self.v_canary_buf is not None
        ):
            buf_specs.append((self.v_canary_buf, self.v_violation))

        for buf, violation_slot in buf_specs:
            # API source of truth: docstring of canary_step in sglang.jit_kernel.kv_cache_canary
            canary_step(
                buf=buf,
                plan=plan,
                seed=int(seed),
                violation_ring=violation_slot.violation_ring,
                violation_ring_valid=violation_slot.violation_ring_valid,
                violation_write_index=violation_slot.violation_write_index,
                first_violation=violation_slot.first_violation,
                first_violation_set=violation_slot.first_violation_set,
                is_errored=violation_slot.is_errored,
                slot_run_counter=self.slot_run_counter,
                kernel_run_counter=self.kernel_run_counter,
                kernel_kind=self.kernel_kind,
                real_kv_buf=self.real_kv_buf,
                real_kv_read_bytes=self.real_kv_read_bytes,
                real_kv_hash_mode=self.real_kv_hash_mode,
            )
