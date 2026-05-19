from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import canary_step
from sglang.srt.kv_cache_canary.host_state import (
    CanaryLaunchBuffers,
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

    Owns the destination canary tensors that this endpoint writes into, the
    violation buckets the launches feed into, and the per-endpoint counters.
    The peer endpoint supplies the source buffers via :meth:`launch` — head
    reads tail's canary buffers, tail reads head's canary buffers.

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
        src: "CanaryEndpoint",
        launch_buffers: CanaryLaunchBuffers,
        seed: int,
    ) -> None:
        """Launch the head|tail kernel pair reading from ``src``.

        Iterates over (K-half) and (V-half, if present); each half launches
        an independent ``canary_step`` against its own
        ``CanaryViolationSlot`` so K-half and V-half mismatches never
        cross-latch into one another's first-violation row.
        """
        buf_specs: List[Tuple[torch.Tensor, torch.Tensor, CanaryViolationSlot]] = [
            (
                src.k_canary_buf,
                self.k_canary_buf,
                self.k_violation,
            )
        ]
        if (
            self.has_v_half
            and src.has_v_half
            and self.v_violation is not None
            and self.v_canary_buf is not None
            and src.v_canary_buf is not None
        ):
            buf_specs.append(
                (
                    src.v_canary_buf,
                    self.v_canary_buf,
                    self.v_violation,
                )
            )

        for src_buf, dst_buf, violation_slot in buf_specs:
            # API source of truth: docstring of canary_step in sglang.jit_kernel.kv_cache_canary
            canary_step(
                src_buf=src_buf,
                dst_buf=dst_buf,
                verify_slot_indices=launch_buffers.verify_slot_indices,
                verify_positions=launch_buffers.verify_positions,
                verify_prev_slot_indices=launch_buffers.verify_prev_slot_indices,
                verify_num_valid=launch_buffers.verify_num_valid,
                write_slot_indices=launch_buffers.write_slot_indices,
                write_token_ids=launch_buffers.write_token_ids,
                write_positions=launch_buffers.write_positions,
                write_req_seed_slot_indices=launch_buffers.write_req_seed_slot_indices,
                write_req_entry_starts=launch_buffers.write_req_entry_starts,
                write_req_entry_counts=launch_buffers.write_req_entry_counts,
                write_req_num_valid=launch_buffers.write_req_num_valid,
                expected_write_token_ids=launch_buffers.expected_write_token_ids,
                expected_write_positions=launch_buffers.expected_write_positions,
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
