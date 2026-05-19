from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import (
    VIOLATION_FIELDS,
)


def translate_alive_slots_for_swa(
    *,
    alive_slots: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Filter + remap a full-pool alive-slot tensor into SWA index space.

    Input alive-slot indices live in the full-pool slot space (sourced
    from ``req_to_token``). The LUT translates each into its SWA
    sub-pool slot; entries mapped to the LUT's ``-1`` sentinel
    (not in window / unmapped) are dropped because the sweep kernel
    would otherwise index the swa-sized canary buffer with garbage.
    """
    if alive_slots.numel() == 0:
        return alive_slots
    cpu_lut = lut.detach().cpu().to(torch.int64)
    lut_len = int(cpu_lut.shape[0])
    cpu_alive = alive_slots.detach().cpu().to(torch.int64)
    if lut_len > 0:
        oob_mask = cpu_alive >= lut_len
        cpu_alive = torch.where(
            oob_mask, torch.full_like(cpu_alive, lut_len - 1), cpu_alive
        )
    translated = cpu_lut[cpu_alive]
    valid = translated >= 0
    filtered = translated[valid]
    if alive_slots.device != filtered.device:
        filtered = filtered.to(alive_slots.device)
    return filtered


VIOLATION_KIND_HEAD_K: str = "head_k"
VIOLATION_KIND_HEAD_V: str = "head_v"
VIOLATION_KIND_TAIL_K: str = "tail_k"
VIOLATION_KIND_TAIL_V: str = "tail_v"
VIOLATION_KIND_SWEEP_K: str = "sweep_k"
VIOLATION_KIND_SWEEP_V: str = "sweep_v"
VIOLATION_KINDS: Tuple[str, ...] = (
    VIOLATION_KIND_HEAD_K,
    VIOLATION_KIND_HEAD_V,
    VIOLATION_KIND_TAIL_K,
    VIOLATION_KIND_TAIL_V,
    VIOLATION_KIND_SWEEP_K,
    VIOLATION_KIND_SWEEP_V,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryViolationSlot:
    """One independent violation-state set (ring + first_violation + is_errored).

    Four instances live on ``CanaryDeviceState`` — one per
    (head|tail) x (K-half|V-half) kernel launch. Keeping them disjoint stops
    the K-half launch from CAS-latching ``first_violation_set`` and
    silently masking every V-half mismatch (and vice versa), so V-half page
    table bugs become visible instead of being shadowed by an earlier K-half
    mismatch in the same forward.
    """

    violation_ring: torch.Tensor
    violation_ring_valid: torch.Tensor
    violation_write_index: torch.Tensor
    first_violation: torch.Tensor
    first_violation_set: torch.Tensor
    is_errored: torch.Tensor

    @classmethod
    def allocate(
        cls, *, device: torch.device, ring_capacity: int
    ) -> "CanaryViolationSlot":
        return cls(
            violation_ring=torch.zeros(
                ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            violation_ring_valid=torch.zeros(
                ring_capacity, dtype=torch.int32, device=device
            ),
            violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
            first_violation=torch.zeros(
                VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            first_violation_set=torch.zeros(1, dtype=torch.int32, device=device),
            is_errored=torch.zeros(1, dtype=torch.int32, device=device),
        )

    def reset(self) -> None:
        self.is_errored.zero_()
        self.first_violation_set.zero_()
        self.first_violation.zero_()
        self.violation_ring_valid.zero_()
        self.violation_write_index.zero_()


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryDeviceState:
    """GPU-resident state shared across head/tail kernel invocations.

    Violation state is split four ways — one ``CanaryViolationSlot`` per
    ``(head|tail) x (K-half|V-half)`` kernel launch — so K-half and V-half
    detection paths never share a ``first_violation`` latch or an
    ``is_errored`` flag. Counters stay split only by head/tail (K/V launches
    of the same kernel kind share write/verify work over the same slot
    indices, so per-kernel-kind counters are enough for health monitoring).
    """

    violation_slots: Dict[str, CanaryViolationSlot]
    slot_run_counter_head: torch.Tensor
    slot_run_counter_tail: torch.Tensor
    slot_run_counter_sweep: torch.Tensor
    kernel_run_counter_head: torch.Tensor
    kernel_run_counter_tail: torch.Tensor
    kernel_run_counter_sweep: torch.Tensor

    @classmethod
    def allocate(
        cls, *, device: torch.device, ring_capacity: int
    ) -> "CanaryDeviceState":
        return cls(
            violation_slots={
                kind: CanaryViolationSlot.allocate(
                    device=device, ring_capacity=ring_capacity
                )
                for kind in VIOLATION_KINDS
            },
            slot_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            slot_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
            slot_run_counter_sweep=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_head=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_tail=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter_sweep=torch.zeros(1, dtype=torch.int64, device=device),
        )

    def get_violation_slot(self, kind: str) -> CanaryViolationSlot:
        return self.violation_slots[kind]

    def reset_violation_state(self) -> None:
        """Zero out every per-kind violation slot.

        Called from the LOG-mode violation handler after each kind's
        first-violation row has been pulled to host. Without this reset the
        GPU-side ``is_errored`` flag stays at 1 forever,
        ``first_violation_set`` latches the first row permanently (new
        violations are silently masked), and ``violation_ring_valid`` fills
        up so subsequent CAS attempts all fail (= permanent ring deadlock
        after capacity rows).

        Counters (slot/kernel run counters) are intentionally NOT reset:
        the host-side health monitor uses their monotonic growth to detect
        "canary stopped running".
        """
        for slot in self.violation_slots.values():
            slot.reset()
