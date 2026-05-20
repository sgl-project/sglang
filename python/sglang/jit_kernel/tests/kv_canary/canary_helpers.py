"""Shared fixtures and helpers for the kv_canary jit_kernel diff tests.

These helpers exist purely so the three per-kernel test files
(``test_kv_canary_verify.py`` / ``test_kv_canary_write.py`` /
``test_kv_canary_plan.py``) can compose plans, canary buffers, violation
logs, and real-KV sources without copy-pasting glue. They are not part of the
public jit_kernel API surface; downstream code must not import from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.jit_kernel.kv_canary_verify import (
    CANARY_CHAIN_ANCHOR,
    CANARY_SLOT_BYTES,
    VIOLATION_FIELDS,
    RealKvSource,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary_write import WritePlan

# Default fixture sizes — small enough for fast tests, large enough that ring overflow / multi-req cases
# stay realistic without bloating the assertion surface.
DEFAULT_RING_CAPACITY: int = 64
DEFAULT_NUM_SLOTS: int = 32
DEFAULT_SLOT_STRIDE_BYTES: int = CANARY_SLOT_BYTES

_U64_MASK: int = (1 << 64) - 1
_I64_SIGN_BIT: int = 1 << 63


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeViolationLog:
    """In-memory mirror of a ViolationLog's three tensors, owned by tests.

    Allocates the violation ring, the monotonic write index, and both health counters on the requested
    device with the right dtypes / shapes; passing the same instance to verify/write step calls lets a
    test inspect the post-kernel state without dragging in srt-layer code.
    """

    ring: torch.Tensor
    write_index: torch.Tensor
    slot_run_counter: torch.Tensor
    kernel_run_counter: torch.Tensor

    @classmethod
    def allocate(
        cls, *, capacity: int = DEFAULT_RING_CAPACITY, device: torch.device
    ) -> "FakeViolationLog":
        return cls(
            ring=torch.zeros(
                capacity, VIOLATION_FIELDS, dtype=torch.int64, device=device
            ),
            write_index=torch.zeros(1, dtype=torch.int32, device=device),
            slot_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
            kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
        )


def make_canary_buf(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    slot_stride_bytes: int = DEFAULT_SLOT_STRIDE_BYTES,
    device: torch.device,
) -> torch.Tensor:
    """Allocate a fresh canary buffer of the conventional uint8 [num_slots, slot_stride_bytes] shape."""
    return torch.zeros(num_slots, slot_stride_bytes, dtype=torch.uint8, device=device)


def make_verify_plan(
    *,
    slot_indices: list[int],
    positions: list[int],
    prev_slot_indices: list[int],
    capacity: Optional[int] = None,
    device: torch.device,
) -> VerifyPlan:
    """Build a VerifyPlan whose active prefix matches the three input lists.

    Padding tail entries (when ``capacity`` exceeds the active length) are zeroed; ``verify_num_valid`` is
    set to the length of the input lists.
    """
    n_active = len(slot_indices)
    if not (len(positions) == n_active and len(prev_slot_indices) == n_active):
        raise ValueError(
            "make_verify_plan: slot_indices, positions, and prev_slot_indices must all have the same length"
        )
    cap = capacity if capacity is not None else max(n_active, 1)
    plan = VerifyPlan.allocate(verify_capacity=cap, device=device)
    if n_active > 0:
        plan.verify_slot_indices[:n_active] = torch.tensor(
            slot_indices, dtype=torch.int32, device=device
        )
        plan.verify_positions[:n_active] = torch.tensor(
            positions, dtype=torch.int32, device=device
        )
        plan.verify_prev_slot_indices[:n_active] = torch.tensor(
            prev_slot_indices, dtype=torch.int32, device=device
        )
    plan.verify_num_valid[0] = n_active
    return plan


def make_write_plan(
    *,
    write_offsets: list[int],
    seed_slot_indices: list[int],
    num_valid_reqs: int,
    req_capacity: Optional[int] = None,
    device: torch.device,
) -> WritePlan:
    """Build a WritePlan from raw offsets and seed slot lists.

    ``write_offsets`` must have length ``num_active_reqs + 1`` (the trailing total entry count).
    """
    n_active = len(seed_slot_indices)
    if len(write_offsets) != n_active + 1:
        raise ValueError(
            "make_write_plan: write_offsets must have length len(seed_slot_indices) + 1"
        )
    cap = req_capacity if req_capacity is not None else max(n_active, 1)
    plan = WritePlan.allocate(write_req_capacity=cap, device=device)
    if n_active > 0:
        plan.write_seed_slot_indices[:n_active] = torch.tensor(
            seed_slot_indices, dtype=torch.int32, device=device
        )
    plan.write_offsets[: n_active + 1] = torch.tensor(
        write_offsets, dtype=torch.int32, device=device
    )
    plan.write_num_valid_reqs[0] = num_valid_reqs
    return plan


def make_real_kv_source(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    num_bytes_per_token: int = 8,
    page_size: int = 1,
    read_bytes: Optional[int] = None,
    pad_dim1: int = 0,
    device: torch.device,
    fill: int = 0,
) -> RealKvSource:
    """Allocate one RealKvSource with the canonical [num_rows, dim1_bytes] uint8 shape.

    ``pad_dim1`` adds trailing per-row bytes the canary should skip — used by the "holey dim 1" case to
    confirm the kernel never reads past ``page_size * num_bytes_per_token``.
    """
    num_rows = (num_slots + page_size - 1) // page_size
    cols = page_size * num_bytes_per_token + pad_dim1
    tensor = torch.full(
        (num_rows, cols), fill_value=fill, dtype=torch.uint8, device=device
    )
    effective_read = read_bytes if read_bytes is not None else num_bytes_per_token
    return RealKvSource(
        tensor=tensor,
        page_size=page_size,
        num_bytes_per_token=num_bytes_per_token,
        read_bytes=effective_read,
    )


def make_real_kv_sources(
    *,
    count: int,
    num_slots: int = DEFAULT_NUM_SLOTS,
    device: torch.device,
) -> tuple[RealKvSource, ...]:
    """Build ``count`` distinct RealKvSource entries with non-trivial bytes per slot.

    Each source's tensor is initialised with a per-source-distinct constant so the fold is sensitive to
    per-source input rather than degenerating to zero.
    """
    sources: list[RealKvSource] = []
    for i in range(count):
        src = make_real_kv_source(
            num_slots=num_slots,
            num_bytes_per_token=8,
            page_size=1,
            read_bytes=8,
            device=device,
            fill=(i + 1) * 17,
        )
        sources.append(src)
    return tuple(sources)


def splitmix64(value: int) -> int:
    """Python splitmix64 finalizer used by hardcoded-expected cases (bit-equal CUDA + ref + cuh).

    Hardcoded cases manually compute multi-step chains via this helper so a ref / kernel co-regression
    cannot silently fix the diff comparison.
    """
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def splitmix64_mix4(a: int, b: int, c: int, d: int) -> int:
    """4-arg chain step matching the cuh + ref helpers."""
    return splitmix64((a ^ b ^ c ^ d) & _U64_MASK)


def to_signed_int64(value: int) -> int:
    """Reinterpret a uint64 bit pattern as signed int64 for torch.int64 storage."""
    value &= _U64_MASK
    if value >= _I64_SIGN_BIT:
        value -= 1 << 64
    return value


def chain_anchor_signed() -> int:
    """Convenience: signed-int64 reinterpretation of splitmix64(CANARY_CHAIN_ANCHOR)."""
    return to_signed_int64(splitmix64(CANARY_CHAIN_ANCHOR))


def write_slot_fields(
    *,
    canary_buf: torch.Tensor,
    slot_idx: int,
    token: int,
    position: int,
    prev_hash: int,
    real_kv_hash: int,
) -> None:
    """Stamp 4 int64 fields into canary_buf[slot_idx]; helper for "verify against canned state" cases."""
    view = canary_buf.view(torch.int64)
    view[slot_idx, 0] = token
    view[slot_idx, 1] = position
    view[slot_idx, 2] = prev_hash
    view[slot_idx, 3] = real_kv_hash


def read_slot_fields(
    *, canary_buf: torch.Tensor, slot_idx: int
) -> tuple[int, int, int, int]:
    """Return ``(token, position, prev_hash, real_kv_hash)`` from canary_buf[slot_idx]."""
    row = canary_buf.view(torch.int64)[slot_idx, :4].detach().cpu().tolist()
    return int(row[0]), int(row[1]), int(row[2]), int(row[3])


def assert_canary_state_equal(
    *, log_a: FakeViolationLog, log_b: FakeViolationLog
) -> None:
    """Byte-equal check on the three globals carried in a FakeViolationLog.

    Used to compare the CUDA-path FakeViolationLog against the ref-path FakeViolationLog in differential
    tests.
    """
    assert torch.equal(log_a.ring, log_b.ring), "violation_ring diverged (CUDA vs ref)"
    assert torch.equal(
        log_a.write_index, log_b.write_index
    ), "violation_write_index diverged (CUDA vs ref)"
    assert torch.equal(
        log_a.slot_run_counter, log_b.slot_run_counter
    ), "slot_run_counter diverged (CUDA vs ref)"
    assert torch.equal(
        log_a.kernel_run_counter, log_b.kernel_run_counter
    ), "kernel_run_counter diverged (CUDA vs ref)"


def assert_canary_buf_equal(*, buf_a: torch.Tensor, buf_b: torch.Tensor) -> None:
    """Byte-equal check on two canary buffers (uint8 [num_slots, slot_stride_bytes])."""
    assert torch.equal(buf_a, buf_b), "canary_buf diverged (CUDA vs ref)"
