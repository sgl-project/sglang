"""Shared construction factories for kv_canary kernel tests.

Collects input-building helpers used by ``test_<kernel>_hand.py`` / ``test_<kernel>_fuzz.py`` and
the legacy ``test_<kernel>.py`` shims. Contents are mechanical extractions from
``canary_helpers.py`` and ``test_plan.py``; downstream code must not import from here.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import (
    RealKvSource,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.write import WritePlan

# Default fixture sizes — small enough for fast tests, large enough that ring overflow / multi-req cases
# stay realistic without bloating the assertion surface.
DEFAULT_NUM_SLOTS: int = 32

_U64_MASK: int = (1 << 64) - 1

_DEVICE = torch.device("cuda")


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


def _allocate_plan_pair(
    *,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan, VerifyPlan, WritePlan]:
    """Allocate (triton_verify, triton_write, ref_verify, ref_write) plan tensors."""
    return (
        VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE),
        WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE),
        VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE),
        WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE),
    )


def _build_req_to_token(*, max_reqs: int, max_seq_len: int) -> torch.Tensor:
    """Construct a deterministic [max_reqs, max_seq_len] req_to_token table.

    Slot index = rp * max_seq_len + pos so every (rp, pos) maps to a distinct slot, which lets per-entry
    assertions reason about which req contributed which slot.
    """
    rp_axis = torch.arange(max_reqs, device=_DEVICE, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    return (rp_axis * max_seq_len + pos_axis).contiguous()


def _empty_extras() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four zero-filled length-1 int32 tensors representing an "extras absent" payload."""
    return (
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
    )


def _make_extras(
    *,
    slot_indices: list[int],
    positions: list[int],
    prev_slot_indices: list[int],
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(slot_indices)
    slots = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    pos = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    prevs = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    if n > 0:
        slots[:n] = torch.tensor(slot_indices, dtype=torch.int32, device=_DEVICE)
        pos[:n] = torch.tensor(positions, dtype=torch.int32, device=_DEVICE)
        prevs[:n] = torch.tensor(prev_slot_indices, dtype=torch.int32, device=_DEVICE)
    num_valid = torch.tensor([n], dtype=torch.int32, device=_DEVICE)
    return slots, pos, prevs, num_valid
