"""Shared construction factories for kv_canary kernel tests.

Collects input-building helpers used by ``test_<kernel>_hand.py`` / ``test_<kernel>_fuzz.py`` and
the legacy ``test_<kernel>.py`` shims. Contents are mechanical extractions from
``canary_helpers.py`` and ``test_plan.py``; downstream code must not import from here.
"""

from __future__ import annotations

import random
from typing import Literal, Optional

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


LutKind = Literal["identity", "shift", "permutation", "with_oob"]


def make_lut(
    *,
    kind: LutKind,
    pool_size: int,
    device: torch.device,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    base = torch.arange(pool_size + 1, dtype=torch.int32, device=device)
    if kind == "identity":
        return base.contiguous()
    if kind == "shift":
        return (base + 100).contiguous()
    if kind in ("permutation", "with_oob"):
        if rng is None:
            rng = random.Random(0)
        perm = list(range(pool_size + 1))
        rng.shuffle(perm)
        out = torch.tensor(perm, dtype=torch.int32, device=device)
        if kind == "with_oob":
            out[-1] = pool_size + 999
        return out.contiguous()
    raise ValueError(f"unknown LutKind: {kind}")


ReqToTokenKind = Literal["linear", "sparse_permuted", "with_holes"]


def make_req_to_token(
    *,
    kind: ReqToTokenKind,
    max_reqs: int,
    max_seq_len: int,
    device: torch.device,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    if kind == "linear":
        rp_axis = torch.arange(max_reqs, device=device, dtype=torch.int32).unsqueeze(1)
        pos_axis = torch.arange(
            max_seq_len, device=device, dtype=torch.int32
        ).unsqueeze(0)
        return (rp_axis * max_seq_len + pos_axis).contiguous()
    if rng is None:
        rng = random.Random(0)
    pool_size = max_reqs * max_seq_len
    slot_universe = list(range(1, pool_size + max_seq_len))
    rng.shuffle(slot_universe)
    rtt = torch.zeros((max_reqs, max_seq_len), dtype=torch.int32, device=device)
    cursor = 0
    for rp in range(max_reqs):
        per_req = slot_universe[cursor : cursor + max_seq_len]
        cursor += max_seq_len
        rtt[rp, :] = torch.tensor(per_req, dtype=torch.int32, device=device)
    if kind == "with_holes":
        flat = rtt.view(-1)
        n_holes = max(1, len(flat) // 10)
        hole_positions = rng.sample(range(len(flat)), k=n_holes)
        for idx in hole_positions:
            flat[idx] = -1
    return rtt.contiguous()


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


FillStrategy = Literal["constant_per_source", "random_bytes"]


def make_real_kv_sources(
    *,
    count: int,
    num_bytes_per_token: int = 8,
    page_size: int = 1,
    num_slots: int = DEFAULT_NUM_SLOTS,
    device: torch.device,
    rng: Optional[random.Random] = None,
    fill_strategy: FillStrategy = "constant_per_source",
) -> tuple[RealKvSource, ...]:
    sources: list[RealKvSource] = []
    for i in range(count):
        read_bytes_eff = num_bytes_per_token
        src = make_real_kv_source(
            num_slots=num_slots,
            num_bytes_per_token=num_bytes_per_token,
            page_size=page_size,
            read_bytes=read_bytes_eff,
            device=device,
            fill=(i + 1) * 17,
        )
        if fill_strategy == "random_bytes":
            if rng is None:
                rng = random.Random(0)
            seed = rng.randint(0, 0xFFFFFFFF)
            gen = torch.Generator(device=device).manual_seed(seed)
            src.tensor.random_(generator=gen)
        sources.append(src)
    return tuple(sources)


PaddingKind = Literal["none", "trailing", "interleaved"]


def make_padding_mask(
    *,
    bs: int,
    kind: PaddingKind,
    rng: Optional[random.Random] = None,
    padding_fraction: float = 0.25,
) -> list[bool]:
    if bs == 0:
        return []
    if kind == "none":
        return [False] * bs
    n_pad = max(1, int(bs * padding_fraction)) if bs > 0 else 0
    n_pad = min(n_pad, bs)
    if kind == "trailing":
        return [False] * (bs - n_pad) + [True] * n_pad
    if kind == "interleaved":
        if rng is None:
            rng = random.Random(0)
        mask = [False] * bs
        chosen = rng.sample(range(bs), k=n_pad)
        for idx in chosen:
            mask[idx] = True
        return mask
    raise ValueError(f"unknown PaddingKind: {kind}")


ExtrasKind = Literal["none", "few", "tile_boundary_64", "many_129"]


def make_extras(
    *,
    kind: ExtrasKind,
    capacity: int,
    device: torch.device,
    rng: Optional[random.Random] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "none":
        n_valid = 0
    elif kind == "few":
        if rng is None:
            rng = random.Random(0)
        n_valid = rng.randint(1, 6)
    elif kind == "tile_boundary_64":
        n_valid = 64
    elif kind == "many_129":
        n_valid = 129
    else:
        raise ValueError(f"unknown ExtrasKind: {kind}")
    n_valid = min(n_valid, capacity)
    slots = torch.zeros(capacity, dtype=torch.int32, device=device)
    positions = torch.zeros(capacity, dtype=torch.int32, device=device)
    prevs = torch.zeros(capacity, dtype=torch.int32, device=device)
    if n_valid > 0:
        if rng is None:
            rng = random.Random(0)
        slot_pool = rng.sample(range(500, 500 + max(1000, n_valid * 8)), k=n_valid)
        slots[:n_valid] = torch.tensor(slot_pool, dtype=torch.int32, device=device)
        pos_list = [rng.randint(0, 0xFFFF) for _ in range(n_valid)]
        positions[:n_valid] = torch.tensor(pos_list, dtype=torch.int32, device=device)
        prev_list = [-1] + slot_pool[: n_valid - 1]
        prevs[:n_valid] = torch.tensor(prev_list, dtype=torch.int32, device=device)
    num_valid = torch.tensor([n_valid], dtype=torch.int32, device=device)
    return slots, positions, prevs, num_valid


CapacityKind = Literal["loose", "tight_match", "under_by_one"]


def derive_plan_capacity(
    *,
    kind: CapacityKind,
    total_verify: int,
    extras_count: int,
    bs: int,
) -> tuple[int, int]:
    needed = total_verify + extras_count
    if kind == "loose":
        return max(needed + 64, 128), max(bs + 4, 8)
    if kind == "tight_match":
        return max(needed, 1), max(bs + 4, 8)
    if kind == "under_by_one":
        return max(needed - 1, 1), max(bs + 4, 8)
    raise ValueError(f"unknown CapacityKind: {kind}")


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


def _dummy_pseudo_tensors(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(num_tokens, dtype=torch.int32, device=_DEVICE),
        torch.zeros(num_tokens, dtype=torch.int32, device=_DEVICE),
    )
