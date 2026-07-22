from __future__ import annotations

import random
from typing import Literal, Optional

import torch

from sglang.jit_kernel.tests.kv_canary._constants import DEFAULT_NUM_SLOTS
from sglang.kernels.ops.kv_canary.verify import (
    RealKvSource,
    VerifyPlan,
)
from sglang.kernels.ops.kv_canary.write import WritePlan

_DEVICE = torch.device("cuda")


LutKind = Literal["identity", "shift", "permutation", "with_oob"]


def make_lut(
    *,
    kind: LutKind,
    pool_size: int,
    device: torch.device,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    base = torch.arange(pool_size + 1, dtype=torch.int64, device=device)
    if kind == "identity":
        return base.contiguous()
    if kind == "shift":
        return (base + 100).contiguous()
    if kind in ("permutation", "with_oob"):
        if rng is None:
            rng = random.Random(0)
        perm = list(range(pool_size + 1))
        rng.shuffle(perm)
        out = torch.tensor(perm, dtype=torch.int64, device=device)
        if kind == "with_oob":
            out[-1] = pool_size + 999
        return out.contiguous()
    raise ValueError(f"unknown LutKind: {kind}")


ReqToTokenKind = Literal["linear", "sparse_permuted"]


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
    # Slots index into a full_to_swa LUT sized [pool_size + 1], so values must stay
    # in [0, pool_size]. The universe spans [1, pool_size] (skipping 0 as reserved),
    # giving exactly max_reqs * max_seq_len unique slots — one per (rp, pos) cell.
    slot_universe = list(range(1, pool_size + 1))
    rng.shuffle(slot_universe)
    rtt = torch.zeros((max_reqs, max_seq_len), dtype=torch.int32, device=device)
    cursor = 0
    for rp in range(max_reqs):
        per_req = slot_universe[cursor : cursor + max_seq_len]
        cursor += max_seq_len
        rtt[rp, :] = torch.tensor(per_req, dtype=torch.int32, device=device)
    return rtt.contiguous()


def make_real_kv_source(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    num_bytes_per_token: int = 16,
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
    num_bytes_per_token: int = 16,
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


def clone_real_kv_sources(
    sources: tuple[RealKvSource, ...],
) -> tuple[RealKvSource, ...]:
    return tuple(
        RealKvSource(
            tensor=src.tensor.clone(),
            page_size=src.page_size,
            num_bytes_per_token=src.num_bytes_per_token,
            read_bytes=src.read_bytes,
        )
        for src in sources
    )


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


def allocate_plan_pair(
    *,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan, VerifyPlan, WritePlan]:
    return (
        VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE),
        WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE),
        VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE),
        WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE),
    )


def empty_extras() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(1, dtype=torch.int64, device=_DEVICE),
        torch.zeros(1, dtype=torch.int64, device=_DEVICE),
        torch.zeros(1, dtype=torch.int64, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
    )


def dummy_pseudo_tensors(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(num_tokens, dtype=torch.int64, device=_DEVICE),
        torch.zeros(num_tokens, dtype=torch.int64, device=_DEVICE),
    )
