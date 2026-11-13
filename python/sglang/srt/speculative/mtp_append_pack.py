"""Exact batched copies for the accepted native-MTP continuation rows."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class PackedAppend:
    tokens: torch.Tensor
    hidden: torch.Tensor
    cache_locations: torch.Tensor
    native_locations: torch.Tensor
    positions: torch.Tensor
    metadata: torch.Tensor
    last_rows: torch.Tensor
    anchor_locations: torch.Tensor
    pending_locations: torch.Tensor


@triton.jit(do_not_specialize=["B"])
def _pack(
    candidates,
    bonus,
    hidden,
    locations,
    reqs,
    meta,
    out_tokens,
    out_hidden,
    out_locations,
    out_native,
    positions,
    last_rows,
    anchors,
    pending,
    B,
    H: tl.constexpr,
    PENDING: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    BONUS_STRIDE: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, tile = tl.program_id(0), tl.program_id(1)
    prefix = tl.load(meta + b)
    n = tl.load(meta + B + b)
    start = tl.load(meta + 2 * B + b)
    j = tl.arange(0, 8)
    d = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        hidden + b * H0 + j[:, None] * H1 + d[None, :] * H2,
        mask=(j[:, None] < n) & (d[None, :] < H),
        other=0,
    )
    tl.store(
        out_hidden + (start + j[:, None]) * H + d[None, :],
        values,
        mask=(j[:, None] < n) & (d[None, :] < H),
    )
    if tile == 0:
        req = tl.load(reqs + b * REQ_STRIDE)
        private = req + PENDING
        token = tl.load(candidates + b * C0 + (j + 1) * C1, mask=j < n - 1, other=0)
        correction = tl.load(bonus + b * BONUS_STRIDE)
        token = tl.where(j == n - 1, correction, token)
        old = tl.load(locations + b * L0 + j * L1, mask=j < n, other=0)
        following = tl.load(locations + b * L0 + (j + 1) * L1, mask=j < n - 1, other=0)
        native = tl.where(j == n - 1, private, following)
        tl.store(out_tokens + start + j, token, mask=j < n)
        tl.store(out_locations + start + j, old, mask=j < n)
        tl.store(out_native + start + j, native, mask=j < n)
        tl.store(positions + start + j, prefix + j + 1, mask=j < n)
        tl.store(last_rows + b, start + n - 1)
        tl.store(anchors + b, tl.load(locations + b * L0))
        tl.store(pending + b, private)


@triton.jit(do_not_specialize=["B"])
def _scatter_mapping(
    reqs,
    meta,
    native,
    anchors,
    mapping,
    B,
    MAP_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(reqs + b * REQ_STRIDE)
    prefix = tl.load(meta + b)
    n = tl.load(meta + B + b)
    start = tl.load(meta + 2 * B + b)
    j = tl.arange(0, 8)
    slots = tl.load(native + start + j, mask=j < n, other=0)
    tl.store(mapping + req * MAP_STRIDE + prefix + j, slots, mask=j < n)
    tl.store(mapping + req * MAP_STRIDE + prefix - 1, tl.load(anchors + b))


def pack_commit(
    candidates, bonus, hidden, locations, reqs, counts, prefixes, pending_base
):
    bs = len(counts)
    if (
        len(prefixes) != bs
        or candidates.shape != (bs, 8)
        or hidden.shape[:2] != (bs, 8)
        or locations.shape != (bs, 8)
        or bonus.numel() != bs
        or reqs.numel() != bs
        or bs < 1
        or any(n < 1 or n > 8 for n in counts)
        or any(p < 1 for p in prefixes)
    ):
        raise ValueError("Invalid native-MTP commit pack shape")
    starts, total = [], 0
    for n in counts:
        starts.append(total)
        total += n
    device = hidden.device
    meta = torch.tensor([prefixes, counts, starts], dtype=torch.int32, device=device)
    packed = PackedAppend(
        tokens=torch.empty(total, dtype=candidates.dtype, device=device),
        hidden=torch.empty(
            (total, hidden.shape[-1]), dtype=hidden.dtype, device=device
        ),
        cache_locations=torch.empty(total, dtype=locations.dtype, device=device),
        native_locations=torch.empty(total, dtype=locations.dtype, device=device),
        positions=torch.empty(total, dtype=torch.int64, device=device),
        metadata=meta,
        last_rows=torch.empty(bs, dtype=torch.int64, device=device),
        anchor_locations=torch.empty(bs, dtype=torch.int64, device=device),
        pending_locations=torch.empty(bs, dtype=torch.int64, device=device),
    )
    _pack[(bs, triton.cdiv(hidden.shape[-1], 256))](
        candidates,
        bonus,
        hidden,
        locations,
        reqs,
        meta,
        packed.tokens,
        packed.hidden,
        packed.cache_locations,
        packed.native_locations,
        packed.positions,
        packed.last_rows,
        packed.anchor_locations,
        packed.pending_locations,
        B=bs,
        H=hidden.shape[-1],
        PENDING=pending_base,
        C0=candidates.stride(0),
        C1=candidates.stride(1),
        BONUS_STRIDE=bonus.stride(0),
        H0=hidden.stride(0),
        H1=hidden.stride(1),
        H2=hidden.stride(2),
        L0=locations.stride(0),
        L1=locations.stride(1),
        REQ_STRIDE=reqs.stride(0),
        BLOCK=256,
    )
    return packed


def scatter_commit_mapping(reqs, packed, mapping):
    _scatter_mapping[(reqs.numel(),)](
        reqs,
        packed.metadata,
        packed.native_locations,
        packed.anchor_locations,
        mapping,
        B=reqs.numel(),
        MAP_STRIDE=mapping.stride(0),
        REQ_STRIDE=reqs.stride(0),
    )
