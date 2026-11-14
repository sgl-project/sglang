import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.ops.moe.gate_topk import fpval_to_key, key_to_indx

PDL = is_arch_support_pdl()

ROUTE_MAX_ROWS = 512
ROUTE_MAX_IDS = 4096
# Larger id counts sort with the CUDA kernel (route.cuh); the Triton bitonic sort
# is faster below.
ROUTE_CUDA_MIN_IDS = 512


@triton.jit
def _route_topk(
    P,
    Bias,
    Logits,
    Ids,
    Weights,
    GlobalScale,
    route_scale,
    M,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RENORM: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    logits = tl.load(P + row * N + cols, mask, other=0)
    for group in tl.static_range(1, GROUPS):
        logits = logits + tl.load(P + (group * M + row) * N + cols, mask, other=0)
    tl.store(Logits + row * N + cols, logits, mask)
    selection = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    if HAS_BIAS:
        selection = selection + tl.load(Bias + cols, mask, other=0)
    selection = tl.where(mask, selection, float("-inf"))
    keys = fpval_to_key(selection.to(tl.uint32, bitcast=True)).to(tl.uint64) << 16
    keys = keys | (BLOCK_N - cols).to(tl.uint64)
    top = tl.sort(
        tl.topk(tl.reshape(keys, (1, BLOCK_N)), TOPK_POW2, dim=1),
        dim=1,
        descending=True,
    )
    slots = tl.arange(0, TOPK_POW2)
    ids = key_to_indx((tl.reshape(top, (TOPK_POW2,)) & 0xFFFF).to(tl.int32), BLOCK_N)
    tl.store(Ids + row * TOPK + slots, ids, slots < TOPK)
    if RENORM:
        tl.debug_barrier()
        weights = _logsigmoid_renorm8(tl.load(Logits + row * N + ids), slots)
        weights *= route_scale
        weights *= tl.load(GlobalScale)
        tl.store(Weights + row * TOPK + slots, weights)


@triton.jit
def _pick(values, slots, i: tl.constexpr):
    return tl.sum(tl.where(slots == i, values, 0.0), 0)


@triton.jit
def _logsigmoid_renorm8(logits, slots):
    """renorm_topk_logits_scaled's logsigmoid normalization of 8 logits, summing in
    the order its kernel layout reduces: ((0+2)+(1+3))+((4+6)+(5+7))."""
    log_probs = tl.minimum(logits, 0.0) - libdevice.log1p(
        libdevice.exp(-tl.abs(logits))
    )
    max_log_probs = tl.max(log_probs, 0)
    shifted = libdevice.exp(log_probs - max_log_probs)
    total = (
        (_pick(shifted, slots, 0) + _pick(shifted, slots, 2))
        + (_pick(shifted, slots, 1) + _pick(shifted, slots, 3))
    ) + (
        (_pick(shifted, slots, 4) + _pick(shifted, slots, 6))
        + (_pick(shifted, slots, 5) + _pick(shifted, slots, 7))
    )
    return libdevice.exp(log_probs - (max_log_probs + libdevice.log(total)))


def route_topk(partials, bias, topk, route_scale=None, global_scale=None):
    """Sum the router's split-K partials in group order, then sigmoid, bias, and
    stable top-k. With route_scale (top-8 only), also return the
    renorm_topk_logits_scaled weights."""
    groups, m, n = partials.shape
    logits = torch.empty((m, n), device=partials.device, dtype=torch.float32)
    ids = torch.empty((m, topk), device=partials.device, dtype=torch.int32)
    fuse_renorm = route_scale is not None
    assert not fuse_renorm or topk == 8
    weights = (
        torch.empty((m, topk), device=partials.device, dtype=torch.float32)
        if fuse_renorm
        else None
    )
    if m:
        _route_topk[(m,)](
            partials,
            bias,
            logits,
            ids,
            weights,
            global_scale,
            route_scale if fuse_renorm else 1.0,
            m,
            n,
            groups,
            bias is not None,
            topk,
            triton.next_power_of_2(topk) if not fuse_renorm else 8,
            triton.next_power_of_2(n),
            fuse_renorm,
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    if fuse_renorm:
        return logits, ids, weights
    return logits, ids


@triton.jit
def _route_metadata(
    Ids,
    Weights,
    SortedIds,
    Src2Dst,
    Dst2Src,
    Gammas,
    Counts,
    Offsets,
    BlockOffsets,
    Schedule,
    n,
    start,
    max_blocks,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SEARCH_N: tl.constexpr,
    SEARCH_E: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    ids = tl.load(Ids + offs, mask, other=65535)
    keys = tl.sort((ids << 12) | offs)
    src = keys & 0xFFF
    tl.store(SortedIds + offs, keys >> 12, mask)
    tl.store(Src2Dst + src, offs, mask)
    tl.store(Dst2Src + offs, src, mask)
    tl.store(Gammas + offs, tl.load(Weights + src, mask), mask)
    tl.debug_barrier()

    experts = tl.arange(0, BLOCK_E)
    low = tl.zeros([BLOCK_E], dtype=tl.int32)
    high = tl.full([BLOCK_E], 0, tl.int32) + n
    for _ in tl.static_range(SEARCH_N):
        mid = (low + high) // 2
        value = tl.load(SortedIds + mid, mid < n, other=65535)
        right = value < start + experts
        low = tl.where(right, mid + 1, low)
        high = tl.where(right, high, mid)
    tl.store(Offsets + experts, low, experts <= E)
    tl.debug_barrier()

    counts = tl.load(Offsets + experts + 1, experts < E, other=0) - tl.load(
        Offsets + experts, experts < E, other=0
    )
    tl.store(Counts + experts, counts, experts < E)
    blocks = tl.cdiv(counts, BLOCK_M)
    block_offsets = tl.cumsum(blocks, 0) - blocks
    total = tl.sum(blocks, 0)
    tl.store(BlockOffsets + experts, block_offsets, experts < E)
    tl.store(BlockOffsets + E, total)
    tl.debug_barrier()

    slots = tl.arange(0, BLOCK_S)
    lo = tl.zeros([BLOCK_S], dtype=tl.int32)
    hi = tl.full([BLOCK_S], E, dtype=tl.int32)
    for _ in tl.static_range(SEARCH_E):
        mid = (lo + hi + 1) // 2
        left = tl.load(BlockOffsets + mid, slots < max_blocks, other=0) > slots
        hi = tl.where(left, mid - 1, hi)
        lo = tl.where(left, lo, mid)
    base = tl.load(BlockOffsets + lo, slots < max_blocks, other=0)
    entry = tl.where(slots < total, ((slots - base) << 16) + lo, -1)
    tl.store(Schedule + slots, entry, slots < max_blocks)


@cache_once
def _route_module():
    return load_jit(
        "inkling2_route",
        cuda_files=["inkling2/route.cuh"],
        cuda_wrappers=[("route_metadata", "inkling2_route::route_metadata")],
    )


def route_metadata(ids, weights, start, num_local, num_experts, block_size_m=128):
    """One-CTA replacement for the stable id sort and the grouped-GEMM metadata of
    this rank's experts. Returns src2dst, gammas in sorted order, and
    (counts, offsets, block_offsets, schedule) for experts [start, start + num_local),
    plus dst2src, the inverse of src2dst, and the globally sorted expert ids."""
    n = ids.numel()
    if n > ROUTE_MAX_IDS or start + num_local > num_experts:
        raise ValueError("route_metadata sorts at most 4096 ids of local experts")
    ids, weights = ids.to(torch.int32), weights.to(torch.float32)
    device = ids.device
    max_blocks = triton.cdiv(n, block_size_m) + num_local - 1
    sorted_ids = torch.empty(n, device=device, dtype=torch.int32)
    src2dst = torch.empty(n, device=device, dtype=torch.int32)
    dst2src = torch.empty(n, device=device, dtype=torch.int32)
    gammas = torch.empty(n, device=device, dtype=torch.float32)
    counts = torch.empty(num_local, device=device, dtype=torch.int32)
    offsets = torch.empty(num_local + 1, device=device, dtype=torch.int32)
    block_offsets = torch.empty(num_local + 1, device=device, dtype=torch.int32)
    schedule = torch.empty(max_blocks, device=device, dtype=torch.int32)
    if n > ROUTE_CUDA_MIN_IDS:
        _route_module().route_metadata(
            ids,
            weights,
            sorted_ids,
            src2dst,
            dst2src,
            gammas,
            counts,
            offsets,
            block_offsets,
            schedule,
            start,
            num_experts,
            block_size_m,
        )
    else:
        block_n = max(triton.next_power_of_2(n), 16)
        block_e = triton.next_power_of_2(num_local + 1)
        _route_metadata[(1,)](
            ids,
            weights,
            sorted_ids,
            src2dst,
            dst2src,
            gammas,
            counts,
            offsets,
            block_offsets,
            schedule,
            n,
            start,
            max_blocks,
            num_local,
            block_n,
            block_e,
            triton.next_power_of_2(max_blocks),
            block_size_m,
            block_n.bit_length(),
            block_e.bit_length(),
            num_warps=min(32, max(4, block_n // 128)),
            launch_pdl=PDL,
        )
    return (
        src2dst,
        dst2src,
        gammas,
        counts,
        offsets,
        block_offsets,
        schedule,
        sorted_ids,
    )
