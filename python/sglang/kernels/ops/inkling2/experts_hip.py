"""NVFP4 and MXFP8 expert GEMMs on ROCm, on rows in source (token, top-k) order.

Like the Hopper path, activations arrive fake-quantized to their exact NVFP4 / MXFP8
values and the weights are dequantized in the main loop, so every product matches the
block-scaled MMA exactly; only the FP32 accumulation order differs. NVFP4 values
without the outer scale lie in [2^-10, 2688], exact in FP16, so on gfx950 the NVFP4
GEMM runs FP16 MMA on natively converted weights; MXFP8 weights convert to BF16 with
their UE8M0 scale in one step. The GEMM reads the rows routed to this rank's experts
through dst2src and writes each result back to its source row."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.inkling2.experts_sm90 import _ue8m0
from sglang.kernels.ops.inkling2.portability import IS_HIP
from sglang.kernels.ops.inkling2.route import ROUTE_MAX_IDS, route_metadata
from sglang.kernels.ops.moe.inkling_moe import (
    compute_expert_block_metadata,
    compute_grouped_gemm_metadata,
)


@triton.jit
def _nvfp4_weight_tile(
    W,
    WS,
    expert,
    cols,
    col_mask,
    k0,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """(BN, BK) BF16 tile of e2m1 values times their e4m3 block-16 scales."""
    rows = expert * N + cols
    packed_cols = k0 // 2 + tl.arange(0, BK // 2)
    p = tl.load(
        W + rows[:, None] * (K // 2) + packed_cols[None, :],
        col_mask[:, None],
        other=0,
    ).to(tl.int16)
    nibbles = tl.reshape(tl.join(p & 0xF, (p >> 4) & 0xF), (BN, BK))
    # e2m1 bits placed in an fp16 so that value * 2^14 is the e2m1 value.
    bits = ((nibbles & 8) << 12) | ((nibbles & 7) << 9)
    values = bits.to(tl.float16, bitcast=True) * 16384.0
    scale_cols = k0 // 16 + tl.arange(0, BK // 16)
    scales = tl.load(
        WS + rows[:, None] * (K // 16) + scale_cols[None, :], col_mask[:, None]
    ).to(tl.float16)
    # Exact in FP16 like _nvfp4_native_tile. In FP32 the scale broadcast became
    # v_pk_mul_f32 reading the scale pair through op_sel, which on gfx950 sometimes
    # zeroes the low result in lanes 48-63 (seen at BN = BK = 128).
    w = tl.reshape(values, (BN, BK // 16, 16)) * scales[:, :, None]
    return tl.reshape(w, (BN, BK)).to(tl.bfloat16)


@triton.jit
def _mxfp8_weight_tile(
    W,
    WS,
    expert,
    cols,
    col_mask,
    k0,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """(BN, BK) BF16 tile of e4m3 values times their ue8m0 block-32 scales."""
    rows = expert * N + cols
    k = k0 + tl.arange(0, BK)
    w = tl.load(W + rows[:, None] * K + k[None, :], col_mask[:, None]).to(tl.float32)
    scale_cols = k0 // 32 + tl.arange(0, BK // 32)
    exponent = tl.load(
        WS + rows[:, None] * (K // 32) + scale_cols[None, :],
        col_mask[:, None],
        other=127,
    )
    w = tl.reshape(w, (BN, BK // 32, 32)) * _ue8m0(exponent)[:, :, None]
    return tl.reshape(w, (BN, BK)).to(tl.bfloat16)


# v_cvt_scalef32_pk_* exist from gfx950 (CDNA4) on.
NATIVE_CVT = tl.constexpr(
    bool(IS_HIP)
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties().gcnArchName.startswith("gfx95")
)
if NATIVE_CVT:
    try:
        from sglang.kernels.ops.inkling2 import experts_gluon
    except ImportError:  # Triton without Gluon's CDNA4 ops
        experts_gluon = None
else:
    experts_gluon = None
# Whether experts can take the MFMA order of experts_gluon (mfma_order_*).
MFMA_ORDER = experts_gluon is not None


@triton.jit
def _cvt_pairs(pairs, scale, OP: tl.constexpr, DTYPE: tl.constexpr):
    """`OP dst, pairs, scale` on each lane: two values packed in dst -> (low, high)."""
    packed = tl.inline_asm_elementwise(
        OP + " $0, $1, $2",
        "=v,v,v",
        [pairs, scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    low = (packed & 0xFFFF).to(tl.int16).to(DTYPE, bitcast=True)
    high = (packed >> 16).to(tl.int16).to(DTYPE, bitcast=True)
    return low, high


@triton.jit
def _nvfp4_native_tile(
    W,
    WS,
    expert,
    cols,
    col_mask,
    k0,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """_nvfp4_weight_tile in FP16 through v_cvt_scalef32_pk_f16_fp4 (unit scale)."""
    rows = expert * N + cols
    packed_cols = k0 // 2 + tl.arange(0, BK // 2)
    p = tl.load(
        W + rows[:, None] * (K // 2) + packed_cols[None, :],
        col_mask[:, None],
        other=0,
    ).to(tl.int32)
    low, high = _cvt_pairs(
        p, tl.full(p.shape, 1.0, tl.float32), "v_cvt_scalef32_pk_f16_fp4", tl.float16
    )
    values = tl.reshape(tl.join(low, high), (BN, BK // 16, 16))
    scale_cols = k0 // 16 + tl.arange(0, BK // 16)
    scales = tl.load(
        WS + rows[:, None] * (K // 16) + scale_cols[None, :], col_mask[:, None]
    ).to(tl.float16)
    return tl.reshape(values * scales[:, :, None], (BN, BK))


@triton.jit
def _mfma_order_units(
    W, expert, k0, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """The (BN, BK // 2) 2-value units (NVFP4 bytes, MXFP8 16-bit pairs) of a tile
    in experts_gluon's MFMA order, loaded in storage order and permuted."""
    tl.static_assert(BK <= 256)
    # [16-row group][K // 128][lane group][row][32-K block][4 units]
    groups = tl.program_id(1) * (BN // 16) + tl.arange(0, BN // 16)
    halves: tl.constexpr = max(BK // 256, 1)
    run: tl.constexpr = min(BK // 8, 16)
    offsets = (
        (groups * 16)[:, None, None, None, None] * (K // 2)
        + (k0 // 128 + tl.arange(0, halves))[None, :, None, None, None] * 1024
        + tl.arange(0, 4)[None, None, :, None, None] * 256
        + tl.arange(0, 16)[None, None, None, :, None] * 16
        + (k0 % 128 // 32 * 4 + tl.arange(0, run))[None, None, None, None, :]
    )
    units = tl.load(W + expert * N * (K // 2) + offsets)
    units = tl.reshape(units, (BN // 16, halves, 4, 16, run // 4, 4))
    return tl.reshape(tl.permute(units, (0, 3, 1, 4, 2, 5)), (BN, BK // 2))


@triton.jit
def _nvfp4_mfma_order_tile(
    W,
    WS,
    expert,
    k0,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """_nvfp4_native_tile on weights and scales in MFMA order."""
    p = _mfma_order_units(W, expert, k0, N, K, BN, BK).to(tl.int32)
    low, high = _cvt_pairs(
        p, tl.full(p.shape, 1.0, tl.float32), "v_cvt_scalef32_pk_f16_fp4", tl.float16
    )
    values = tl.reshape(tl.join(low, high), (BN, BK // 16, 16))
    # Scales: [16-row group][K // 256][16-K half][row][32-K block].
    groups = tl.program_id(1) * (BN // 16) + tl.arange(0, BN // 16)
    blocks: tl.constexpr = min(BK // 32, 8)
    s_offsets = (
        groups[:, None, None, None] * K
        + tl.arange(0, 2)[None, :, None, None] * 128
        + tl.arange(0, 16)[None, None, :, None] * 8
        + (k0 // 256 * 256 + k0 % 256 // 32 + tl.arange(0, blocks))[None, None, None, :]
    )
    scales = tl.load(WS + expert * N * (K // 16) + s_offsets)
    scales = tl.reshape(tl.permute(scales, (0, 2, 3, 1)), (BN, BK // 16)).to(tl.float16)
    return tl.reshape(values * scales[:, :, None], (BN, BK))


@triton.jit
def _mxfp8_mfma_order_tile(
    W,
    WS,
    expert,
    cols,
    k0,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """_mxfp8_native_tile on weights in MFMA order (the scales keep their layout)."""
    pairs = _mfma_order_units(W.to(tl.pointer_type(tl.int16)), expert, k0, N, K, BN, BK)
    pairs = pairs.to(tl.int32) & 0xFFFF
    rows = expert * N + cols
    scale_cols = k0 // 32 + tl.arange(0, BK // 32)
    exponent = tl.load(WS + rows[:, None] * (K // 32) + scale_cols[None, :])
    scale = tl.broadcast_to(_ue8m0(exponent)[:, :, None], (BN, BK // 32, 16))
    low, high = _cvt_pairs(
        pairs,
        tl.reshape(scale, (BN, BK // 2)),
        "v_cvt_scalef32_pk_bf16_fp8",
        tl.bfloat16,
    )
    return tl.reshape(tl.join(low, high), (BN, BK))


@triton.jit
def _mxfp8_native_tile(
    W,
    WS,
    expert,
    cols,
    col_mask,
    k0,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """_mxfp8_weight_tile through v_cvt_scalef32_pk_bf16_fp8, which applies the
    exponent of its FP32 scale: exactly the UE8M0 scale."""
    rows = expert * N + cols
    pair_cols = k0 // 2 + tl.arange(0, BK // 2)
    pairs = (
        tl.load(
            W.to(tl.pointer_type(tl.int16))
            + rows[:, None] * (K // 2)
            + pair_cols[None, :],
            col_mask[:, None],
            other=0,
        ).to(tl.int32)
        & 0xFFFF
    )
    scale_cols = k0 // 32 + tl.arange(0, BK // 32)
    exponent = tl.load(
        WS + rows[:, None] * (K // 32) + scale_cols[None, :],
        col_mask[:, None],
        other=127,
    )
    scale = tl.broadcast_to(_ue8m0(exponent)[:, :, None], (BN, BK // 32, 16))
    low, high = _cvt_pairs(
        pairs,
        tl.reshape(scale, (BN, BK // 2)),
        "v_cvt_scalef32_pk_bf16_fp8",
        tl.bfloat16,
    )
    return tl.reshape(tl.join(low, high), (BN, BK))


@triton.jit
def _grouped_dequant_gemm(
    X,
    W,
    WS,
    Y,
    Offsets,
    Dst2Src,
    Schedule,
    rows,
    N,
    K: tl.constexpr,
    GATHER_DIV: tl.constexpr,
    NVFP4: tl.constexpr,
    MFMA_ORDER: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_SPLIT: tl.constexpr,
):
    entry = tl.load(Schedule + tl.program_id(0))
    if entry == -1:
        return
    expert = (entry & 0xFFFF).to(tl.int64)
    block = entry >> 16
    lo = tl.load(Offsets + expert)
    hi = tl.load(Offsets + expert + 1)
    positions = lo + block * BM + tl.arange(0, BM)
    row_mask = positions < hi
    src = tl.load(Dst2Src + positions, row_mask, other=0).to(tl.int64)
    x_rows = src // GATHER_DIV
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    col_mask = cols < N
    split = tl.program_id(2)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(split * K_SPLIT, (split + 1) * K_SPLIT, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + x_rows[:, None] * K + k[None, :], row_mask[:, None], other=0)
        if NVFP4 and MFMA_ORDER:
            w = _nvfp4_mfma_order_tile(W, WS, expert, k0, N, K, BN, BK)
        elif MFMA_ORDER:
            w = _mxfp8_mfma_order_tile(W, WS, expert, cols, k0, N, K, BN, BK)
        elif NVFP4 and NATIVE_CVT:
            w = _nvfp4_native_tile(W, WS, expert, cols, col_mask, k0, N, K, BN, BK)
        elif NVFP4:
            w = _nvfp4_weight_tile(W, WS, expert, cols, col_mask, k0, N, K, BN, BK)
        elif NATIVE_CVT:
            w = _mxfp8_native_tile(W, WS, expert, cols, col_mask, k0, N, K, BN, BK)
        else:
            w = _mxfp8_weight_tile(W, WS, expert, cols, col_mask, k0, N, K, BN, BK)
        w = w.to(x.dtype)
        acc = tl.dot(x, tl.trans(w), acc)
    # With a split K the partials go to slab `split` of Y, reduced in split order.
    tl.store(
        Y + (split * rows + src[:, None]) * N + cols[None, :],
        acc.to(Y.dtype.element_ty),
        row_mask[:, None] & col_mask[None, :],
    )


class GemmConfig(NamedTuple):
    bm: int
    bn: int
    bk: int
    split_k: int
    warps: int
    stages: int


def default_config(rows, num_experts, nvfp4=False, mfma_order=False):
    """Tiles from MI350X sweeps (EP8, 48 of 384 experts local): decode is
    weight-bandwidth bound and splits K while few experts are active. 16-row tiles on
    MFMA-order weights run experts_gluon, whose width is fixed. mfma_order: weights in
    experts_gluon's MFMA order, which take other prefill tiles with the same K split
    (bitwise)."""
    # MTP verify at c1 is 64 rows (8 draft tokens x top-8); above 64 rows the
    # consumers' reads of the FP32 partials cost more than splitting K saves
    if rows <= 64:
        return GemmConfig(16, 64, 256, 6, 4, 1)
    if rows <= 256:
        return GemmConfig(16, 64, 256, 1, 4, 1)
    if rows <= 1024:
        return GemmConfig(16, 64, 256, 2, 4, 1)
    rows_per_expert = triton.cdiv(rows, num_experts)
    if rows_per_expert <= 32:
        # MTP verify above c16, decode above c128
        return GemmConfig(16, 64, 256, 1, 4, 1)
    # MFMA-order prefill tiles are timed on a real 8k-token prefill's routing (rows per
    # expert up to 25x the mean); uniform routing picks slower tiles here
    if mfma_order and (rows_per_expert > 64 or not nvfp4):
        return GemmConfig(64, 256, 128, 1, 8, 1)
    if rows_per_expert <= 64:
        if mfma_order:
            return GemmConfig(64, 128, 128, 1, 4, 1)
        return GemmConfig(64, 128, 64, 1, 8, 2)
    # swept on MI350X at 8192 tokens (~170 rows per expert)
    return GemmConfig(128, 128, 128, 1, 4, 1)


def activation_dtype(weight_dtype):
    """The dtype expert_gemm takes the fake-quantized activations in."""
    return (
        torch.float16 if weight_dtype == torch.uint8 and NATIVE_CVT else torch.bfloat16
    )


class Route(NamedTuple):
    dst2src: torch.Tensor  # sorted position -> source (token, top-k) row
    counts: torch.Tensor  # rows of each local expert
    offsets: torch.Tensor  # absolute sorted positions of the local experts' rows
    block_m: int | None  # the block height of schedule, if routing built one
    schedule: torch.Tensor | None


def route(ids, start, num_local, num_experts, weights=None, *, block_m=None):
    """Sort the flattened routed ids by expert for this rank's experts
    [start, start + num_local). Up to ROUTE_MAX_IDS ids sort in one launch that also
    builds the block_m schedule."""
    flat = ids.flatten()
    if block_m is not None and flat.numel() <= ROUTE_MAX_IDS:
        if weights is None:
            weights = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
        _, dst2src, _, counts, offsets, _, schedule, _ = route_metadata(
            flat, weights.flatten(), start, num_local, num_experts, block_size_m=block_m
        )
        return Route(dst2src, counts, offsets, block_m, schedule)
    sorted_ids, dst2src = torch.sort(flat.to(torch.int16), stable=True)
    counts, offsets, _, _ = compute_grouped_gemm_metadata(
        sorted_ids, num_experts, block_size_m=128
    )
    counts = counts[start : start + num_local]
    offsets = offsets[start : start + num_local + 1]
    return Route(dst2src.to(torch.int32), counts, offsets, None, None)


def expert_gemm(
    x,
    weight,
    scales,
    route_info,
    out_size,
    gather_div,
    out_dtype,
    rows,
    *,
    num_experts,
    config=None,
    mfma_order=False,
):
    """Row r (source order, routed to a local expert) of the result is
    x[r // gather_div] times that expert's dequantized weights; other rows are unset.
    NVFP4 weights are (experts, n, k // 2) uint8 with (experts, n, k // 16) e4m3
    scales; MXFP8 weights are (experts, n, k) e4m3 with (experts, n, k // 32) ue8m0
    scales. The NVFP4 outer scales are left to the consumer. rows counts every routed
    row over num_experts global experts. Returns the split-K partials as (splits,
    rows, out_size) FP32 (one split: the result itself, unsqueezed), which the
    consumer sums in split order.
    mfma_order: weights and scales in experts_gluon's MFMA order; 16-row tiles then
    run the Gluon kernel, which is bitwise the Triton one."""
    dst2src, counts, offsets = route_info.dst2src, route_info.counts, route_info.offsets
    nvfp4 = weight.dtype == torch.uint8
    assert x.dtype == activation_dtype(weight.dtype), x.dtype
    assert not mfma_order or MFMA_ORDER
    num_local = weight.shape[0]
    width = x.shape[1]
    out = torch.empty((rows, out_size), dtype=out_dtype, device=x.device)
    if not rows:
        return out.unsqueeze(0)
    cfg = config or default_config(rows, num_experts, nvfp4, mfma_order)
    assert width % cfg.bk == 0 and out_size % cfg.bn == 0
    # The largest split of at most cfg.split_k that divides the K tiles evenly.
    tiles = width // cfg.bk
    split_k = max(s for s in range(1, cfg.split_k + 1) if tiles % s == 0)
    cfg = cfg._replace(split_k=split_k)
    k_split = width // split_k
    if route_info.block_m == cfg.bm:
        schedule = route_info.schedule
    else:
        _, schedule = compute_expert_block_metadata(counts, rows, block_size_m=cfg.bm)
    partials = (
        out
        if cfg.split_k == 1
        else torch.empty(
            (cfg.split_k, rows, out_size), dtype=torch.float32, device=x.device
        )
    )
    # Valid schedule entries come first; at most one partial block per active expert.
    blocks = min(schedule.numel(), triton.cdiv(rows, cfg.bm) + min(num_local, rows))
    if mfma_order and (cfg.bm, cfg.bk) == (experts_gluon.BM, experts_gluon.BK):
        # BN is free: the result depends only on the K split.
        experts_gluon.grouped_gemm(
            x,
            weight,
            scales,
            partials,
            route_info._replace(schedule=schedule),
            rows,
            gather_div,
            cfg.split_k,
            blocks,
        )
    else:
        _grouped_dequant_gemm[(blocks, triton.cdiv(out_size, cfg.bn), cfg.split_k)](
            x,
            weight,
            scales,
            partials,
            offsets,
            dst2src,
            schedule,
            rows,
            out_size,
            width,
            gather_div,
            nvfp4,
            mfma_order,
            cfg.bm,
            cfg.bn,
            cfg.bk,
            k_split,
            num_warps=cfg.warps,
            num_stages=cfg.stages,
        )
    return partials if cfg.split_k > 1 else out.unsqueeze(0)
