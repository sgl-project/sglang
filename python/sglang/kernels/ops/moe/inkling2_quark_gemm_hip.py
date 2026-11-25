"""Triton grouped GEMM for the packed MXFP4 / UINT4-AWQ experts of Inkling-2 Quark
exports. Weight tiles are decoded in the K loop to the BF16 values of the reference
dequantizer and fed to a BF16 MFMA with FP32 accumulation; nothing is repacked.

Existing MoE kernels expect other layouts or recipes (the int4 fused MoE another bit
order, aiter's MXFP4 MoE another activation rounding), and the native Inkling-2
kernel decodes only NVFP4/MXFP8. This one keeps the native routing and split-K
scheme and adds the two decoders, in a separate file so native serving is unchanged."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.inkling_moe import compute_expert_block_metadata

# Weight layouts, E local experts, N outputs, K inputs (as in the checkpoint):
# MXFP4      uint8 [E, N, K/2], even k in the low nibble; E8M0 scale uint8 [E, N, K/32]
# UINT4_AWQ  int32 [E, K, N/8] AutoAWQ words, scale [E, K/128, N], zero point
#            int32 [E, K/128, N/8]; W = bf16((q - zp) * scale), rounded once like Quark
MXFP4 = 0
UINT4_AWQ = 1

UINT4_GROUP = 128
MXFP4_BLOCK = 32
# Globals read by a Triton kernel must be constexprs.
_GROUP = tl.constexpr(UINT4_GROUP)
_MXFP4 = tl.constexpr(MXFP4)


@triton.jit
def _expand8(w):
    """int32 (R, C) AutoAWQ words -> int32 (R, 8 * C) 4-bit values in column order.
    Nibble i holds column (0, 2, 4, 6, 1, 3, 5, 7)[i]; tl.join appends a minor
    dimension, so this nesting flattens to c0, c1, ..., c7
    (join(join(c0, c1), join(c2, c3)) would permute)."""
    c0 = w & 0xF
    c1 = (w >> 16) & 0xF
    c2 = (w >> 4) & 0xF
    c3 = (w >> 20) & 0xF
    c4 = (w >> 8) & 0xF
    c5 = (w >> 24) & 0xF
    c6 = (w >> 12) & 0xF
    c7 = (w >> 28) & 0xF
    t = tl.join(
        tl.join(tl.join(c0, c4), tl.join(c2, c6)),
        tl.join(tl.join(c1, c5), tl.join(c3, c7)),
    )
    return tl.reshape(t, (w.shape[0], w.shape[1] * 8))


@triton.jit
def _uint4_awq_weight_tile(
    W,
    WS,
    WZ,
    expert,
    k0,
    n0,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """(BK, BN) BF16 tile of (q - zp) * scale, computed in FP32 and rounded once.
    The K tile lies inside one 128-wide scale group. v_pk_mul_f32 zeroes lanes 48-63
    at BN = BK = 128 in the native NVFP4 kernel; the one-hot test runs every config."""
    g = k0 // _GROUP
    krows = expert * K + k0 + tl.arange(0, BK)
    ncols = n0 + tl.arange(0, BN)
    wcols = n0 // 8 + tl.arange(0, BN // 8)
    if N % BN != 0:
        # N tail: read the last real column instead (the result is never stored)
        ncols = tl.minimum(ncols, N - 1)
        wcols = tl.minimum(wcols, N // 8 - 1)
    sc = tl.load(WS + (expert * (K // _GROUP) + g) * N + ncols).to(tl.float32)
    zrow = WZ + (expert * (K // _GROUP) + g) * (N // 8)
    words = tl.load(W + krows[:, None] * (N // 8) + wcols[None, :])
    q = _expand8(words)
    zp = _expand8(tl.load(zrow + wcols)[None, :])
    return ((q - zp).to(tl.float32) * sc[None, :]).to(tl.bfloat16)


@triton.jit
def _grouped_quark_gemm(
    X,
    W,
    WS,
    WZ,
    Y,
    Offsets,
    Dst2Src,
    Schedule,
    rows,
    N: tl.constexpr,
    K: tl.constexpr,
    GATHER_DIV: tl.constexpr,
    FMT: tl.constexpr,
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
    n0 = tl.program_id(1) * BN
    cols = n0 + tl.arange(0, BN)
    col_mask = cols < N
    split = tl.program_id(2)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(split * K_SPLIT, (split + 1) * K_SPLIT, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + x_rows[:, None] * K + k[None, :], row_mask[:, None], other=0)
        if FMT == _MXFP4:
            # tl.dot_scaled on (BK/2, BN) packed bytes and (BN, BK/32) scales. No inline
            # asm convert: LLVM cannot see its VALU write, so v_mfma may read it early.
            wrows = expert * N + cols
            wt = tl.load(
                W
                + wrows[None, :] * (K // 2)
                + (k0 // 2 + tl.arange(0, BK // 2))[:, None],
                col_mask[None, :],
                other=0,
            )
            ws = tl.load(
                WS
                + wrows[:, None] * (K // 32)
                + (k0 // 32 + tl.arange(0, BK // 32))[None, :],
                col_mask[:, None],
                other=127,
            )
            acc = tl.dot_scaled(x, None, "bf16", wt, ws, "e2m1", acc)
        else:
            w = _uint4_awq_weight_tile(W, WS, WZ, expert, k0, n0, N, K, BN, BK)
            acc = tl.dot(x, w, acc)
    # With a split K the partials go to slab `split` of Y; the consumer sums the slabs.
    tl.store(
        Y + (split * rows + src[:, None]) * N + cols[None, :],
        acc.to(Y.dtype.element_ty),
        row_mask[:, None] & col_mask[None, :],
    )


class QConfig(NamedTuple):
    bm: int
    bn: int
    bk: int
    split_k: int
    warps: int
    stages: int


# (max rows, config) per format, rows = tokens x top-8 over all experts, scaled by
# num_experts / 128; tuned on MI355X with the real weights. Both GEMMs of a layer share
# one routing schedule, so BM depends on the row count only.
# Entry: least worst-case w13 + w2 slowdown in its bucket; uniform top-8, 1-8192 tokens.
_TABLES = {
    MXFP4: [  # T <= 32, 128, 256, 512, 1024, more
        (256, QConfig(bm=16, bn=128, bk=256, split_k=3, warps=4, stages=2)),
        (1024, QConfig(bm=16, bn=128, bk=256, split_k=1, warps=4, stages=2)),
        (2048, QConfig(bm=32, bn=128, bk=128, split_k=1, warps=4, stages=2)),
        (4096, QConfig(bm=64, bn=128, bk=128, split_k=1, warps=4, stages=2)),
        (8192, QConfig(bm=64, bn=256, bk=128, split_k=1, warps=4, stages=2)),
        (None, QConfig(bm=128, bn=256, bk=128, split_k=1, warps=4, stages=1)),
    ],
    UINT4_AWQ: [
        (256, QConfig(bm=16, bn=128, bk=64, split_k=9, warps=2, stages=2)),
        (1024, QConfig(bm=16, bn=128, bk=128, split_k=3, warps=2, stages=2)),
        (2048, QConfig(bm=32, bn=128, bk=64, split_k=1, warps=2, stages=2)),
        (4096, QConfig(bm=64, bn=128, bk=128, split_k=1, warps=4, stages=2)),
        (8192, QConfig(bm=128, bn=128, bk=64, split_k=1, warps=8, stages=2)),
        (16384, QConfig(bm=128, bn=128, bk=64, split_k=1, warps=8, stages=2)),
        (None, QConfig(bm=256, bn=128, bk=64, split_k=1, warps=4, stages=1)),
    ],
}


def default_config_quark(
    fmt: int, rows: int, num_experts: int, scale_dtype=None
) -> QConfig:
    """Config for `rows` routed rows over `num_experts` experts. UINT4_AWQ with FP32
    scales takes 8 warps above 1024 tokens (the 4-warp config is 30-40 % slower)."""
    experts = max(1, num_experts)
    if (
        fmt == UINT4_AWQ
        and scale_dtype == torch.float32
        and rows * 128 > 8192 * experts
    ):
        return QConfig(bm=128, bn=128, bk=64, split_k=1, warps=8, stages=2)
    table = _TABLES[fmt]
    for max_rows, cfg in table:
        if max_rows is None or rows <= max_rows * experts // 128:
            return cfg
    return table[-1][1]


def resolve_config(cfg: QConfig, fmt: int, width: int) -> QConfig:
    """BK halved until it divides K, split-K clamped to a divisor of the K tiles."""
    bk = cfg.bk
    while width % bk and bk > MXFP4_BLOCK:
        bk //= 2
    if width % bk or bk % MXFP4_BLOCK:
        raise ValueError(f"K = {width} must be a multiple of {MXFP4_BLOCK}")
    if fmt == UINT4_AWQ and (UINT4_GROUP % bk or width % UINT4_GROUP):
        raise ValueError(
            f"UINT4_AWQ needs K a multiple of {UINT4_GROUP} and BK dividing it "
            f"(K = {width}, BK = {bk}): a K tile must lie inside one scale group"
        )
    tiles = width // bk
    split_k = max(s for s in range(1, cfg.split_k + 1) if tiles % s == 0)
    return cfg._replace(bk=bk, split_k=split_k)


def _check_operands(fmt, x, weight, scale, zero_point, out_size):
    """The kernel trusts shapes, dtypes and strides; check them (host metadata only)."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(f"x must be a contiguous 2-D BF16 tensor: {x.dtype} {x.shape}")
    for name, t in (("weight", weight), ("scale", scale), ("zero_point", zero_point)):
        if t is not None and not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    experts, width = weight.shape[0], x.shape[1]
    if fmt == MXFP4:
        expected = (
            (experts, out_size, width // 2),
            (experts, out_size, width // MXFP4_BLOCK),
            None,
        )
        dtypes_ok = weight.dtype == scale.dtype == torch.uint8 and zero_point is None
    else:
        expected = (
            (experts, width, out_size // 8),
            (experts, width // UINT4_GROUP, out_size),
            (experts, width // UINT4_GROUP, out_size // 8),
        )
        dtypes_ok = (
            weight.dtype == torch.int32
            and scale.dtype in (torch.bfloat16, torch.float32)
            and zero_point is not None
            and zero_point.dtype == torch.int32
            and out_size % 8 == 0
        )
    actual = tuple(
        None if t is None else tuple(t.shape) for t in (weight, scale, zero_point)
    )
    if not dtypes_ok or actual != expected:
        raise ValueError(
            f"format {fmt}, N = {out_size}, K = {width}: expected shapes {expected}, got "
            f"{actual}; dtypes weight {weight.dtype}, scale {scale.dtype}, zero point "
            f"{None if zero_point is None else zero_point.dtype}"
        )


def expert_gemm_quark(
    x,
    weight,
    scale,
    zero_point,
    route_info,
    out_size,
    gather_div,
    out_dtype,
    rows,
    *,
    fmt,
    num_experts,
    config: QConfig | None = None,
):
    """Row r (source order) of the result is x[r // gather_div] @ W_e^T for its local
    expert e; other rows are unset. route_info comes from inkling2_experts_hip.route; a
    block_m other than the config's BM costs a schedule rebuild. Returns the split-K
    partials (splits, rows, out_size), FP32 when splits > 1, else out_dtype; the UINT4
    callers sum them in split order."""
    _check_operands(fmt, x, weight, scale, zero_point, out_size)
    num_local = weight.shape[0]
    if (
        route_info.offsets.numel() != num_local + 1
        or route_info.dst2src.numel() != rows
        or x.shape[0] * gather_div < rows
    ):
        raise ValueError(
            f"route / operand mismatch: {route_info.offsets.numel() - 1} local experts "
            f"in the route vs {num_local} weights, {route_info.dst2src.numel()} routed "
            f"rows vs rows = {rows}, x has {x.shape[0]} rows (gather_div {gather_div})"
        )
    if not rows:
        return torch.empty((1, 0, out_size), dtype=out_dtype, device=x.device)
    width = x.shape[1]
    if config is None:
        config = default_config_quark(fmt, rows, num_experts, scale.dtype)
    cfg = resolve_config(config, fmt, width)
    if route_info.schedule is not None and route_info.block_m == cfg.bm:
        schedule = route_info.schedule
    else:
        _, schedule = compute_expert_block_metadata(
            route_info.counts, rows, block_size_m=cfg.bm
        )
    partials = torch.empty(
        (cfg.split_k, rows, out_size),
        dtype=out_dtype if cfg.split_k == 1 else torch.float32,
        device=x.device,
    )
    # Valid schedule entries come first; at most one partial block per active expert.
    blocks = min(schedule.numel(), triton.cdiv(rows, cfg.bm) + min(num_local, rows))
    _grouped_quark_gemm[(blocks, triton.cdiv(out_size, cfg.bn), cfg.split_k)](
        x,
        weight,
        scale,
        zero_point if zero_point is not None else scale,  # unused for MXFP4
        partials,
        route_info.offsets,
        route_info.dst2src,
        schedule,
        rows,
        out_size,
        width,
        gather_div,
        fmt,
        cfg.bm,
        cfg.bn,
        cfg.bk,
        width // cfg.split_k,
        num_warps=cfg.warps,
        num_stages=cfg.stages,
    )
    return partials
