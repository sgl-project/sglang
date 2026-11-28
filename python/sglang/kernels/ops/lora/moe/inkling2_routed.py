"""Shared-outer LoRA deltas on Inkling2's routed expert rows (CUDA, Triton).

Rows follow the Inkling2 routed kernels: expert e owns rows [offsets[e], offsets[e + 1]),
row r reads token dst2src[r] // top_k, and the gate/up accumulator interleaves gate
and up columns. With NVFP4 scales the accumulator is the raw FP32 GEMM output, scaled
per row by x_amax[r] * w_amax[e] / (1536 * denominator) when the SwiGLU reads it; MXFP8
and BF16 accumulators are already scaled. Coverage: any number of local experts, rank a
power of two, intermediate width a multiple of 128 for gate_up B reads.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.inkling2_moe import (
    _capped_swiglu,
    _nvfp4_row_scale,
    _row_expert,
    _sm_count,
)


@triton.jit
def _gate_up_expand(
    H,
    B,
    Acc,
    XAmax,
    WAmax,
    WDenominator,
    Dst2Src,
    Offsets,
    stride_be,
    N: tl.constexpr,
    R: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    split,
):
    expert = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    start = tl.load(Offsets + expert)
    end = tl.load(Offsets + expert + 1)
    if start + tl.program_id(2) * BLOCK_M >= end:
        return
    ks = tl.arange(0, R)
    col_mask = cols < N
    # B[e] is [2N, R]: gate rows [0, N), up rows [N, 2N).
    b = B + expert.to(tl.int64) * stride_be
    b_gate = tl.load(b + cols[None, :] * R + ks[:, None], col_mask[None, :], other=0)
    b_up = tl.load(
        b + (N + cols[None, :]) * R + ks[:, None], col_mask[None, :], other=0
    )
    for m in range(start + tl.program_id(2) * BLOCK_M, end, split * BLOCK_M):
        rows = m + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        token = tl.load(Dst2Src + rows, row_mask, other=0) // TOP_K
        h = H + token.to(tl.int64)[:, None] * (2 * R) + ks[None, :]
        gate = tl.dot(tl.load(h, row_mask[:, None], other=0), b_gate)
        up = tl.dot(tl.load(h + R, row_mask[:, None], other=0), b_up)
        if SCALED:
            scale = _nvfp4_row_scale(XAmax, WAmax, WDenominator, expert, rows)
            # A zero row has a zero delta; keep 0 / 0 out of the accumulator.
            inverse = tl.where(scale > 0, 1.0 / scale, 0.0)[:, None]
            gate *= inverse
            up *= inverse
        mask = row_mask[:, None] & col_mask[None, :]
        acc = Acc + rows.to(tl.int64)[:, None] * (2 * N) + 2 * cols[None, :]
        base_gate = tl.load(acc, mask, other=0).to(tl.float32)
        tl.store(acc, (base_gate + gate).to(Acc.dtype.element_ty), mask)
        base_up = tl.load(acc + 1, mask, other=0).to(tl.float32)
        tl.store(acc + 1, (base_up + up).to(Acc.dtype.element_ty), mask)


@triton.jit
def _routed_act(
    X,
    XAmax,
    WAmax,
    WDenominator,
    Offsets,
    G,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    SCALED: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH: tl.constexpr,
    programs,
):
    # The capped swiglu's inline asm compiles badly on 2D tiles, so this stays 1D.
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    first, last = tl.load(Offsets) + tl.program_id(0), tl.load(Offsets + E)
    for row in range(first, last, programs):
        gate = tl.load(X + row.to(tl.int64) * 2 * N + 2 * cols, mask, other=0)
        up = tl.load(X + row.to(tl.int64) * 2 * N + 2 * cols + 1, mask, other=0)
        if SCALED:
            # Launch 3 (swiglu_quantize_routed) rounds the scaled gate and up to BF16.
            scale = _nvfp4_row_scale(
                XAmax, WAmax, WDenominator, _row_expert(Offsets, row, E, SEARCH), row
            )
            gate = (gate * scale).to(Y.dtype.element_ty)
            up = (up * scale).to(Y.dtype.element_ty)
        gamma = tl.load(G + row)
        y = _capped_swiglu(
            gate.to(tl.float32), up.to(tl.float32), gamma, GATE_CAP, UP_CAP
        )
        tl.store(Y + row.to(tl.int64) * N + cols, y, mask)


@triton.jit
def _down_shrink(
    Act,
    A,
    Inner,
    Offsets,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    split,
):
    expert = tl.program_id(0)
    start = tl.load(Offsets + expert)
    end = tl.load(Offsets + expert + 1)
    ks = tl.arange(0, R)
    a = A + expert.to(tl.int64) * R * N
    for m in range(start + tl.program_id(1) * BLOCK_M, end, split * BLOCK_M):
        rows = m + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        inner = tl.zeros([BLOCK_M, R], dtype=tl.float32)
        for k in range(0, N, BLOCK_K):
            cols = k + tl.arange(0, BLOCK_K)
            act = tl.load(
                Act + rows.to(tl.int64)[:, None] * N + cols[None, :],
                row_mask[:, None] & (cols[None, :] < N),
                other=0,
            )
            weight = tl.load(a + ks[None, :] * N + cols[:, None], cols[:, None] < N)
            inner = tl.dot(act, weight, inner)
        tl.store(
            Inner + rows.to(tl.int64)[:, None] * R + ks[None, :],
            inner,
            row_mask[:, None],
        )


@triton.jit
def _token_rank(
    Inner,
    Src2Dst,
    Offsets,
    Mask,
    Out,
    R: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, R)
    lo = tl.load(Offsets)
    hi = tl.load(Offsets + E)
    total = tl.zeros([R], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        row = tl.load(Src2Dst + token * TOP_K + k)
        inside = (row >= lo) & (row < hi)
        total += tl.load(Inner + row.to(tl.int64) * R + ks, inside, other=0)
    total *= tl.load(Mask + token).to(tl.float32)
    tl.store(Out + token * R + ks, total)


def _split(rows: int, global_experts: int, block_m: int) -> int:
    """Programs per expert, sized for twice the mean rows of an expert."""
    return max(1, triton.cdiv(2 * rows, global_experts * block_m))


def routed_gate_up_delta(
    *, h, b, acc, dst2src, offsets, top_k, global_experts, nvfp4_scales=None
):
    """acc[r] += B[e] @ h[token(r)] for this rank's rows, in place. h is [T, 2R] FP32
    or BF16 (gate | up halves of x @ A13^T), b is [local_experts, 2N, R]; with
    nvfp4_scales the delta is divided by the row's outer scale first."""
    experts, two_n, rank = b.shape
    n = two_n // 2
    if b.stride(1) != rank or b.stride(2) != 1:
        raise ValueError("expert LoRA B rows must be contiguous")
    x_amax, w_amax, w_denominator = nvfp4_scales or (None, None, None)
    block_m, block_n = 16, 128
    split = _split(dst2src.numel(), global_experts, block_m)
    _gate_up_expand[(experts, triton.cdiv(n, block_n), split)](
        h,
        b,
        acc,
        x_amax,
        w_amax,
        w_denominator,
        dst2src,
        offsets,
        b.stride(0),
        n,
        rank,
        top_k,
        nvfp4_scales is not None,
        block_m,
        block_n,
        split,
        num_warps=4,
    )


def routed_down_inner(
    *, acc, gammas, a, offsets, gate_cap, up_cap, global_experts, nvfp4_scales=None
):
    """[rows, R] FP32 A2[e] @ h[r] for this rank's rows (others unset), where h is the
    capped swiglu times gamma rounded to BF16 as the routed SwiGLU kernels round it."""
    experts, rank, n = a.shape
    act = torch.empty((acc.shape[0], n), dtype=torch.bfloat16, device=acc.device)
    x_amax, w_amax, w_denominator = nvfp4_scales or (None, None, None)
    # Rows outside this rank's experts are never read: one program per local row, at most.
    programs = min(
        triton.next_power_of_2(max(acc.shape[0], 1)), 4 * _sm_count(acc.device)
    )
    _routed_act[(programs, triton.cdiv(n, 512))](
        acc,
        x_amax,
        w_amax,
        w_denominator,
        offsets,
        gammas,
        act,
        n,
        experts,
        nvfp4_scales is not None,
        gate_cap,
        up_cap,
        512,
        experts.bit_length(),
        programs,
    )
    inner = torch.empty((acc.shape[0], rank), dtype=torch.float32, device=acc.device)
    block_m = 16
    split = _split(acc.shape[0], global_experts, block_m)
    _down_shrink[(experts, split)](
        act, a, inner, offsets, n, rank, block_m, 256, split, num_warps=4
    )
    return inner


def routed_token_rank(*, inner, src2dst, offsets, mask, top_k):
    """[T, R] FP32 sum of each token's rows on this rank, times its adapter mask."""
    tokens = src2dst.numel() // top_k
    out = torch.empty(
        (tokens, inner.shape[1]), dtype=torch.float32, device=inner.device
    )
    if tokens:
        _token_rank[(tokens,)](
            inner,
            src2dst,
            offsets,
            mask,
            out,
            inner.shape[1],
            offsets.numel() - 1,
            top_k,
        )
    return out
