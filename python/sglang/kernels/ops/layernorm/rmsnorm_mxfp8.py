"""RMSNorm with the MXFP8 epilogue of ``ops.quantization.mxfp8_epilogue``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_epilogue import mxfp8_epilogue


@triton.jit
def _sumsq_8warp_order(v):
    """``tl.sum(v * v, 0)`` of a 2048-element row in the order 8 warps reduce it
    (8 contiguous elements per thread, warp butterfly 16..1, then warps 4, 2, 1),
    spelled out as size-2 reductions: each is one add per output, so fewer warps
    produce the same bits."""
    # warp, lane bits 16..1, element bits 4, 2, 1
    sq = tl.reshape(v * v, (8, 2, 2, 2, 2, 2, 2, 2, 2))
    sq = tl.sum(sq, 7)  # (e0 + e2), (e1 + e3), (e4 + e6), (e5 + e7)
    sq = tl.sum(sq, 6)  # even / odd halves
    sq = tl.sum(sq, 6)  # thread total
    for _ in tl.static_range(5):  # lane butterfly 16, 8, 4, 2, 1
        sq = tl.sum(sq, 1)
    sq = tl.reshape(sq, (2, 2, 2))  # warps 4, 2, 1
    for _ in tl.static_range(3):
        sq = tl.sum(sq, 0)
    return sq


@triton.jit
def _rmsnorm_mxfp8_kernel(
    X,
    W,
    Y,
    Q,
    S,
    SX: tl.constexpr,
    M,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    ORDER_8WARP: tl.constexpr,
):
    row = tl.program_id(0)
    if row < M:
        h = tl.arange(0, BLOCK)
        v = tl.load(X + row * SX + h, h < K, 0).to(tl.float32)
        weight = tl.load(W + h, h < K, 0).to(tl.float32)
        if ORDER_8WARP:
            ss = _sumsq_8warp_order(v)
        else:
            ss = tl.sum(v * v, 0)
        inv = tl.rsqrt(ss / K + EPS)
        y = (v * inv * weight).to(tl.bfloat16)
        tl.store(Y + row * K + h, y, h < K)
        mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, 0, GROUPS)
    else:
        # Padding scale entries are disjoint from the live rows above.
        last = (tl.cdiv(M, 128) - 1) * 128
        local = (row - M) * 512 + tl.arange(0, 512)
        pad_row = last + ((local // 4) % 4) * 32 + ((local // 16) % 32)
        tl.store(S + last * GROUPS + local, 0, (local < GROUPS * 128) & (pad_row >= M))


def _num_warps(m: int, block: int) -> int:
    # 8 warps leave 3 of them fully masked at k=1280; 2 warps with the 8-warp
    # sum order spelled out give the same bits and pay once the grid is deep.
    return 2 if block == 2048 and m >= 1024 else 8


def rmsnorm_mxfp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm returning ``(y_bf16, y_q, y_sf)`` with the same MXFP8 epilogue."""
    m, k = x.shape
    assert m > 0 and k % 128 == 0
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(
        (k // 32) * triton.cdiv(m, 128) * 128, dtype=torch.uint8, device=x.device
    )
    block = triton.next_power_of_2(k)
    num_warps = _num_warps(m, block)
    _rmsnorm_mxfp8_kernel[(m + triton.cdiv((k // 32) * 128, 512),)](
        x,
        weight,
        y,
        q,
        s,
        SX=x.stride(0),
        M=m,
        K=k,
        EPS=eps,
        BLOCK=block,
        GROUPS=k // 32,
        ORDER_8WARP=num_warps != 8,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return y, q, s
