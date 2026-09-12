# SPDX-License-Identifier: Apache-2.0
"""Batched bf16 GEMM ``Y[g] = X[g] @ W[g]^T`` with the consumer's fp8-grid rounding in the epilogue
(the DeepSeek-V4 ``wo_a`` absorb GEMM on gfx950). Above ``_SPLIT_K_MAX_M`` rows aiter's tile loop,
bitwise aiter's; at or below, the same tile split eight ways along K and reduced in a fixed order."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_round

# 16 x 32 x 512, 2 warps: the fastest N tile holding whole 32-groups; nonkdim 16 is aiter's MFMA
_BLOCK_M, _BLOCK_N, _BLOCK_K = 16, 32, 512
_NUM_WARPS, _NUM_STAGES, _WAVES_PER_EU, _MFMA_NONKDIM = 2, 2, 2, 16
_CACHE_MODIFIER = ".cg"
# 8 splits of 256-wide K steps fill the machine up to this many rows
_SPLIT_K, _SPLIT_K_BLOCK_K, _SPLIT_K_MAX_M = 8, 256, 64


@triton.jit
def _batched_gemm_bf16_fp8_grid_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    FP8_GRID: tl.constexpr,
    cache_modifier: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
):
    """aiter ``_batched_gemm_bf16_kernel`` (GROUP_SIZE_M = 1, NUM_KSPLIT = 1, no
    bias) with the per-32 fp8 e4m3 quantize-dequantize of the bf16 result."""
    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    batch_id = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = tl.cast((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, tl.int64)
    offs_bn = tl.cast((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N, tl.int64)
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(num_k_iter):
        if EVEN_K:
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            a = tl.load(a_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)
    if FP8_GRID:
        # the consumer's fake-quant on the bf16-rounded output: one ue8m0 group per 32 N elements
        xg = tl.reshape(c.to(tl.float32), (BLOCK_SIZE_M * (BLOCK_SIZE_N // 32), 32))
        c = tl.reshape(fp8_grid_round(xg, eps), (BLOCK_SIZE_M, BLOCK_SIZE_N))
        c = c.to(c_ptr.type.element_ty)

    offs_cm = tl.cast(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), tl.int64)
    offs_cn = tl.cast(pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), tl.int64)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _batched_gemm_bf16_split_k_partial_kernel(
    a_ptr,
    b_ptr,
    part_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    cache_modifier: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
):
    """Grid (G, row tiles x N tiles, SPLIT_K): the fp32 partial of one K slice, stored as
    ``part[g, split, row_tile, m, n]``. ``K % (SPLIT_K * BLOCK_SIZE_K) == 0`` and
    ``N % BLOCK_SIZE_N == 0`` (unmasked N tiles)."""
    batch_id = tl.cast(tl.program_id(axis=0), tl.int64)
    pid = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    k_per_split = K // SPLIT_K
    offs_k = pid_k * k_per_split + tl.arange(0, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_m < M
    a_ptrs = a_ptr + (
        batch_id * tl.cast(stride_ab, tl.int64)
        + tl.cast(offs_m, tl.int64)[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * tl.cast(stride_bb, tl.int64)
        + offs_k[:, None] * stride_bk
        + tl.cast(offs_n, tl.int64)[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(k_per_split // BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    tile = ((batch_id * SPLIT_K + pid_k) * num_pid_m + pid_m) * BLOCK_SIZE_M
    tl.store(
        part_ptr
        + (tile + tl.arange(0, BLOCK_SIZE_M))[:, None] * N
        + tl.cast(offs_n, tl.int64)[None, :],
        acc,
    )


@triton.jit
def _batched_gemm_split_k_reduce_kernel(
    part_ptr,
    c_ptr,
    M,
    N,
    stride_cb,
    stride_cm,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FP8_GRID: tl.constexpr,
):
    """Grid (G, row tiles x N tiles): sums the partials of one output tile in split order,
    rounds to bf16 and, with FP8_GRID, onto the consumer's fp8 grid."""
    batch_id = tl.cast(tl.program_id(axis=0), tl.int64)
    pid = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.cast(pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), tl.int64)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        tile = ((batch_id * SPLIT_K + s) * num_pid_m + pid_m) * BLOCK_SIZE_M
        acc += tl.load(
            part_ptr
            + (tile + tl.arange(0, BLOCK_SIZE_M))[:, None] * N
            + offs_n[None, :]
        )
    c = acc.to(c_ptr.type.element_ty)
    if FP8_GRID:
        xg = tl.reshape(c.to(tl.float32), (BLOCK_SIZE_M * (BLOCK_SIZE_N // 32), 32))
        c = tl.reshape(fp8_grid_round(xg, eps), (BLOCK_SIZE_M, BLOCK_SIZE_N))
        c = c.to(c_ptr.type.element_ty)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * tl.cast(offs_m, tl.int64)[:, None]
        + offs_n[None, :]
    )
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _split_k_applies(T: int, D: int, R: int) -> bool:
    # the partial kernel stores whole N tiles with pitch R, so R must be a tile multiple
    return (
        0 < T <= _SPLIT_K_MAX_M
        and D % (_SPLIT_K * _SPLIT_K_BLOCK_K) == 0
        and R % _BLOCK_N == 0
    )


def _batched_gemm_split_k(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, fp8_grid: bool, eps: float
) -> None:
    T, G, D = x.shape
    R = w.shape[1]
    tiles_m, tiles_n = triton.cdiv(T, _BLOCK_M), triton.cdiv(R, _BLOCK_N)
    partials = torch.empty(
        (G, _SPLIT_K, tiles_m * _BLOCK_M, R), dtype=torch.float32, device=x.device
    )
    _batched_gemm_bf16_split_k_partial_kernel[(G, tiles_m * tiles_n, _SPLIT_K)](
        x,
        w,
        partials,
        T,
        R,
        D,
        x.stride(1),
        x.stride(0),
        x.stride(2),
        w.stride(0),
        w.stride(2),
        w.stride(1),
        BLOCK_SIZE_M=_BLOCK_M,
        BLOCK_SIZE_N=_BLOCK_N,
        BLOCK_SIZE_K=_SPLIT_K_BLOCK_K,
        SPLIT_K=_SPLIT_K,
        cache_modifier=_CACHE_MODIFIER,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        waves_per_eu=_WAVES_PER_EU,
        matrix_instr_nonkdim=_MFMA_NONKDIM,
    )
    _batched_gemm_split_k_reduce_kernel[(G, tiles_m * tiles_n)](
        partials,
        out,
        T,
        R,
        R,
        G * R,
        eps,
        BLOCK_SIZE_M=_BLOCK_M,
        BLOCK_SIZE_N=_BLOCK_N,
        SPLIT_K=_SPLIT_K,
        FP8_GRID=fp8_grid,
        num_warps=_NUM_WARPS,
    )


def batched_gemm_bf16_fp8_grid(
    x: torch.Tensor,
    w: torch.Tensor,
    fp8_grid: bool = True,
    eps: float = 1e-10,
    split_k: Optional[bool] = None,
) -> torch.Tensor:
    """``x`` [T, G, D] bf16, ``w`` [G, R, D] bf16 -> [T, G * R] bf16 with ``out[t, g*R:(g+1)*R] =
    x[t, g] @ w[g]^T``, on the fp8 grid when ``fp8_grid``; ``split_k`` forces a regime (tests)."""
    assert x.dim() == 3 and w.dim() == 3, (x.shape, w.shape)
    T, G, D = x.shape
    assert w.shape[0] == G and w.shape[2] == D, (x.shape, w.shape)
    R = w.shape[1]
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.stride(2) == 1 and w.is_contiguous()
    assert not fp8_grid or R % 32 == 0, R
    out = torch.empty((T, G * R), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    if split_k is None:
        split_k = _split_k_applies(T, D, R)
    if split_k:
        assert _split_k_applies(T, D, R), (T, D, R)
        _batched_gemm_split_k(x, w, out, fp8_grid, eps)
        return out
    grid = (G, triton.cdiv(T, _BLOCK_M) * triton.cdiv(R, _BLOCK_N))
    _batched_gemm_bf16_fp8_grid_kernel[grid](
        x,
        w,
        out,
        T,
        R,
        D,
        x.stride(1),
        x.stride(0),
        x.stride(2),
        w.stride(0),
        w.stride(2),
        w.stride(1),
        R,
        G * R,
        1,
        eps,
        BLOCK_SIZE_M=_BLOCK_M,
        BLOCK_SIZE_N=_BLOCK_N,
        BLOCK_SIZE_K=_BLOCK_K,
        EVEN_K=(D % _BLOCK_K == 0),
        FP8_GRID=fp8_grid,
        cache_modifier=_CACHE_MODIFIER,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        waves_per_eu=_WAVES_PER_EU,
        matrix_instr_nonkdim=_MFMA_NONKDIM,
    )
    return out
