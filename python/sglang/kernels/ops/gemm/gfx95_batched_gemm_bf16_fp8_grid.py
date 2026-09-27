# SPDX-License-Identifier: Apache-2.0
"""Batched bf16 GEMM ``Y[g] = X[g] @ W[g]^T`` with the consumer's fp8-grid rounding in the
epilogue: the DeepSeek-V4 ``wo_a`` absorb GEMM at decode on gfx950. The main loop is aiter's
``_batched_gemm_bf16_kernel`` at a fixed 16 x 32 x 512 tile, so the bf16 result is bitwise
aiter's, and with ``fp8_grid=True`` the output is bitwise
``fake_quant_fp8_activation(batched_gemm_bf16(...))``. Requires ``R % 32 == 0``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_round

# 16 x 32 x 512 with 2 warps: the fastest N tile that holds whole 32-groups; nonkdim 16 is aiter's MFMA
_BLOCK_M, _BLOCK_N, _BLOCK_K = 16, 32, 512
_NUM_WARPS, _NUM_STAGES, _WAVES_PER_EU, _MFMA_NONKDIM = 2, 2, 2, 16
_CACHE_MODIFIER = ".cg"


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
        # the consumer's fake-quant rule on the bf16-rounded output: one ue8m0 group per 32 N elements
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


def batched_gemm_bf16_fp8_grid(
    x: torch.Tensor,
    w: torch.Tensor,
    fp8_grid: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    """``x`` [T, G, D] bf16 (any strides, contiguous last dim), ``w`` [G, R, D]
    bf16 contiguous -> [T, G * R] bf16, ``out[t, g*R:(g+1)*R] = x[t, g] @ w[g]^T``,
    on the fp8 grid when ``fp8_grid``. ``R % 32 == 0`` is required for the grid."""
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
