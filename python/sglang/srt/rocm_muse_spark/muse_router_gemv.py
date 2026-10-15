"""Decode-sized router GEMM: logits[M, N] = fp32(hidden_bf16[M, K]) @ W_fp32[K, N].

The eager path is `hidden.float()` (a copy kernel) + hipBLASLt fp32 GEMM (split-K + reduce
kernel): 3 launches, ~23 us/layer at M=1 on MI355X. Here: split-K partial products with IEEE fp32
MFMA dots, then a deterministic reduce over the splits (2 launches). Used for M <= ROUTER_GEMV_MAX_M;
larger M keeps the torch path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ROUTER_GEMV_MAX_M = 64


@triton.jit
def _router_partial_kernel(
    X,
    W,
    P,
    M,
    N,
    K,
    stride_xm,
    stride_wk,
    stride_pk,
    stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k0 = pid_k * (K // SPLIT_K)
    x_ptrs = X + offs_m[:, None].to(tl.int64) * stride_xm + (k0 + offs_k)[None, :]
    w_ptrs = W + (k0 + offs_k)[:, None].to(tl.int64) * stride_wk + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // SPLIT_K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc, input_precision="ieee")
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    tl.store(
        P
        + pid_k * stride_pk
        + offs_m[:, None].to(tl.int64) * stride_pm
        + offs_n[None, :],
        acc,
        mask=m_mask[:, None],
    )


@triton.jit
def _router_reduce_kernel(
    P, OUT, M, N, stride_pk, stride_pm, SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(P + k * stride_pk + m * stride_pm + offs_n, mask=mask, other=0.0)
    tl.store(OUT + m * N + offs_n, acc, mask=mask)


def router_gemv_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[0] <= ROUTER_GEMV_MAX_M
        and x.stride(1) == 1
        and w.dtype == torch.float32
        and w.dim() == 2
        and w.stride(1) == 1
        and w.shape[0] == x.shape[1]
        and 32 <= w.shape[1] <= 512
        and (w.shape[1] & (w.shape[1] - 1)) == 0
        and w.shape[0] % 8192 == 0
    )


def router_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [M, K] bf16 (row stride free), w: [K, N] fp32 contiguous rows. Returns fp32 [M, N]."""
    M, K = x.shape
    N = w.shape[1]
    # full-width N tiles (contiguous 1 KB rows of W, vectorized loads) and a deep K split so
    # ~128 programs each stream one 64-row slab of W; tuned on MI355X for K=8192, N=256
    BLOCK_M, BLOCK_K = 16, 64
    BLOCK_N = min(N, 128)
    SPLIT_K = max(
        1, min(64, K // BLOCK_K)
    )  # MI355X sweep (K=8192, N=256): 12.4 us vs 17.7 us torch at M=1
    part = torch.empty((SPLIT_K, M, N), dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    if M == 0:
        return out
    grid = (N // BLOCK_N, SPLIT_K, triton.cdiv(M, BLOCK_M))
    _router_partial_kernel[grid](
        x,
        w,
        part,
        M,
        N,
        K,
        x.stride(0),
        w.stride(0),
        part.stride(0),
        part.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        num_warps=4,
    )
    _router_reduce_kernel[(M,)](
        part,
        out,
        M,
        N,
        part.stride(0),
        part.stride(1),
        SPLIT_K=SPLIT_K,
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=4,
    )
    return out
