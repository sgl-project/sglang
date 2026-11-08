"""Prefill-width router logits GEMM: the fp32 weight split into three bf16 terms in-kernel.

The eager path is ``hidden.float() @ W`` on hipBLASLt (XF32 under
``torch.set_float32_matmul_precision("high")``). Here each W tile is split as
W = hi + mid + lo (round-to-nearest bf16 residues: 3 x 8 significant bits cover fp32's 24,
so the split reconstructs W bitwise away from the denormal range) and the three bf16 x bf16
dots accumulate in fp32 at the MFMA BF16 rate. The activation is already bf16 and bf16
products are exact in fp32, so only the K summation order differs from an fp32 GEMM.

From ``ROUTER_BF16X3_MIN_M`` rows; smaller widths stay on the GEMV and the tuned table.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ROUTER_BF16X3_MIN_M = 257


@triton.jit(do_not_specialize=["M"])
def _router_bf16x3_partial(
    X,
    W,
    P,
    M,
    stride_xm,
    stride_wk,
    K: tl.constexpr,
    N: tl.constexpr,
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
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        hi = w.to(tl.bfloat16)
        r = w - hi.to(tl.float32)
        mid = r.to(tl.bfloat16)
        lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(x, hi, acc)
        acc = tl.dot(x, mid, acc)
        acc = tl.dot(x, lo, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    tl.store(
        P + pid_k * (M * N) + offs_m[:, None].to(tl.int64) * N + offs_n[None, :],
        acc,
        mask=m_mask[:, None],
    )


@triton.jit(do_not_specialize=["MN"])
def _router_bf16x3_reduce(P, OUT, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(P + k * MN + i, mask=mask, other=0.0)
    tl.store(OUT + i, acc, mask=mask)


def router_gemm_bf16x3_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[0] >= ROUTER_BF16X3_MIN_M
        and x.stride(1) == 1
        and w.dtype == torch.float32
        and w.dim() == 2
        and w.stride(1) == 1
        and w.shape[0] == x.shape[1]
        and w.shape[0] % 512 == 0
        and w.shape[1] % 128 == 0
    )


def router_gemm_bf16x3(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [M, K] bf16 (any row stride), w: [K, N] fp32 with contiguous rows. Returns fp32 [M, N]."""
    M, K = x.shape
    N = w.shape[1]
    # Split K by 8: N is only two 128-wide tiles, so the M x N tiles alone leave most CUs idle.
    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K = 64, 128, 64, 8
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    if M == 0:
        return out
    part = torch.empty((SPLIT_K, M, N), dtype=torch.float32, device=x.device)
    grid = (N // BLOCK_N, SPLIT_K, triton.cdiv(M, BLOCK_M))
    _router_bf16x3_partial[grid](
        x,
        w,
        part,
        M,
        x.stride(0),
        w.stride(0),
        K=K,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        num_warps=4,
    )
    _router_bf16x3_reduce[(triton.cdiv(M * N, 1024),)](
        part, out, M * N, SPLIT_K, 1024, num_warps=4
    )
    return out
