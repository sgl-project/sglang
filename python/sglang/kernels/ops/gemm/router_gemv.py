# SPDX-License-Identifier: Apache-2.0
"""Skinny router-logits GEMV: [M, K] bf16 x [N, K] bf16 -> [M, N] fp32.

MoE router gates are tiny (N ~ 128 experts, K ~ hidden) but hipblaslt's
solutions for M<=8, N=128 run at ~0.1 TB/s on gfx950 (~12us for 1.6MB); this
split-K reduction reads the weight once at near-roofline instead.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _router_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_ptr + pid_m * stride_xm + offs_k).to(tl.float32)
        wv = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]).to(
            tl.float32
        )
        acc += tl.sum(wv * xv[None, :], axis=1)
    if SPLIT_K == 1:
        tl.store(out_ptr + pid_m * stride_om + offs_n, acc)
    else:
        tl.atomic_add(out_ptr + pid_m * stride_om + offs_n, acc)


def router_gemv_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    m, k = x.shape
    n = w.shape[0]
    return (
        x.stride(1) == 1
        and w.stride(1) == 1
        and n % 16 == 0
        and k % 512 == 0
        and m <= 64
    )


def router_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Router logits in fp32; caller guards with router_gemv_supported()."""
    m, k = x.shape
    n = w.shape[0]
    out = torch.zeros(m, n, device=x.device, dtype=torch.float32)
    BLOCK_N, BLOCK_K, SPLIT_K = 16, 512, 8
    grid = (n // BLOCK_N, m, SPLIT_K)
    _router_gemv_kernel[grid](
        x,
        w,
        out,
        k,
        x.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        SPLIT_K=SPLIT_K,
        num_warps=2,
    )
    return out
