"""
Triton sum kernel for cross-CUDA-version bitwise parity.

torch.sum() uses different reduction algorithms across CUDA versions
(e.g. 13.0 vs 13.1), producing 1-ULP different results. This kernel
uses tl.sum, whose reduction order is fixed by the Triton compiler
and does not change across CUDA runtime versions.

Usage:
    from sglang.test.kernels.triton_sum import triton_sum
    score_sum = triton_sum(topk_scores, dim=-1, keepdim=True)
"""

import torch
import triton
import triton.language as tl

BLOCK_N = 128


def triton_sum(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Sum reduction with Triton-compiler-fixed accumulation order.

    Only supports 2D tensors with dim=-1 or dim=1.
    """
    assert x.dim() == 2, f"Only 2D tensors supported, got {x.dim()}D"
    if dim < 0:
        dim = x.dim() + dim
    assert dim == 1, f"Only dim=1 (or -1) supported, got dim={dim}"

    N, K = x.shape
    out = torch.empty(N, 1, dtype=x.dtype, device=x.device)

    grid = (triton.cdiv(N, BLOCK_N),)
    _triton_sum_kernel[grid](
        x,
        out,
        N,
        K,
        x.stride(0),
        x.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=triton.next_power_of_2(K),
    )

    if not keepdim:
        out = out.squeeze(-1)
    return out


@triton.jit
def _triton_sum_kernel(
    X,
    OUT,
    N,
    K,
    stride_xn,
    stride_xk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_mask = offs_n < N

    x = tl.load(
        X + offs_n[:, None] * stride_xn + offs_k[None, :] * stride_xk,
        mask=row_mask[:, None] & (offs_k[None, :] < K),
        other=0.0,
    )
    s = tl.sum(x, axis=1)
    tl.store(OUT + offs_n, s, mask=row_mask)
