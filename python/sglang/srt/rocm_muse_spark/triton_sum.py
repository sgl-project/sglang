"""
Triton sum kernel for cross-CUDA-version bitwise parity.

torch.sum() uses different reduction algorithms across CUDA versions
(e.g. 13.0 vs 13.1), producing 1-ULP different results. This kernel
uses tl.sum, whose reduction order is fixed by the Triton compiler
and does not change across CUDA runtime versions.

Usage:
    from sglang.srt.rocm_muse_spark.triton_sum import triton_sum
    score_sum = triton_sum(topk_scores, dim=-1, keepdim=True)
"""

import torch
import triton
import triton.language as tl


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

    grid = (N,)
    _triton_sum_kernel[grid](
        x, out, K, x.stride(0), x.stride(1), BLOCK_K=triton.next_power_of_2(K)
    )

    if not keepdim:
        out = out.squeeze(-1)
    return out


@triton.jit
def _triton_sum_kernel(
    X,
    OUT,
    K,
    stride_xn,
    stride_xk,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K

    x = tl.load(X + pid * stride_xn + offs_k * stride_xk, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(OUT + pid, s)
