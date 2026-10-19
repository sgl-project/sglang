from __future__ import annotations

from typing import TYPE_CHECKING

import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported, is_npu

if TYPE_CHECKING:
    import torch

_is_npu = is_npu()

# gfx950: (n, k) -> the largest num_tokens _k128_gemm_kernel takes, measured
# against F.linear on the KDA f_b projection at EAGLE verify.
_K3_GFX95_K128_GEMM_MAX_TOKENS = {(1536, 128): 64}


@triton.jit
def _k128_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    stride_xm,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)
    mask_m = offs_m[:, None] < M
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :], mask=mask_m)
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
    acc = tl.dot(x, tl.trans(w))
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m,
    )


def _k128_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    import torch

    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m, block_n = 16, 32
    _k128_gemm_kernel[(n // block_n, triton.cdiv(m, block_m))](
        x,
        w,
        out,
        m,
        x.stride(0),
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=2,
    )
    return out


# (n, k) -> the largest num_tokens where the tiny GEMM still beats cuBLAS.
# Doubles as the compile-time max_m: one kernel is built per m up to it.
_K3_TINY_GEMM_MAX_TOKENS = {
    (144, 7168): 16,
    (896, 7168): 8,
    (1536, 128): 12,
}


def kimi_k3_tiny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    import torch

    from .tiny_gemm import tiny_gemm_bf16

    m, k = x.shape
    n, _ = w.shape
    if (
        0 < m <= _K3_GFX95_K128_GEMM_MAX_TOKENS.get((n, k), 0)
        and is_gfx95_supported()
        and x.dtype == w.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and w.is_contiguous()
    ):
        return _k128_gemm(x, w)
    max_num_tokens = _K3_TINY_GEMM_MAX_TOKENS.get((n, k))
    if not _is_npu and max_num_tokens is not None and 0 < m <= max_num_tokens:
        return tiny_gemm_bf16(x, w, max_m=max_num_tokens)
    return torch.nn.functional.linear(x, w)
