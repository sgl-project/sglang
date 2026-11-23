# SPDX-License-Identifier: Apache-2.0
"""Per-token FP8 activation quant feeding the gfx1250 FP8 GEMM tile (``gemm.fp8_gfx1250``)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_quant_fp8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    K,
    stride_xm,
    stride_qm,
    FP8_MAX: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(x)) / FP8_MAX
    # An all-zero row would divide by zero; its quantized values are zero
    # either way, so any positive scale reproduces it exactly.
    scale = tl.where(scale > 0.0, scale, 1.0)
    q = tl.clamp(x / scale, -FP8_MAX, FP8_MAX).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)
    tl.store(scale_ptr + row, scale)


def per_token_quant_fp8_gfx1250(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` activations to ``float8_e4m3fn`` with one scale per row.

    Returns ``(q, scale)`` where ``scale`` is ``[M]`` float32 and
    ``q * scale[:, None]`` reconstructs ``x``.
    """
    assert x.dim() == 2
    x = x.contiguous()
    M, K = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    block_k = triton.next_power_of_2(K)
    _per_token_quant_fp8_kernel[(M,)](
        x,
        q,
        scale,
        K,
        x.stride(0),
        q.stride(0),
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return q, scale
