"""MXFP8 quantization helpers for Inkling attention."""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

MXFP8_BLOCK_SIZE = 32


class MXFP8Tensor(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    ssm,
    ssk,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M

    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.exp2(scale_biased - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)

    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    tl.store(
        s_ptr + offs_m * ssm + pid_b * ssk,
        scale_biased.to(tl.uint8),
        mask=m_mask,
    )


def to_mxfp8(x: torch.Tensor) -> MXFP8Tensor:
    """Quantize the last dimension into E4M3 values plus E8M0 scale bytes."""
    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 quantization requires last dim divisible by {MXFP8_BLOCK_SIZE}, "
            f"got {x.shape[-1]}."
        )
    orig_shape = x.shape
    x2d = x.contiguous().view(-1, orig_shape[-1])
    M, K = x2d.shape
    xq = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
    scales = torch.empty((M, K // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=x.device)
    block_m = 64
    grid = (triton.cdiv(M, block_m), K // MXFP8_BLOCK_SIZE)
    _mxfp8_quant_kernel[grid](
        x2d,
        xq,
        scales,
        M,
        K,
        x2d.stride(0),
        x2d.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=block_m,
    )
    return MXFP8Tensor(
        data=xq.view(orig_shape),
        scale=scales.view(*orig_shape[:-1], K // MXFP8_BLOCK_SIZE),
    )
