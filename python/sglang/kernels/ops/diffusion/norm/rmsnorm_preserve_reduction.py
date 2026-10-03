# SPDX-License-Identifier: Apache-2.0
"""Fuse RMSNorm pointwise work while retaining the native FP32 mean reduction.

Matches ``weight * (x.float() * rsqrt(mean(x.float()**2) + eps)).to(x.dtype)``
for contiguous FP16/BF16 inputs. The FP32 square buffer has the original shape,
so aten selects the same reduction as the eager chain. Verified at head width
128, including 131072 rows; callers verify their first dispatch before reuse.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _square_fp32_kernel(x_ptr, square_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x_ptr + index, index < N, 0).to(tl.float32)
    tl.store(square_ptr + index, value * value, index < N)


@triton.jit
def _rmsnorm_finish_kernel(
    x_ptr,
    variance_ptr,
    weight_ptr,
    out_ptr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    value = tl.load(x_ptr + index, mask, 0).to(tl.float32)
    variance = tl.load(variance_ptr + index // DIM, mask, 0)
    weight = tl.load(weight_ptr + index % DIM, mask, 0).to(tl.float32)
    # eager rounds the normalized activation before multiplying the weight
    normalized = (value * tl.rsqrt(variance + EPS)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + index, normalized.to(tl.float32) * weight, mask)


def can_use_rmsnorm_preserve_reduction(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim >= 2
        and x.numel() > 0
        and x.is_contiguous()
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
    )


def _fake_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="rmsnorm_preserve_reduction",
    mutates_args=[],
    fake_impl=_fake_rmsnorm,
)
def rmsnorm_preserve_reduction(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Preserve aten's mean and cast-before-weight semantics without residuals."""
    assert can_use_rmsnorm_preserve_reduction(x, weight)
    squares = torch.empty_like(x, dtype=torch.float32)
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _square_fp32_kernel[(triton.cdiv(x.numel(), 1024),)](
            x, squares, x.numel(), 1024
        )
        variance = squares.mean(dim=-1, keepdim=True)
        _rmsnorm_finish_kernel[(triton.cdiv(x.numel(), 512),)](
            x,
            variance,
            weight,
            out,
            x.numel(),
            x.shape[-1],
            eps,
            512,
            enable_fp_fusion=False,
        )
    return out
