# SPDX-License-Identifier: Apache-2.0
"""Single-launch Triton RMSNorm for pre-Hopper devices.

:mod:`inkling2_rms_norm_ref` mirrors the CUTE ``RMSNormFwd`` math bit for bit,
but it does so with a fistful of eager kernels per call (fp32 widen, optional
pre-scale, square/mean/rsqrt, rstd scale, optional post-scale, cast). On A100 the
norm shows up as 21.9% of a decode step, so this module keeps the same math in a
single kernel:

    x'          = x * pre_weight            (when the layer is pre-affine)
    rstd        = rsqrt(mean(x'^2) + eps)
    y           = x' * rstd * weight        (weight when affine)

One CTA owns one row and scans it twice (statistics, then the scaled store) so
the row stays hot in cache and the whole norm becomes a single launch. The math
is unchanged from the reference and from the CUTE kernel.
"""

import torch
import triton
import triton.language as tl

_MAX_BLOCK_H = 1024


@triton.jit
def _rms_norm_kernel(
    x,  # [rows, H], bf16 or fp32
    weight,  # [H] post-scale, only read when HAS_WEIGHT
    pre_weight,  # [H] pre-scale, only read when HAS_PRE_WEIGHT
    out,  # [rows, H], bf16 or fp32
    stride_row,
    eps,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_PRE_WEIGHT: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_base = x + row * stride_row

    # Pass 1: sum of squares of the (optionally pre-scaled) row.
    sumsq = 0.0
    for off in tl.static_range(0, H, BLOCK_H):
        offs = off + tl.arange(0, BLOCK_H)
        mask = offs < H
        xv = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_PRE_WEIGHT:
            pw = tl.load(pre_weight + offs, mask=mask, other=0.0).to(tl.float32)
            xv = xv * pw
        sumsq += tl.sum(xv * xv, 0)

    rstd = tl.rsqrt(sumsq / H + eps)

    # Pass 2: apply rstd (and the post-scale) and store in the output dtype.
    out_base = out + row * stride_row
    for off in tl.static_range(0, H, BLOCK_H):
        offs = off + tl.arange(0, BLOCK_H)
        mask = offs < H
        xv = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_PRE_WEIGHT:
            pw = tl.load(pre_weight + offs, mask=mask, other=0.0).to(tl.float32)
            xv = xv * pw
        yv = xv * rstd
        if HAS_WEIGHT:
            w = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
            yv = yv * w
        if OUT_FP32:
            tl.store(out_base + offs, yv, mask=mask)
        else:
            tl.store(out_base + offs, yv.to(tl.bfloat16), mask=mask)


def _fallback(x, weight, eps, pre_weight, output_dtype, out):
    from sglang.kernels.ops.layernorm.inkling2_rms_norm_ref import rms_norm_ref

    return rms_norm_ref(
        x,
        weight,
        eps,
        pre_weight=pre_weight,
        output_dtype=output_dtype,
        out=out,
    )


def rms_norm_fused(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    *,
    pre_weight: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ``rms_norm_ref`` on pre-Hopper devices.

    Falls back to the torch reference for the shapes this kernel does not cover
    (no CUDA, unsupported dtype, non-contiguous input, or a ``weight`` /
    ``pre_weight`` that is not a contiguous per-channel vector).
    """
    output_dtype = output_dtype or x.dtype
    supported_dtypes = (torch.bfloat16, torch.float32)
    if (
        not x.is_cuda
        or x.dtype not in supported_dtypes
        or output_dtype not in supported_dtypes
    ):
        return _fallback(x, weight, eps, pre_weight, output_dtype, out)
    if not x.is_contiguous():
        x = x.contiguous()

    width = x.shape[-1]
    if width == 0:
        return _fallback(x, weight, eps, pre_weight, output_dtype, out)
    for param in (weight, pre_weight):
        if param is not None:
            if (
                not param.is_cuda
                or param.dtype not in supported_dtypes
                or not param.is_contiguous()
                or param.shape != (width,)
            ):
                return _fallback(x, weight, eps, pre_weight, output_dtype, out)

    rows = x.numel() // width
    if rows == 0:
        return _fallback(x, weight, eps, pre_weight, output_dtype, out)
    if out is not None and (
        out.shape != x.shape or out.dtype != output_dtype or not out.is_contiguous()
    ):
        return _fallback(x, weight, eps, pre_weight, output_dtype, out)

    y = (
        out
        if out is not None
        else torch.empty(x.shape, dtype=output_dtype, device=x.device)
    )
    block_h = min(_MAX_BLOCK_H, triton.next_power_of_2(width))
    _rms_norm_kernel[(rows,)](
        x,
        weight if weight is not None else x,
        pre_weight if pre_weight is not None else x,
        y,
        width,
        float(eps),
        H=width,
        BLOCK_H=block_h,
        HAS_WEIGHT=weight is not None,
        HAS_PRE_WEIGHT=pre_weight is not None,
        OUT_FP32=output_dtype == torch.float32,
        num_warps=min(8, max(1, block_h // 128)),
    )
    return y
