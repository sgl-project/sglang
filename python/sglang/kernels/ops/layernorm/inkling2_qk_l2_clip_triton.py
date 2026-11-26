# SPDX-License-Identifier: Apache-2.0
"""Single-launch Triton QK/relative L2 clipping for pre-Hopper devices.

:mod:`inkling2_qk_l2_clip_ref` matches the CuTe-DSL kernel bit for bit, but it
spends seven eager kernels per call (fp32 widen, optional scale, pow/sum, where,
sqrt, reciprocal, clamp, optional output scale). On A100 the clip pair (q and
the relative branch) became 9.5% of a decode step, so this module keeps the same
math in one kernel:

    xf      = x * scale                     (per-dim or per-head scale, fp32)
    coeff   = min(1, threshold / (||xf||_2 + eps))
    y       = xf * coeff
    y       = y.to(output_dtype).float() * output_scale   (per-row, then cast back)

One CTA owns one row and scans it twice (norm, then the scaled store) so the row
stays hot in cache. The math is unchanged from the reference and from the CuTe
kernel, including the round-trip cast around the per-row output scale.
"""

import torch
import triton
import triton.language as tl

_MAX_BLOCK_H = 1024


@triton.jit
def _qk_l2_clip_kernel(
    x,  # [rows, H], bf16 or fp32
    scale,  # [H] or [heads, H], only read when SCALE_RANK != 0
    output_scale,  # [rows / ROWS_PER_OSCALE] fp32, only read when HAS_OSCALE
    out,  # [rows, H], bf16 or fp32
    stride_row,
    threshold,
    eps,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS_PER_OSCALE: tl.constexpr,
    SCALE_RANK: tl.constexpr,
    HAS_OSCALE: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_base = x + row * stride_row
    head = row % HEADS

    # Pass 1: L2 norm of the (optionally scaled) row.
    sumsq = 0.0
    for off in tl.static_range(0, H, BLOCK_H):
        offs = off + tl.arange(0, BLOCK_H)
        mask = offs < H
        xv = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        if SCALE_RANK == 2:
            s = tl.load(scale + head * H + offs, mask=mask, other=0.0).to(tl.float32)
            xv = xv * s
        elif SCALE_RANK == 1:
            s = tl.load(scale + offs, mask=mask, other=0.0).to(tl.float32)
            xv = xv * s
        sumsq += tl.sum(xv * xv, 0)

    coeff = tl.minimum(threshold / (tl.sqrt(sumsq) + eps), 1.0)
    os = 1.0
    if HAS_OSCALE:
        os = tl.load(output_scale + row // ROWS_PER_OSCALE).to(tl.float32)

    # Pass 2: apply the coefficient (and the per-row output scale) and store.
    out_base = out + row * stride_row
    for off in tl.static_range(0, H, BLOCK_H):
        offs = off + tl.arange(0, BLOCK_H)
        mask = offs < H
        xv = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
        if SCALE_RANK == 2:
            s = tl.load(scale + head * H + offs, mask=mask, other=0.0).to(tl.float32)
            xv = xv * s
        elif SCALE_RANK == 1:
            s = tl.load(scale + offs, mask=mask, other=0.0).to(tl.float32)
            xv = xv * s
        yv = xv * coeff
        if OUT_FP32:
            if HAS_OSCALE:
                yv = yv * os
            tl.store(out_base + offs, yv, mask=mask)
        else:
            if HAS_OSCALE:
                # The reference rounds to the output dtype before the per-row
                # scale and again on the final store; keep the double rounding.
                yv = yv.to(tl.bfloat16).to(tl.float32) * os
            tl.store(out_base + offs, yv.to(tl.bfloat16), mask=mask)


def _fallback(x, threshold, eps, output_dtype, scale, output_scale):
    from sglang.kernels.ops.layernorm.inkling2_qk_l2_clip_ref import (
        qk_l2_clip_fwd_ref,
    )

    return qk_l2_clip_fwd_ref(
        x,
        threshold,
        eps,
        output_dtype=output_dtype,
        scale=scale,
        output_scale=output_scale,
    )


def qk_l2_clip_fwd_triton(
    x: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
    *,
    output_dtype: torch.dtype | None = None,
    scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ``qk_l2_clip_fwd_ref`` on pre-Hopper devices.

    Falls back to the torch reference for the shapes this kernel does not cover
    (no CUDA, unsupported dtype, non-contiguous input, an unexpected ``scale``
    rank/shape, or an ``output_scale`` whose groups do not divide the rows).
    """
    output_dtype = output_dtype or x.dtype
    supported_dtypes = (torch.bfloat16, torch.float32)
    if (
        not x.is_cuda
        or x.dtype not in supported_dtypes
        or output_dtype not in supported_dtypes
    ):
        return _fallback(x, threshold, eps, output_dtype, scale, output_scale)
    # The prologue hands us q/r as narrow slices of the fused qkvr projection, so
    # the rows are strided even though the inner dim is contiguous. One cheap
    # copy beats the seven eager kernels of the reference, so materialize instead
    # of falling back.
    if not x.is_contiguous():
        x = x.contiguous()

    width = x.shape[-1]
    if width == 0:
        return _fallback(x, threshold, eps, output_dtype, scale, output_scale)

    if scale is not None:
        # ``scale`` lives on the module and is cast to the model dtype (bf16)
        # when the model is moved, so accept any float dtype; the kernel widens
        # it to FP32 exactly as the reference's ``scale.float()`` does.
        if (
            not scale.is_cuda
            or scale.dtype not in supported_dtypes
            or scale.ndim not in (1, 2)
            or scale.shape[-1] != width
        ):
            return _fallback(x, threshold, eps, output_dtype, scale, output_scale)
        if not scale.is_contiguous():
            scale = scale.contiguous()

    rows = x.numel() // width
    if rows == 0:
        return _fallback(x, threshold, eps, output_dtype, scale, output_scale)

    rows_per_oscale = 0
    if output_scale is not None:
        if (
            not output_scale.is_cuda
            or output_scale.dtype not in supported_dtypes
            or output_scale.ndim != 1
            or output_scale.numel() == 0
            or rows % output_scale.numel() != 0
        ):
            return _fallback(x, threshold, eps, output_dtype, scale, output_scale)
        if not output_scale.is_contiguous():
            output_scale = output_scale.contiguous()
        rows_per_oscale = rows // output_scale.numel()

    heads = scale.shape[0] if scale is not None and scale.ndim == 2 else 1
    y = torch.empty((rows, width), dtype=output_dtype, device=x.device)
    block_h = min(_MAX_BLOCK_H, triton.next_power_of_2(width))
    _qk_l2_clip_kernel[(rows,)](
        x,
        scale if scale is not None else x,
        output_scale if output_scale is not None else x,
        y,
        width,
        float(threshold),
        float(eps),
        H=width,
        BLOCK_H=block_h,
        HEADS=heads,
        ROWS_PER_OSCALE=max(rows_per_oscale, 1),
        SCALE_RANK=0 if scale is None else scale.ndim,
        HAS_OSCALE=output_scale is not None,
        OUT_FP32=output_dtype == torch.float32,
        num_warps=min(8, max(1, block_h // 128)),
    )
    return y.view(x.shape)
