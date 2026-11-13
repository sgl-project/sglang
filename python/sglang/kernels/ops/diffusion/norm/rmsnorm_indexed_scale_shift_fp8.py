# SPDX-License-Identifier: Apache-2.0
"""Indexed RMSNorm + modulation with native per-token FP8 output."""

import torch
import triton
import triton.language as tl

from .rmsnorm_scale_shift_triton import (
    _rmsnorm_indexed_scale_shift_row,
    can_use_rmsnorm_indexed_scale_shift,
)


def can_use_rmsnorm_indexed_scale_shift_fp8(x, weight, shift, scale, indices) -> bool:
    """SM90 BF16 width-5376 rows matching the native norm + per-token quantizer."""
    return (
        can_use_rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices)
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) == (9, 0)
        and weight.device == shift.device == scale.device == indices.device == x.device
        and weight.is_contiguous()
        and x.dtype == weight.dtype == shift.dtype == scale.dtype == torch.bfloat16
        and indices.dtype
        in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        and shift.shape[0] > 0
        and x.shape[1] == 5376
        and x.shape[0] >= 4096
    )


@triton.jit
def _rmsnorm_indexed_scale_shift_fp8_kernel(
    X,
    W,
    SHIFT,
    SCALE,
    IDX,
    Y,
    Q,
    S,
    MOD_STRIDE: tl.constexpr,
    IDX_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    KEEP_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < DIM
    mod = tl.load(IDX + row * IDX_STRIDE).to(tl.int64) * MOD_STRIDE + cols
    value = (
        _rmsnorm_indexed_scale_shift_row(X, W, SCALE, SHIFT, mod, row, cols, DIM, EPS)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    value = tl.where(mask, value, 0.0)
    if KEEP_BF16:
        tl.store(Y + row * DIM + cols, value, mask)
    scale = tl.max(tl.abs(value), 0) * (1.0 / 448.0)
    inv = tl.where(
        scale == 0.0,
        0.0,
        tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [scale],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        ),
    )
    q = tl.minimum(tl.maximum(value * inv, -448.0), 448.0)
    # Eight-element store groups preserve the BF16 norm's reduction tree on
    # the pinned Triton compiler; unrestricted FP8 stores select groups of 16.
    cols = tl.max_contiguous(cols, 8)
    tl.store(Q + row * DIM + cols, q.to(tl.float8e4nv), cols < DIM)
    tl.store(S + row, scale)


def rmsnorm_indexed_scale_shift_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
    *,
    keep_bf16: bool = False,
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
    """Indexed RMSNorm + modulation + per-token E4M3 quantization on SM90.

    Preserves the width-5376 BF16 norm output, then uses amax/448 scales
    (zero for zero rows) and approximate-reciprocal multiplication, matching
    sglang_per_token_quant_fp8. Inputs must be BF16, contiguous rows/weight,
    with at least 4096 rows. Modulation tables and integer indices may be strided;
    indices must lie in [0, modulation rows). The caller owns the quality gate.

    Returns (optional_bf16, (fp8_rows, fp32_scales[M,1])). keep_bf16 retains
    native rows for additional consumers without another norm launch.
    """
    if not can_use_rmsnorm_indexed_scale_shift_fp8(x, weight, shift, scale, indices):
        raise ValueError(
            "unsupported indexed RMSNorm FP8 inputs (weight must be contiguous)"
        )
    rows, dim = x.shape
    y = torch.empty_like(x) if keep_bf16 else None
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        _rmsnorm_indexed_scale_shift_fp8_kernel[(rows,)](
            x,
            weight,
            shift,
            scale,
            indices,
            y,
            q,
            scales,
            scale.stride(0),
            indices.stride(0),
            dim,
            eps,
            triton.next_power_of_2(dim),
            keep_bf16,
            num_warps=8,
        )
    return y, (q, scales)
