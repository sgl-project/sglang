# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and indexed adaLN scale/shift for MiniMax H3.

The operation matches the packed-token MiniMax H3 expression::

    normalized = torch.nn.functional.rms_norm(x, (H,), weight, eps)
    output = normalized * (1 + scale.index_select(0, indices)) \
        + shift.index_select(0, indices)

The Triton kernel uses a two-pass tiled RMS reduction, then reloads each tile
to apply normalization and indexed modulation before its final store. Its
RMSNorm reduction is not bit-exact with every eager backend, so H3 enables it
only through the request-scoped ``quality="high"`` fusion gate.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.triton.numerics import round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op

_MAX_HIDDEN_SIZE = 8192


@triton.jit
def _indexed_rmsnorm_adaln_bf16_kernel(
    output_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    eps,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x_row
    FULL: tl.constexpr = (HIDDEN // BLOCK_N) * BLOCK_N
    REM: tl.constexpr = HIDDEN - FULL

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in tl.static_range(0, FULL, BLOCK_N):
        columns = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + columns).to(tl.float32)
        acc += x * x
    if REM > 0:
        columns = FULL + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + columns, mask=columns < HIDDEN, other=0.0).to(tl.float32)
        acc += x * x
    rstd = tl.rsqrt(tl.sum(acc, axis=0) / HIDDEN + eps)

    index = tl.load(indices_ptr + row * stride_indices)
    shift_row = shift_ptr + index * stride_shift_row
    scale_row = scale_ptr + index * stride_scale_row
    out_row = output_ptr + row * stride_x_row

    for start in tl.static_range(0, FULL, BLOCK_N):
        columns = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + columns).to(tl.float32)
        weight = tl.load(weight_ptr + columns).to(tl.float32)
        normalized = round_bf16_to_fp32(x * rstd * weight)
        shift = tl.load(shift_row + columns).to(tl.float32)
        scale = tl.load(scale_row + columns).to(tl.float32)
        one_plus_scale = round_bf16_to_fp32(1.0 + scale)
        scaled = round_bf16_to_fp32(normalized * one_plus_scale)
        tl.store(out_row + columns, scaled + shift)
    if REM > 0:
        columns = FULL + tl.arange(0, BLOCK_N)
        mask = columns < HIDDEN
        x = tl.load(x_row + columns, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        normalized = round_bf16_to_fp32(x * rstd * weight)
        shift = tl.load(shift_row + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_row + columns, mask=mask, other=0.0).to(tl.float32)
        one_plus_scale = round_bf16_to_fp32(1.0 + scale)
        scaled = round_bf16_to_fp32(normalized * one_plus_scale)
        tl.store(out_row + columns, scaled + shift, mask=mask)


def can_use_fused_indexed_rmsnorm_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Return whether the fused kernel supports these inputs."""
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 2
        and x.shape[0] > 0
        and 0 < x.shape[1] <= _MAX_HIDDEN_SIZE
        and x.is_contiguous()
        and weight.is_cuda
        and weight.device == x.device
        and weight.dtype is torch.bfloat16
        and weight.shape == (x.shape[1],)
        and weight.is_contiguous()
        and shift.is_cuda
        and shift.device == x.device
        and shift.dtype is torch.bfloat16
        and shift.dim() == 2
        and shift.shape[0] > 0
        and shift.shape[1] == x.shape[1]
        and shift.stride(1) == 1
        and scale.is_cuda
        and scale.device == x.device
        and scale.dtype is torch.bfloat16
        and scale.shape == shift.shape
        and scale.stride(1) == 1
        and indices.is_cuda
        and indices.device == x.device
        and indices.dtype in (torch.int32, torch.int64)
        and indices.shape == (x.shape[0],)
        and indices.is_contiguous()
    )


def _fake_indexed_rmsnorm_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="triton_fused_indexed_rmsnorm_adaln",
    mutates_args=[],
    fake_impl=_fake_indexed_rmsnorm_adaln,
)
def fused_indexed_rmsnorm_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse BF16 RMSNorm with packed-token indexed adaLN scale and shift."""
    if not can_use_fused_indexed_rmsnorm_adaln(x, weight, shift, scale, indices):
        raise RuntimeError("unsupported input for fused indexed RMSNorm + adaLN")

    rows, hidden_size = x.shape
    output = torch.empty_like(x)
    with torch.get_device_module().device(x.device):
        _indexed_rmsnorm_adaln_bf16_kernel[(rows,)](
            output,
            x,
            weight,
            shift,
            scale,
            indices,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            eps,
            HIDDEN=hidden_size,
            BLOCK_N=1024,
            num_warps=4,
        )
    return output


__all__ = [
    "can_use_fused_indexed_rmsnorm_adaln",
    "fused_indexed_rmsnorm_adaln",
]
