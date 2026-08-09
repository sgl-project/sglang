# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and indexed adaLN scale/shift prototype.

The operation matches the packed-token MiniMax H3 expression::

    normalized = torch.nn.functional.rms_norm(x, (H,), weight, eps)
    output = normalized * (1 + scale.index_select(0, indices)) \
        + shift.index_select(0, indices)

The Triton kernel keeps the normalized activation in registers and applies the
indexed modulation before its final store.  It is intentionally kept separate
from the H3 model while correctness and performance are evaluated.
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
    hidden_size,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size

    x = tl.load(
        x_ptr + row * stride_x_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    rstd = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    normalized = round_bf16_to_fp32(x * rstd * weight)

    index = tl.load(indices_ptr + row * stride_indices)
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)

    # Reproduce the BF16 storage boundaries in H3's existing indexed
    # modulation kernel: round(1 + scale), round(norm * that), then the final
    # store rounds round(product + shift).
    one_plus_scale = round_bf16_to_fp32(1.0 + scale)
    scaled = round_bf16_to_fp32(normalized * one_plus_scale)
    tl.store(
        output_ptr + row * stride_x_row + columns,
        scaled + shift,
        mask=mask,
    )


def can_use_fused_indexed_rmsnorm_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Return whether the prototype kernel supports these inputs."""
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
        and shift.is_contiguous()
        and scale.is_cuda
        and scale.device == x.device
        and scale.dtype is torch.bfloat16
        and scale.shape == shift.shape
        and scale.is_contiguous()
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
    if not can_use_fused_indexed_rmsnorm_adaln(
        x, weight, shift, scale, indices
    ):
        raise RuntimeError("unsupported input for fused indexed RMSNorm + adaLN")

    rows, hidden_size = x.shape
    output = torch.empty_like(x)
    block_n = triton.next_power_of_2(hidden_size)
    with torch.get_device_module().device(x.device):
        _indexed_rmsnorm_adaln_bf16_kernel[(rows,)](
            output,
            x,
            weight,
            shift,
            scale,
            indices,
            hidden_size,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            eps,
            BLOCK_N=block_n,
            num_warps=8,
        )
    return output


__all__ = [
    "can_use_fused_indexed_rmsnorm_adaln",
    "fused_indexed_rmsnorm_adaln",
]
