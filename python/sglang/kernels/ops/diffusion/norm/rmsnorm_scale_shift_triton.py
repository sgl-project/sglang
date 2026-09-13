# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm + adaLN scale/shift with per-token modulation (Triton).

Computes, for each token row,

    y = rmsnorm(x) * (1 + scale) + shift

where ``x`` is ``[B, S, D]`` (bf16/fp16/fp32), the RMSNorm weight is ``[D]``, and
the per-token ``scale``/``shift`` are ``[B, S, D]`` views (cast to fp32 in
kernel). ``scale``/``shift`` are typically non-contiguous ``chunk`` views of the
``[B, S, 6D]`` modulation tensor, so the kernel takes their row stride and reads
them strided instead of materializing contiguous copies. The whole chain runs in
one kernel in fp32 with a single rounding to the output dtype, replacing the
eager LingBot chain ``(norm(x) * (1 + scale) + shift).to(dtype)``.

Unlike the bit-exact ``rmsnorm_scale_shift_bitexact`` (ERNIE), this kernel does
*not* reproduce PyTorch's parallel variance reduction order bit-for-bit, so it
is intended for the request-gated (``quality="extra-high"``/``"high"``) fusion
path, matching the existing quality-gated LingBot RMSNorm fusion.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _rmsnorm_scale_shift_kernel(
    y_ptr,
    x_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    mod_row_stride,
    SEQ,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    seq_blk_id = tl.program_id(0)
    seq_id = seq_blk_id * BLOCK_SIZE_SEQ

    seq_offset = seq_id + tl.arange(0, BLOCK_SIZE_SEQ)[:, None]
    s_mask = seq_offset < SEQ
    d_offset = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
    d_mask = d_offset < DIM
    mask = s_mask & d_mask

    xy_ptr = seq_offset * DIM + d_offset
    mod_ptr = seq_offset * mod_row_stride + d_offset

    x = tl.load(x_ptr + xy_ptr, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1, keep_dims=True) / DIM
    rstd = tl.math.rsqrt(mean_square + EPS)
    w = tl.load(w_ptr + d_offset, mask=d_mask).to(tl.float32)
    scale = tl.load(scale_ptr + mod_ptr, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + mod_ptr, mask=mask, other=0.0).to(tl.float32)

    y = (x * rstd * w) * (1.0 + scale) + shift
    tl.store(y_ptr + xy_ptr, y, mask=mask)


@register_custom_op(op_name="rmsnorm_scale_shift_per_token_cuda", out_shape="x")
def _rmsnorm_scale_shift_per_token_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    shape = x.shape
    out = torch.empty_like(x)
    x2 = x.reshape(-1, shape[-1])
    out2 = out.reshape(-1, shape[-1])
    scale2 = scale.reshape(-1, shape[-1])
    shift2 = shift.reshape(-1, shape[-1])
    S, D = x2.shape

    # scale/shift share the row stride of their parent modulation tensor
    # (contiguous rows, possibly strided across the 6*D row).
    assert scale2.stride(0) == shift2.stride(0)
    mod_row_stride = scale2.stride(0)

    block_size_seq = min(16, triton.next_power_of_2(max(1, S // 512)))
    grid = (triton.cdiv(S, block_size_seq),)
    with torch.get_device_module().device(x.device):
        _rmsnorm_scale_shift_kernel[grid](
            out2,
            x2,
            weight,
            scale2,
            shift2,
            mod_row_stride,
            S,
            DIM=D,
            EPS=eps,
            BLOCK_SIZE_DIM=triton.next_power_of_2(D),
            BLOCK_SIZE_SEQ=block_size_seq,
        )
    return out


def can_use_rmsnorm_scale_shift_per_token(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    return (
        x.is_cuda
        and x.dim() == 3
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.is_cuda
        and weight.dim() == 1
        and weight.numel() == x.shape[-1]
        and scale.shape == x.shape
        and shift.shape == x.shape
        and scale.dtype == shift.dtype
        and scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
        # rows must be contiguous (stride(1) == 1); row stride may differ
        # (non-contiguous chunk views of the [B, S, 6D] modulation tensor).
        and scale.stride(0) == shift.stride(0)
        and scale.stride(2) == 1
        and shift.stride(2) == 1
        and x.numel() > 0
    )


def rmsnorm_scale_shift_per_token(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused ``rmsnorm(x) * (1 + scale) + shift`` with per-token scale/shift."""
    return _rmsnorm_scale_shift_per_token_cuda(x, weight, scale, shift, eps)


__all__ = [
    "can_use_rmsnorm_scale_shift_per_token",
    "rmsnorm_scale_shift_per_token",
]
