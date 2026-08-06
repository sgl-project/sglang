# SPDX-License-Identifier: Apache-2.0
"""Channels-last-3d Wan VAE RMSNorm(+SiLU) Triton kernel.

Fuses the Wan VAE ``WanRMS_norm -> SiLU`` chain
(``SiLU(F.normalize(x, dim=1) * scale * gamma + bias)`` on channel-first 5D
activations) into one kernel for ``channels_last_3d`` tensors: one program
reduces one (b, t, h, w) pixel's channel row with fully coalesced loads.

Numerics contract: fp32 channel-norm statistics, every step materialized at
the same dtype boundary as eager ``WanRMS_norm.forward`` (including the
aten promotion to fp32 at ``* gamma`` for half-precision x with fp32 affine
params -- the autocast case), SiLU in fp32. Bitwise equality with aten is
still not guaranteed (different reduction and SiLU paths), so callers must
keep this behind an opt-in gate. ``wan_rmsnorm_silu`` returns ``None`` for
unsupported inputs (see ``can_use_wan_rmsnorm_silu``); callers must fall
back to their reference path.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.utils.custom_op import register_custom_op

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_CHANNELS = 1024


@triton.jit
def _wan_rmsnorm_silu_kernel(
    x_ptr,
    gamma_ptr,
    bias_ptr,
    out_ptr,
    channels: tl.constexpr,
    t_size,
    h_size,
    w_size,
    x_stride_b,
    x_stride_c,
    x_stride_t,
    x_stride_h,
    x_stride_w,
    out_stride_b,
    out_stride_c,
    out_stride_t,
    out_stride_h,
    out_stride_w,
    rms_scale,
    eps,
    has_bias: tl.constexpr,
    block_c: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_c)
    mask = offsets < channels

    w = row % w_size
    tmp = row // w_size
    h = tmp % h_size
    tmp = tmp // h_size
    t = tmp % t_size
    b = tmp // t_size

    x_base = b * x_stride_b + t * x_stride_t + h * x_stride_h + w * x_stride_w
    out_base = b * out_stride_b + t * out_stride_t + h * out_stride_h + w * out_stride_w

    x = tl.load(x_ptr + x_base + offsets * x_stride_c, mask=mask, other=0.0).to(
        tl.float32
    )
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    inv_norm = 1.0 / tl.maximum(norm, eps)

    # Eager op boundaries: normalize/*scale in x.dtype; *gamma/+bias in the
    # promoted output dtype; SiLU in fp32, stored in the output dtype.
    y = (x * inv_norm).to(x_ptr.dtype.element_ty)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    y = (y * rms_scale).to(x_ptr.dtype.element_ty)
    y = (y.to(tl.float32) * gamma.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if has_bias:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        y = (y.to(tl.float32) + bias.to(tl.float32)).to(out_ptr.dtype.element_ty)
    y = y.to(tl.float32)
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + out_base + offsets * out_stride_c, y, mask=mask)


def _fake_wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    rms_scale: float,
    eps: float,
    has_bias: bool,
) -> torch.Tensor:
    dtype = torch.promote_types(x.dtype, gamma.dtype)
    return torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=dtype)


@register_custom_op(
    op_name="triton_wan_rmsnorm_silu_cuda",
    fake_impl=_fake_wan_rmsnorm_silu,
)
def _triton_wan_rmsnorm_silu_cuda(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    rms_scale: float,
    eps: float,
    has_bias: bool,
) -> torch.Tensor:
    bsz, channels, t_size, h_size, w_size = x.shape
    # Preserve the input strides so the VAE keeps its channels_last_3d layout.
    dtype = torch.promote_types(x.dtype, gamma.dtype)
    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=dtype)
    block_c = triton.next_power_of_2(channels)
    num_warps = 1 if block_c <= 64 else 4 if block_c <= 512 else 8

    with torch.cuda.device(x.device):
        _wan_rmsnorm_silu_kernel[(bsz * t_size * h_size * w_size,)](
            x,
            gamma,
            bias,
            out,
            channels,
            t_size,
            h_size,
            w_size,
            *x.stride(),
            *out.stride(),
            rms_scale,
            eps,
            has_bias,
            block_c,
            num_warps=num_warps,
        )
    return out


def _affine_supported(x: torch.Tensor, t: torch.Tensor) -> bool:
    # Same dtype, or fp32 affine params on half-precision x (autocast case).
    return (
        t.is_cuda
        and t.device == x.device
        and (t.dtype == x.dtype or t.dtype == torch.float32)
        and t.numel() == x.shape[1]
    )


def can_use_wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    return (
        x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dtype in _SUPPORTED_DTYPES
        and x.ndim == 5
        and 0 < x.shape[1] <= _MAX_CHANNELS
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and _affine_supported(x, gamma)
        and (bias is None or _affine_supported(x, bias))
    )


def wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    rms_scale: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor | None:
    """Fused ``SiLU(F.normalize(x, dim=1) * rms_scale * gamma + bias)``.

    Returns ``None`` when the input is unsupported; callers must fall back.
    """
    if not can_use_wan_rmsnorm_silu(x, gamma, bias):
        return None

    channels = x.shape[1]
    gamma = gamma.reshape(channels).contiguous()
    has_bias = bias is not None
    bias = gamma if bias is None else bias.reshape(channels).contiguous()
    if rms_scale is None:
        rms_scale = channels**0.5
    return _triton_wan_rmsnorm_silu_cuda(
        x, gamma, bias, float(rms_scale), eps, has_bias
    )


__all__ = ["can_use_wan_rmsnorm_silu", "wan_rmsnorm_silu"]
