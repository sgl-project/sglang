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
    t_size: tl.constexpr,
    h_size: tl.constexpr,
    w_size: tl.constexpr,
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

    # Match the eager PyTorch op boundaries more closely: F.normalize returns the
    # input dtype before the affine and SiLU ops run.
    y = (x * inv_norm).to(out_ptr.dtype.element_ty)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    y = y * rms_scale * gamma
    if has_bias:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        y += bias
    y = y.to(out_ptr.dtype.element_ty)
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
    return torch.empty_strided(
        x.shape,
        x.stride(),
        device=x.device,
        dtype=x.dtype,
    )


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
    out = torch.empty_strided(
        x.shape,
        x.stride(),
        device=x.device,
        dtype=x.dtype,
    )
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
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            rms_scale,
            eps,
            has_bias,
            block_c,
            num_warps=num_warps,
        )
    return out


def can_use_triton_wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    has_channels_last_3d = hasattr(torch, "channels_last_3d")
    return (
        has_channels_last_3d
        and x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dtype in _SUPPORTED_DTYPES
        and x.ndim == 5
        and 0 < x.shape[1] <= _MAX_CHANNELS
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and gamma.is_cuda
        and gamma.device == x.device
        and gamma.dtype == x.dtype
        and gamma.numel() == x.shape[1]
        and (
            bias is None
            or (
                bias.is_cuda
                and bias.device == x.device
                and bias.dtype == x.dtype
                and bias.numel() == x.shape[1]
            )
        )
    )


def triton_wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    rms_scale: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not can_use_triton_wan_rmsnorm_silu(x, gamma, bias):
        raise RuntimeError("unsupported input for Wan RMSNorm+SiLU Triton kernel")

    channels = x.shape[1]
    gamma = gamma.reshape(channels).contiguous()
    has_bias = bias is not None
    bias = gamma if bias is None else bias.reshape(channels).contiguous()
    if rms_scale is None:
        rms_scale = channels**0.5
    return _triton_wan_rmsnorm_silu_cuda(
        x, gamma, bias, float(rms_scale), eps, has_bias
    )


__all__ = ["can_use_triton_wan_rmsnorm_silu", "triton_wan_rmsnorm_silu"]
