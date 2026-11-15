# SPDX-License-Identifier: Apache-2.0
"""FP32 Wan normalization post-ops with the native L2 denominator supplied."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _module(x_dtype, affine_dtype, channels_last):
    args = make_cpp_args(x_dtype, affine_dtype, channels_last)
    return load_jit(
        "wan_norm_silu_post",
        *args,
        cuda_files=["elementwise/wan_norm_silu_post.cuh"],
        extra_cuda_cflags=["--fmad=false"],
        cuda_wrappers=[("run", f"wan_norm_silu_post<{args}>")],
    )


def can_use_wan_norm_silu_post(x, denominator, gamma, bias=None):
    if (
        not x.is_cuda
        or torch.version.hip is not None
        or torch.is_grad_enabled()
        or x.requires_grad
        or x.ndim != 5
        or x.numel() == 0
        or x.dtype not in (torch.bfloat16, torch.float32)
        or x.shape[1] % 4
        or (x.shape[2] * x.shape[3] * x.shape[4]) % 4
        or not (
            x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last_3d)
        )
        or x.data_ptr() % 16
        or denominator.dtype != torch.float32
        or denominator.device != x.device
        or denominator.shape != (x.shape[0], 1, *x.shape[2:])
        or not denominator.is_contiguous()
        or denominator.data_ptr() % 16
        or gamma.dtype not in (torch.bfloat16, torch.float32)
        or gamma.device != x.device
        or gamma.numel() != x.shape[1]
        or not gamma.is_contiguous()
        or gamma.data_ptr() % 16
    ):
        return False
    return bias is None or (
        bias.device == x.device
        and bias.dtype == gamma.dtype
        and bias.numel() == x.shape[1]
        and bias.is_contiguous()
        and bias.data_ptr() % 16 == 0
    )


def _storage(x, channels_last):
    return x.permute(0, 2, 3, 4, 1).view(-1) if channels_last else x.view(-1)


@register_custom_op(mutates_args=["out"])
def _wan_norm_silu_post(
    x: torch.Tensor,
    denominator: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    has_bias: bool,
) -> None:
    channels_last = not x.is_contiguous()
    _module(x.dtype, gamma.dtype, channels_last).run(
        _storage(x, channels_last),
        denominator.view(-1),
        gamma.view(-1),
        bias.view(-1),
        _storage(out, channels_last),
        x.shape[1],
        x.shape[2] * x.shape[3] * x.shape[4],
        scale,
        has_bias,
    )


def wan_norm_silu_post(x, denominator, gamma, bias=None, *, scale=None):
    if not can_use_wan_norm_silu_post(x, denominator, gamma, bias):
        raise ValueError("unsupported Wan normalization post-op inputs")
    out = torch.empty_like(x, dtype=torch.float32)
    _wan_norm_silu_post(
        x,
        denominator,
        gamma,
        gamma if bias is None else bias,
        out,
        float(x.shape[1] ** 0.5 if scale is None else scale),
        bias is not None,
    )
    return out
