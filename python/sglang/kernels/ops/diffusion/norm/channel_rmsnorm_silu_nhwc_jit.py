# SPDX-License-Identifier: Apache-2.0
"""Channels-last channel RMSNorm + SiLU for the Qwen-Image 2.1 VAE (quality-gated, not bit-exact).

``silu(F.normalize(x.float(), dim=1).to(bf16) * scale * gamma + 0.0)`` for activations
whose channel dimension is innermost (``x.stride(1) == 1``, e.g. a decoder running in
``channels_last``). A group of lanes reduces the fp32 sum of squares of a pixel's
channels itself, so the NCHW reduction and the NCHW/NHWC transposes cuDNN otherwise
inserts around the convs disappear. The pointwise tail keeps the eager rounding points;
only the reduction order differs from aten's, so callers mount this behind the request
``quality`` gate (``extra-high`` / ``high``). An optional per-channel ``bias`` is added
first with aten's bf16 rounding, so a preceding conv can skip its own bias pass.
Even channel counts up to 2048. CUDA only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_CHANNELS = 2048
# Channels held in registers per lane stay at 64 whatever the load width.
_MAX_UNITS_PER_LANE = {8: 8, 4: 16, 2: 32}


def _specialization(channels: int) -> tuple[int, int, int]:
    """(bf16 per load, lanes per pixel, loads per lane) for a channel count.

    The widest load the channel count allows, then the fewest lanes per pixel
    whose per-lane load count fits the register budget: fewer lanes per pixel
    means more pixels per warp and more loads in flight per lane, which is
    what keeps a narrow C (144) near bandwidth instead of latency-bound.
    """
    vec = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 2
    units = channels // vec
    lanes = 1
    while lanes < 32 and -(-units // lanes) > _MAX_UNITS_PER_LANE[vec]:
        lanes *= 2
    return vec, lanes, -(-units // lanes)


@cache_once
def _jit_module(vec: int, lanes: int, units: int, has_bias: bool) -> Module:
    args = make_cpp_args(vec, lanes, units, has_bias)
    return load_jit(
        "channel_rmsnorm_silu_nhwc",
        *args,
        cuda_files=["diffusion/channel_rmsnorm_silu_nhwc.cuh"],
        cuda_wrappers=[("run", f"channel_rmsnorm_silu_nhwc::Kernel<{args}>::run")],
    )


def _channel_last_dims(x: torch.Tensor) -> tuple[int, ...]:
    return (0, *range(2, x.ndim), 1)


def is_channels_last_dense(x: torch.Tensor) -> bool:
    """Channel innermost and the remaining dims dense in (N, spatial...) order."""
    return (
        x.ndim in (4, 5)
        and x.stride(1) == 1
        and x.permute(*_channel_last_dims(x)).is_contiguous()
    )


def _is_channel_vector(t: torch.Tensor, x: torch.Tensor, align: int) -> bool:
    return (
        t.device == x.device
        and t.dtype is torch.bfloat16
        and t.shape == (x.shape[1],) + (1,) * (x.ndim - 2)
        and t.is_contiguous()
        and t.data_ptr() % align == 0
    )


def can_use_channel_rmsnorm_silu_nhwc(
    x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    if not (
        x.is_cuda
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and x.dtype is torch.bfloat16
        and x.numel() > 0
        and is_channels_last_dense(x)
        and x.shape[1] % 2 == 0
        and x.shape[1] <= _MAX_CHANNELS
    ):
        return False
    align = 2 * _specialization(x.shape[1])[0]
    return (
        x.data_ptr() % align == 0
        and _is_channel_vector(gamma, x, align)
        and (bias is None or _is_channel_vector(bias, x, align))
    )


def _fake(x, gamma, scale, bias=None):
    return torch.empty_like(x)


@register_custom_op(
    op_name="channel_rmsnorm_silu_nhwc", mutates_args=[], fake_impl=_fake
)
def channel_rmsnorm_silu_nhwc(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``silu(rmsnorm(x + bias))`` over the channel dim of a channels-last tensor; output keeps ``x``'s strides."""
    out = torch.empty_like(x)
    dims = _channel_last_dims(x)
    channels = x.shape[1]
    vec, lanes, units = _specialization(channels)
    with torch.cuda.device(x.device):
        _jit_module(vec, lanes, units, bias is not None).run(
            out.permute(*dims).view(-1, channels),
            x.permute(*dims).view(-1, channels),
            gamma.reshape(-1),
            (gamma if bias is None else bias).reshape(-1),
            float(scale),
        )
    return out


__all__ = [
    "can_use_channel_rmsnorm_silu_nhwc",
    "channel_rmsnorm_silu_nhwc",
    "is_channels_last_dense",
]
