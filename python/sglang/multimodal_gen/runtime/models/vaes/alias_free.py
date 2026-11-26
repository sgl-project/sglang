# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
"""Alias-free-resampling DSP primitives shared by the BigVGAN-lineage vocoders.

Generic, architecture-agnostic building blocks (Kaiser-windowed-sinc low-pass
filtering, 2x up/down resampling, and the up-activate-down ``Activation1d``
wrapper) -- not specific to any one model. Previously duplicated
byte-for-byte between ``kandinsky6_audio_vae/alias_free.py`` and
``minimax_h3_audio_vae/alias_free.py``; both ``BigVGANV2``/``BigVGAN``
implementations now import ``Activation1d`` from here instead, matching this
package's existing cross-model-shared-VAE-code convention (see ``common.py``
in this same directory, reused by ``dac.py``, ``ltx_2_audio.py``,
``wanvae.py``, ``hunyuanvae.py``, and others).
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

sinc = torch.sinc

_Conv = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
# ROCm: MIOpen serves these FP32 depthwise filters with a naive FP64 solver or a
# GEMM solver depending on the shape, at up to 0.9 ms a call. Each shape keeps
# the first Triton replica of those orders that gives the eager conv's bytes at
# its first sight, or stays eager.
_ROCM_DEPTHWISE_REPLICA: dict[tuple, _Conv | None] = {}


def _conv1d_replicas() -> list[_Conv]:
    from sglang.kernels.ops.diffusion import conv1d_depthwise_fma, conv1d_fp64acc

    def naive(x, w, stride):
        return conv1d_fp64acc(x, w, stride=stride, groups=x.shape[1])

    def gemm(x, w, stride):
        return conv1d_depthwise_fma(x, w, stride=stride)

    return [naive, gemm]


def _conv_transpose1d_replicas() -> list[_Conv]:
    from sglang.kernels.ops.diffusion import conv_transpose1d_depthwise

    def naive(x, w, stride):
        return conv_transpose1d_depthwise(x, w, stride=stride, fp64=True)

    def col2im(x, w, stride):
        return conv_transpose1d_depthwise(
            x, w, stride=stride, fp64=False, taps_ascending=False
        )

    return [naive, col2im]


def _depthwise(
    eager: _Conv, replicas: Callable[[], list[_Conv]], x, weight, stride
) -> torch.Tensor:
    if not (current_platform.is_rocm() and x.is_cuda and x.dtype == torch.float32):
        return eager(x, weight, stride)
    sig = (eager, tuple(x.shape), tuple(weight.shape), stride)
    if sig in _ROCM_DEPTHWISE_REPLICA:
        replica = _ROCM_DEPTHWISE_REPLICA[sig] or eager
        return replica(x, weight, stride)
    out = eager(x, weight, stride)
    # the check syncs the host: never inside compile tracing or graph capture
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return out
    _ROCM_DEPTHWISE_REPLICA[sig] = next(
        (r for r in replicas() if _same_bytes(r, x, weight, stride, out)), None
    )
    return out


def _same_bytes(replica: _Conv, x, weight, stride, out) -> bool:
    try:
        return torch.equal(replica(x, weight, stride), out)
    except Exception as e:  # a replica that cannot launch leaves the shape eager
        logger.warning("depthwise conv replica %s failed: %s", replica.__name__, e)
        return False


def _eager_conv1d(x, weight, stride):
    return F.conv1d(x, weight, stride=stride, groups=x.shape[1])


def _eager_conv_transpose1d(x, weight, stride):
    return F.conv_transpose1d(x, weight, stride=stride, groups=x.shape[1])


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
def kaiser_sinc_filter1d(
    cutoff, half_width, kernel_size
):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # Input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = _depthwise(
            _eager_conv1d,
            _conv1d_replicas,
            x,
            self.filter.expand(C, -1, -1),
            self.stride,
        )

        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = _depthwise(
            _eager_conv_transpose1d,
            _conv_transpose1d_replicas,
            x,
            self.filter.expand(C, -1, -1),
            self.stride,
        )
        x.mul_(self.ratio)
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x
