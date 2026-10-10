# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 NVIDIA CORPORATION.
# Adapted from NVIDIA BigVGAN-v2 (MIT) via FastVideo.
# Original license terms are retained in THIRD_PARTY_LICENSES.
"""BigVGAN-v2 with the plain weights stored in Kandinsky audio checkpoints."""

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.vaes.alias_free import Activation1d


class Snake(nn.Module):
    def __init__(self, channels: int, *, logscale: bool, separate_beta: bool):
        super().__init__()
        initial = torch.zeros(channels) if logscale else torch.ones(channels)
        self.alpha = nn.Parameter(initial)
        self.beta = nn.Parameter(initial.clone()) if separate_beta else None
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha[None, :, None]
        beta = alpha if self.beta is None else self.beta[None, :, None]
        if self.logscale:
            alpha = alpha.exp()
            beta = alpha if self.beta is None else beta.exp()
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


def _activation(config: SimpleNamespace, channels: int) -> Activation1d:
    if config.activation not in ("snake", "snakebeta"):
        raise ValueError(f"Unsupported BigVGAN activation: {config.activation}")
    return Activation1d(
        Snake(
            channels,
            logscale=config.snake_logscale,
            separate_beta=config.activation == "snakebeta",
        )
    )


class AMPBlock(nn.Module):
    def __init__(self, config, channels: int, kernel: int, dilations):
        super().__init__()
        self.paired = config.resblock == "1"
        convs = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel,
                dilation=rate,
                padding=(kernel - 1) * rate // 2,
            )
            for rate in dilations
        )
        # retain both published state-dict layouts
        if self.paired:
            self.convs1 = convs
            self.convs2 = nn.ModuleList(
                nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)
                for _ in dilations
            )
        else:
            self.convs = convs
        self.activations = nn.ModuleList(
            _activation(config, channels)
            for _ in range(len(dilations) * (2 if self.paired else 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.paired:
            for index, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
                hidden = conv1(self.activations[2 * index](x))
                x = conv2(self.activations[2 * index + 1](hidden)) + x
        else:
            for conv, activation in zip(self.convs, self.activations):
                x = conv(activation(x)) + x
        return x


class BigVGANV2(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        if config.get("use_cuda_kernel", False):
            raise ValueError(
                "Kandinsky6 BigVGANV2 supports only the portable PyTorch path"
            )
        h = SimpleNamespace(**config)
        if h.resblock not in ("1", "2"):
            raise ValueError(f"Unsupported BigVGAN resblock: {h.resblock}")
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = h.upsample_initial_channel
        for rate, kernel in zip(h.upsample_rates, h.upsample_kernel_sizes, strict=True):
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose1d(
                            channels,
                            channels // 2,
                            kernel,
                            rate,
                            padding=(kernel - rate) // 2,
                        )
                    ]
                )
            )
            channels //= 2
            self.resblocks.extend(
                AMPBlock(h, channels, kernel, dilations)
                for kernel, dilations in zip(
                    h.resblock_kernel_sizes, h.resblock_dilation_sizes, strict=True
                )
            )
        self.activation_post = _activation(h, channels)
        self.conv_post = nn.Conv1d(
            channels, 1, 7, padding=3, bias=config.get("use_bias_at_final", True)
        )
        self.use_tanh_at_final = config.get("use_tanh_at_final", True)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_pre(mel)
        for index, upsamplers in enumerate(self.ups):
            for upsampler in upsamplers:
                hidden = upsampler(hidden)
            accumulated = None
            for kernel in range(self.num_kernels):
                output = self.resblocks[index * self.num_kernels + kernel](hidden)
                if accumulated is None:
                    accumulated = output
                else:
                    # preserve sequential accumulation through nonlinear upsampling
                    accumulated += output
            hidden = accumulated / self.num_kernels
        hidden = self.conv_post(self.activation_post(hidden))
        return torch.tanh(hidden) if self.use_tanh_at_final else hidden.clamp(-1.0, 1.0)
