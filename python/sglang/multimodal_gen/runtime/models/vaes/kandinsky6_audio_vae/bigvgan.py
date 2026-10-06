# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 NVIDIA CORPORATION.
# Adapted from NVIDIA BigVGAN-v2 (MIT license) via FastVideo's
# fastvideo/models/audio/bigvgan.py. The CUDA activation kernel is
# intentionally excluded; this module uses the portable PyTorch path for
# deterministic loading and parity across platforms.
# Original license terms are retained in THIRD_PARTY_LICENSES.
#
# This top-level generator class (BigVGANV2) and its Snake/SnakeBeta/
# AMPBlock1/AMPBlock2 building blocks are freshly ported here rather than
# reused from this codebase's sibling ``minimax_h3_audio_vae/bigvgan.py`` --
# that module's ``BigVGAN``/``AMPBlock1`` only support resblock="1" and
# activation="snakebeta" (MiniMax's own DAC-vocoder config never needs more),
# while this implementation supports either resblock variant or activation. The
# alias-free resampling primitives (LowPassFilter1d/UpSample1d/DownSample1d/
# Activation1d) ARE architecture-agnostic DSP building blocks shared with
# that same sibling module, though -- both import ``Activation1d`` from
# ``runtime/models/vaes/alias_free.py`` (sibling of this package, alongside
# ``common.py``) instead of each keeping its own copy.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from sglang.multimodal_gen.runtime.models.vaes.alias_free import Activation1d


class AttrDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        initial = (
            torch.zeros(in_features) * alpha
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha = nn.Parameter(initial, requires_grad=alpha_trainable)
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        initial = (
            torch.zeros(in_features) * alpha
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha = nn.Parameter(initial.clone(), requires_grad=alpha_trainable)
        self.beta = nn.Parameter(initial.clone(), requires_grad=alpha_trainable)
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )


def _activation(name: str, channels: int, logscale: bool) -> Activation1d:
    if name == "snake":
        activation = Snake(channels, alpha_logscale=logscale)
    elif name == "snakebeta":
        activation = SnakeBeta(channels, alpha_logscale=logscale)
    else:
        raise ValueError(f"Unsupported BigVGAN activation: {name}")
    return Activation1d(activation)


class AMPBlock1(nn.Module):
    def __init__(
        self,
        config: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
        activation: str = "snake",
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=rate,
                        padding=get_padding(kernel_size, rate),
                    )
                )
                for rate in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)
        self.activations = nn.ModuleList(
            [
                _activation(activation, channels, config.snake_logscale)
                for _ in range(len(self.convs1) + len(self.convs2))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, acts1, acts2, strict=True
        ):
            residual = conv2(act2(conv1(act1(x))))
            x = residual + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            remove_parametrizations(layer, "weight")


class AMPBlock2(nn.Module):
    def __init__(
        self,
        config: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
        activation: str = "snake",
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=rate,
                        padding=get_padding(kernel_size, rate),
                    )
                )
                for rate in dilation
            ]
        )
        self.convs.apply(init_weights)
        self.activations = nn.ModuleList(
            [
                _activation(activation, channels, config.snake_logscale)
                for _ in self.convs
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, activation in zip(self.convs, self.activations, strict=True):
            x = conv(activation(x)) + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_parametrizations(layer, "weight")


class BigVGANV2(nn.Module):
    """BigVGAN-v2 using checkpoint vocoder_config and the portable PyTorch path."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = dict(config)
        config.pop("_class_name", None)
        config.pop("architectures", None)
        weight_norm_removed = bool(config.pop("weight_norm_removed", False))
        self.config = AttrDict(config)
        if self.config.get("use_cuda_kernel", False):
            raise ValueError(
                "Kandinsky6 BigVGANV2 supports only the portable PyTorch path"
            )
        self.config["use_cuda_kernel"] = False
        self.num_kernels = len(self.config.resblock_kernel_sizes)
        self.num_upsamples = len(self.config.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(
                self.config.num_mels,
                self.config.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )
        if self.config.resblock == "1":
            block_class = AMPBlock1
        elif self.config.resblock == "2":
            block_class = AMPBlock2
        else:
            raise ValueError(f"Unsupported BigVGAN resblock: {self.config.resblock}")

        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(
                self.config.upsample_rates,
                self.config.upsample_kernel_sizes,
                strict=True,
            )
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                self.config.upsample_initial_channel // (2**index),
                                self.config.upsample_initial_channel
                                // (2 ** (index + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = self.config.upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(
                self.config.resblock_kernel_sizes,
                self.config.resblock_dilation_sizes,
                strict=True,
            ):
                self.resblocks.append(
                    block_class(
                        self.config,
                        channels,
                        kernel,
                        tuple(dilation),
                        activation=self.config.activation,
                    )
                )

        channels = self.config.upsample_initial_channel // (2 ** len(self.ups))
        self.activation_post = _activation(
            self.config.activation, channels, self.config.snake_logscale
        )
        self.use_bias_at_final = self.config.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(channels, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )
        for upsampler in self.ups:
            upsampler.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.use_tanh_at_final = self.config.get("use_tanh_at_final", True)
        if weight_norm_removed:
            self.remove_weight_norm()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_pre(mel)
        for index in range(self.num_upsamples):
            for upsampler in self.ups[index]:
                hidden = upsampler(hidden)
            accumulated = None
            for kernel in range(self.num_kernels):
                block_output = self.resblocks[index * self.num_kernels + kernel](hidden)
                if accumulated is None:
                    accumulated = block_output
                else:
                    # Preserve BigVGAN's published sequential accumulation
                    # order. Tree reduction drifts through later nonlinear
                    # upsampling stages with the full checkpoint.
                    accumulated += block_output
            assert accumulated is not None
            hidden = accumulated / self.num_kernels
        hidden = self.conv_post(self.activation_post(hidden))
        if self.use_tanh_at_final:
            return torch.tanh(hidden)
        return torch.clamp(hidden, min=-1.0, max=1.0)

    def remove_weight_norm(self) -> None:
        try:
            for upsamplers in self.ups:
                for upsampler in upsamplers:
                    remove_parametrizations(upsampler, "weight")
            for block in self.resblocks:
                block.remove_weight_norm()
            remove_parametrizations(self.conv_pre, "weight")
            remove_parametrizations(self.conv_post, "weight")
        except ValueError:
            # Idempotent for pipeline setup and converted checkpoints.
            return
