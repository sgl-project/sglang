# SPDX-License-Identifier: MIT
# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Parameter
from torch.nn.utils.parametrizations import weight_norm

from .alias_free import Activation1d


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# Adapted from https://github.com/EdwardDixon/snake under the MIT license.
@torch.jit.script
def snakebeta(x, alpha, beta):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class SnakeBeta(nn.Module):
    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snakebeta(x, alpha, beta)
        return x


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Must be 'snakebeta'.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )

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
                for _ in range(len(dilation))
            ]
        )

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        if activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        activation_iter = iter(self.activations)
        for c1, c2 in zip(self.convs1, self.convs2):
            a1 = next(activation_iter)
            a2 = next(activation_iter)
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt.add_(x)

        return x


class BigVGAN(torch.nn.Module):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).

    Args:
        h (AttrDict): Hyperparameters.
    """

    def __init__(self, h: AttrDict):
        super().__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        if h.activation != "snakebeta":
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )
        activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs.div_(self.num_kernels)

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x.tanh_()
        else:
            x.clamp_(min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x
