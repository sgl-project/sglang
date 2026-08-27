import math
from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import LTXVocoderConfig


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: Tuple[int, ...] = (1, 3, 5),
        leaky_relu_negative_slope: float = 0.1,
        padding_mode: str = "same",
    ):
        super().__init__()
        self.dilations = dilations
        self.negative_slope = leaky_relu_negative_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_mode,
                )
                for dilation in dilations
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    dilation=1,
                    padding=padding_mode,
                )
                for _ in range(len(dilations))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, negative_slope=self.negative_slope)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, negative_slope=self.negative_slope)
            xt = conv2(xt)
            x = x + xt
        return x


class LTX2Vocoder(ABC, nn.Module):
    r"""
    LTX 2.0 vocoder for converting generated mel spectrograms back to audio waveforms.
    """

    def __init__(
        self,
        config: LTXVocoderConfig,
    ):
        super().__init__()
        self.config = config
        self.sample_rate = (
            getattr(config.arch_config, "sample_rate", None)
            or getattr(config.arch_config, "sampling_rate", None)
            or getattr(config.arch_config, "audio_sample_rate", None)
        )

        in_channels = config.arch_config.in_channels
        hidden_channels = config.arch_config.hidden_channels
        out_channels = config.arch_config.out_channels
        upsample_kernel_sizes = config.arch_config.upsample_kernel_sizes
        upsample_factors = config.arch_config.upsample_factors
        resnet_kernel_sizes = config.arch_config.resnet_kernel_sizes
        resnet_dilations = config.arch_config.resnet_dilations
        leaky_relu_negative_slope = config.arch_config.leaky_relu_negative_slope

        self.num_upsample_layers = len(upsample_kernel_sizes)
        self.resnets_per_upsample = len(resnet_kernel_sizes)
        self.out_channels = out_channels
        self.total_upsample_factor = math.prod(upsample_factors)
        self.negative_slope = leaky_relu_negative_slope

        if self.num_upsample_layers != len(upsample_factors):
            raise ValueError(
                f"`upsample_kernel_sizes` and `upsample_factors` should be lists of the same length but are length"
                f" {self.num_upsample_layers} and {len(upsample_factors)}, respectively."
            )

        if self.resnets_per_upsample != len(resnet_dilations):
            raise ValueError(
                f"`resnet_kernel_sizes` and `resnet_dilations` should be lists of the same length but are length"
                f" {len(self.resnets_per_upsample)} and {len(resnet_dilations)}, respectively."
            )

        self.conv_in = nn.Conv1d(
            in_channels, hidden_channels, kernel_size=7, stride=1, padding=3
        )

        self.upsamplers = nn.ModuleList()
        self.resnets = nn.ModuleList()
        input_channels = hidden_channels
        for i, (stride, kernel_size) in enumerate(
            zip(upsample_factors, upsample_kernel_sizes)
        ):
            output_channels = input_channels // 2
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    input_channels,  # hidden_channels // (2 ** i)
                    output_channels,  # hidden_channels // (2 ** (i + 1))
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

            for kernel_size, dilations in zip(resnet_kernel_sizes, resnet_dilations):
                self.resnets.append(
                    ResBlock(
                        output_channels,
                        kernel_size,
                        dilations=dilations,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
            input_channels = output_channels

        self.conv_out = nn.Conv1d(output_channels, out_channels, 7, stride=1, padding=3)

    def forward(
        self, hidden_states: torch.Tensor, time_last: bool = False
    ) -> torch.Tensor:
        r"""
        Forward pass of the vocoder.

        Args:
            hidden_states (`torch.Tensor`):
                Input Mel spectrogram tensor of shape `(batch_size, num_channels, time, num_mel_bins)` if `time_last`
                is `False` (the default) or shape `(batch_size, num_channels, num_mel_bins, time)` if `time_last` is
                `True`.
            time_last (`bool`, *optional*, defaults to `False`):
                Whether the last dimension of the input is the time/frame dimension or the Mel bins dimension.

        Returns:
            `torch.Tensor`:
                Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """

        # Ensure that the time/frame dimension is last
        if not time_last:
            hidden_states = hidden_states.transpose(2, 3)
        # Combine channels and frequency (mel bins) dimensions
        hidden_states = hidden_states.flatten(1, 2)

        hidden_states = self.conv_in(hidden_states)

        for i in range(self.num_upsample_layers):
            hidden_states = F.leaky_relu(
                hidden_states, negative_slope=self.negative_slope
            )
            hidden_states = self.upsamplers[i](hidden_states)

            # Run all resnets in parallel on hidden_states
            start = i * self.resnets_per_upsample
            end = (i + 1) * self.resnets_per_upsample
            resnet_outputs = torch.stack(
                [self.resnets[j](hidden_states) for j in range(start, end)], dim=0
            )

            hidden_states = torch.mean(resnet_outputs, dim=0)

        # NOTE: unlike the first leaky ReLU, this leaky ReLU is set to use the default F.leaky_relu negative slope of
        # 0.01 (whereas the others usually use a slope of 0.1). Not sure if this is intended
        hidden_states = F.leaky_relu(hidden_states, negative_slope=0.01)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        return hidden_states


EntryClass = LTX2Vocoder
