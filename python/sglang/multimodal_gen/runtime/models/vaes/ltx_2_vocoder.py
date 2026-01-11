import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=(kernel_size - 1) * dilation[0] // 2,
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=(kernel_size - 1) * dilation[1] // 2,
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=(kernel_size - 1) * dilation[2] // 2,
                ),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: List[int] = [1, 3]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=(kernel_size - 1) * dilation[0] // 2,
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=(kernel_size - 1) * dilation[1] // 2,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class LTXVocoder(torch.nn.Module):
    """
    Vocoder model for synthesizing audio from Mel spectrograms.
    """

    def __init__(
        self,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        resblock: str = "1",
        output_sample_rate: int = 24000,
    ):
        super().__init__()

        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size,
                    stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i, _ in enumerate(self.ups):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_class(ch, kernel_size, dilations))

        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.Conv1d(final_channels, out_channels, 7, 1, padding=3)

        self.upsample_factor = math.prod(layer.stride[0] for layer in self.ups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vocoder.
        Args:
            x: Input Mel spectrogram tensor. Can be either:
               - 3D: (batch_size, time, mel_bins) for mono
               - 4D: (batch_size, 2, time, mel_bins) for stereo
        Returns:
            Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """
        # x: (batch, channels, time, mel_bins) -> (batch, channels, mel_bins, time)
        # Note: LTX implementation does x.transpose(2, 3) assuming (B, C, T, F)
        # We need to ensure input shape matches.
        
        if x.dim() == 4:
             x = x.transpose(2, 3)
             
        if x.dim() == 4:  # stereo
            # assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            # einops.rearrange(x, "b s c t -> b (s c) t")
            b, s, c, t = x.shape
            x = x.reshape(b, s * c, t)

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels

            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                dim=0,
            )

            x = block_outputs.mean(dim=0)

        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)
