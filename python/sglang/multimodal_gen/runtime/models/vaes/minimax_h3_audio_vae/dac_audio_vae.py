# SPDX-License-Identifier: Apache-2.0
# DAC-lineage audio VAE: waveform encoder + BigVGAN decoder (inference-only bundle).
import math
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .dac_attn_proj import AttnProjection
from .dac_bigvgan import BigVGAN


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DacAudioVAE(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        sample_rate: int = 44100,
        vae_latent_channels: int = 64,
        attn_proj: bool = False,
        decoder_type: str = "bigvgan",
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.attn_proj = attn_proj
        self.decoder_type = decoder_type

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        if latent_dim % vae_latent_channels == 0:
            self.attn_proj_dim = vae_latent_channels
        else:
            # smallest power of two >= vae_latent_channels
            self.attn_proj_dim = 2 ** int(np.ceil(np.log2(vae_latent_channels)))

        self.mean_proj = nn.Conv1d(self.attn_proj_dim, vae_latent_channels, 1)
        self.logs_proj = nn.Conv1d(self.attn_proj_dim, vae_latent_channels, 1)

        self.dec_in_proj = nn.Conv1d(vae_latent_channels, latent_dim, 1)

        if self.decoder_type == "bigvgan":
            if sample_rate == 16000:
                bigvgan_conf = {
                    "resblock": "1",
                    "num_mels": latent_dim,
                    "upsample_rates": [5, 5, 2, 2, 2, 2],
                    "upsample_kernel_sizes": [9, 9, 4, 4, 4, 4],
                    "upsample_initial_channel": decoder_dim,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "use_tanh_at_final": False,
                    "use_bias_at_final": False,
                    "activation": "snakebeta",
                    "snake_logscale": True,
                }
            elif sample_rate == 32000:
                bigvgan_conf = {
                    "resblock": "1",
                    "num_mels": latent_dim,
                    "upsample_rates": [5, 5, 2, 2, 2, 2, 2],
                    "upsample_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
                    "upsample_initial_channel": decoder_dim,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "use_tanh_at_final": False,
                    "use_bias_at_final": False,
                    "activation": "snakebeta",
                    "snake_logscale": True,
                }
            else:
                raise ValueError(f"Invalid sample_rate: {sample_rate}")

            h = AttrDict(**bigvgan_conf)
            self.decoder = BigVGAN(h)
        else:
            raise ValueError(f"Invalid decoder type: {self.decoder_type}")

        if self.attn_proj:
            self.pre_block = AttnProjection(latent_dim, self.attn_proj_dim, num_heads=8)

        self.sample_rate = sample_rate

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        if right_pad:
            audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Continuous latent representation

        Returns
        -------
        Tensor[B x 1 x length]
            Decoded audio data.
        """
        z = self.dec_in_proj(z)
        return self.decoder(z)
