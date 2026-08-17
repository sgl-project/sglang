# SPDX-License-Identifier: Apache-2.0
# DAC-lineage audio VAE: waveform encoder + BigVGAN decoder (inference-only bundle).
import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

from .bigvgan import AttrDict, BigVGAN


class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)).mul_(self.w1(x))
        x = self.w2(x)
        return x


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer("zero_k_bias", torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim**-0.5
        self.proj = nn.Linear(out_dim, out_dim)
        self.attn = (
            USPAttention(
                num_heads=num_heads,
                head_size=self.head_dim,
                causal=True,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.TORCH_SDPA,
                },
                default_attention_backend=AttentionBackendEnum.TORCH_SDPA,
                skip_sequence_parallel=True,
            )
            if current_platform.is_cuda()
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(
            input=x,
            weight=self.qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        q, k, v = (
            qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )

        if self.attn is None:
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            ).transpose(1, 2)
        else:
            input_dtype = q.dtype
            if self.attn.backend != AttentionBackendEnum.TORCH_SDPA:
                # released audio VAE stays FP32; an explicit fused backend
                # owns only the attention compute precision
                q, k, v = (tensor.to(self.attn.dtype) for tensor in (q, k, v))
            x = self.attn(q, k, v).to(input_dtype)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=2)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.reshape(B, N, -1)
        x = self.proj(x)
        return x


class AttnProjection(nn.Module):
    def __init__(
        self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2
    ):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=hidden_dim)
        # self.mlp = FeedForward(out_dim, out_dim)

    def forward(self, x):
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return self.mlp(self.norm2(x)).add_(x)


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
