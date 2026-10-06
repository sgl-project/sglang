# SPDX-License-Identifier: Apache-2.0
#
# Adapted from MMAudio (https://github.com/hkchengrex/MMAudio), MIT License,
# Copyright (c) 2024 Sony Research Inc., via FastVideo's
# fastvideo/models/audio/mmaudio_vae.py. This is the mel-spectrogram
# <-> latent 1D-conv VAE codec nested at ``Kandinsky6AudioVAE.vae``
# (see kandinsky6_audio.py in the parent directory).
# Original license terms are retained in THIRD_PARTY_LICENSES.
#
# Ported verbatim except: FastVideo's ``MMAudioVAE.load_weights()`` (and its
# ``fastvideo.models.loader.weight_utils.default_weight_loader`` import) is
# dropped -- sglang's generic VAELoader never calls a per-submodule
# ``load_weights()`` hook, it does a single plain ``nn.Module.load_state_dict``
# over the whole ``Kandinsky6AudioVAE`` tree, so that method would be dead code
# here.
"""Native 1D audio VAE used by MMAudio (mel-spectrogram <-> latent codec)."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


DATA_MEAN_80D = [
    -1.6058,
    -1.3676,
    -1.2520,
    -1.2453,
    -1.2078,
    -1.2224,
    -1.2419,
    -1.2439,
    -1.2922,
    -1.2927,
    -1.3170,
    -1.3543,
    -1.3401,
    -1.3836,
    -1.3907,
    -1.3912,
    -1.4313,
    -1.4152,
    -1.4527,
    -1.4728,
    -1.4568,
    -1.5101,
    -1.5051,
    -1.5172,
    -1.5623,
    -1.5373,
    -1.5746,
    -1.5687,
    -1.6032,
    -1.6131,
    -1.6081,
    -1.6331,
    -1.6489,
    -1.6489,
    -1.6700,
    -1.6738,
    -1.6953,
    -1.6969,
    -1.7048,
    -1.7280,
    -1.7361,
    -1.7495,
    -1.7658,
    -1.7814,
    -1.7889,
    -1.8064,
    -1.8221,
    -1.8377,
    -1.8417,
    -1.8643,
    -1.8857,
    -1.8929,
    -1.9173,
    -1.9379,
    -1.9531,
    -1.9673,
    -1.9824,
    -2.0042,
    -2.0215,
    -2.0436,
    -2.0766,
    -2.1064,
    -2.1418,
    -2.1855,
    -2.2319,
    -2.2767,
    -2.3161,
    -2.3572,
    -2.3954,
    -2.4282,
    -2.4659,
    -2.5072,
    -2.5552,
    -2.6074,
    -2.6584,
    -2.7107,
    -2.7634,
    -2.8266,
    -2.8981,
    -2.9673,
]

DATA_STD_80D = [
    1.0291,
    1.0411,
    1.0043,
    0.9820,
    0.9677,
    0.9543,
    0.9450,
    0.9392,
    0.9343,
    0.9297,
    0.9276,
    0.9263,
    0.9242,
    0.9254,
    0.9232,
    0.9281,
    0.9263,
    0.9315,
    0.9274,
    0.9247,
    0.9277,
    0.9199,
    0.9188,
    0.9194,
    0.9160,
    0.9161,
    0.9146,
    0.9161,
    0.9100,
    0.9095,
    0.9145,
    0.9076,
    0.9066,
    0.9095,
    0.9032,
    0.9043,
    0.9038,
    0.9011,
    0.9019,
    0.9010,
    0.8984,
    0.8983,
    0.8986,
    0.8961,
    0.8962,
    0.8978,
    0.8962,
    0.8973,
    0.8993,
    0.8976,
    0.8995,
    0.9016,
    0.8982,
    0.8972,
    0.8974,
    0.8949,
    0.8940,
    0.8947,
    0.8936,
    0.8939,
    0.8951,
    0.8956,
    0.9017,
    0.9167,
    0.9436,
    0.9690,
    1.0003,
    1.0225,
    1.0381,
    1.0491,
    1.0545,
    1.0604,
    1.0761,
    1.0929,
    1.1089,
    1.1196,
    1.1176,
    1.1156,
    1.1117,
    1.1070,
]

DATA_MEAN_128D = [
    -3.3462,
    -2.6723,
    -2.4893,
    -2.3143,
    -2.2664,
    -2.3317,
    -2.1802,
    -2.4006,
    -2.2357,
    -2.4597,
    -2.3717,
    -2.4690,
    -2.5142,
    -2.4919,
    -2.6610,
    -2.5047,
    -2.7483,
    -2.5926,
    -2.7462,
    -2.7033,
    -2.7386,
    -2.8112,
    -2.7502,
    -2.9594,
    -2.7473,
    -3.0035,
    -2.8891,
    -2.9922,
    -2.9856,
    -3.0157,
    -3.1191,
    -2.9893,
    -3.1718,
    -3.0745,
    -3.1879,
    -3.2310,
    -3.1424,
    -3.2296,
    -3.2791,
    -3.2782,
    -3.2756,
    -3.3134,
    -3.3509,
    -3.3750,
    -3.3951,
    -3.3698,
    -3.4505,
    -3.4509,
    -3.5089,
    -3.4647,
    -3.5536,
    -3.5788,
    -3.5867,
    -3.6036,
    -3.6400,
    -3.6747,
    -3.7072,
    -3.7279,
    -3.7283,
    -3.7795,
    -3.8259,
    -3.8447,
    -3.8663,
    -3.9182,
    -3.9605,
    -3.9861,
    -4.0105,
    -4.0373,
    -4.0762,
    -4.1121,
    -4.1488,
    -4.1874,
    -4.2461,
    -4.3170,
    -4.3639,
    -4.4452,
    -4.5282,
    -4.6297,
    -4.7019,
    -4.7960,
    -4.8700,
    -4.9507,
    -5.0303,
    -5.0866,
    -5.1634,
    -5.2342,
    -5.3242,
    -5.4053,
    -5.4927,
    -5.5712,
    -5.6464,
    -5.7052,
    -5.7619,
    -5.8410,
    -5.9188,
    -6.0103,
    -6.0955,
    -6.1673,
    -6.2362,
    -6.3120,
    -6.3926,
    -6.4797,
    -6.5565,
    -6.6511,
    -6.8130,
    -6.9961,
    -7.1275,
    -7.2457,
    -7.3576,
    -7.4663,
    -7.6136,
    -7.7469,
    -7.8815,
    -8.0132,
    -8.1515,
    -8.3071,
    -8.4722,
    -8.7418,
    -9.3975,
    -9.6628,
    -9.7671,
    -9.8863,
    -9.9992,
    -10.0860,
    -10.1709,
    -10.5418,
    -11.2795,
    -11.3861,
]

DATA_STD_128D = [
    2.3804,
    2.4368,
    2.3772,
    2.3145,
    2.2803,
    2.2510,
    2.2316,
    2.2083,
    2.1996,
    2.1835,
    2.1769,
    2.1659,
    2.1631,
    2.1618,
    2.1540,
    2.1606,
    2.1571,
    2.1567,
    2.1612,
    2.1579,
    2.1679,
    2.1683,
    2.1634,
    2.1557,
    2.1668,
    2.1518,
    2.1415,
    2.1449,
    2.1406,
    2.1350,
    2.1313,
    2.1415,
    2.1281,
    2.1352,
    2.1219,
    2.1182,
    2.1327,
    2.1195,
    2.1137,
    2.1080,
    2.1179,
    2.1036,
    2.1087,
    2.1036,
    2.1015,
    2.1068,
    2.0975,
    2.0991,
    2.0902,
    2.1015,
    2.0857,
    2.0920,
    2.0893,
    2.0897,
    2.0910,
    2.0881,
    2.0925,
    2.0873,
    2.0960,
    2.0900,
    2.0957,
    2.0958,
    2.0978,
    2.0936,
    2.0886,
    2.0905,
    2.0845,
    2.0855,
    2.0796,
    2.0840,
    2.0813,
    2.0817,
    2.0838,
    2.0840,
    2.0917,
    2.1061,
    2.1431,
    2.1976,
    2.2482,
    2.3055,
    2.3700,
    2.4088,
    2.4372,
    2.4609,
    2.4731,
    2.4847,
    2.5072,
    2.5451,
    2.5772,
    2.6147,
    2.6529,
    2.6596,
    2.6645,
    2.6726,
    2.6803,
    2.6812,
    2.6899,
    2.6916,
    2.6931,
    2.6998,
    2.7062,
    2.7262,
    2.7222,
    2.7158,
    2.7041,
    2.7485,
    2.7491,
    2.7451,
    2.7485,
    2.7233,
    2.7297,
    2.7233,
    2.7145,
    2.6958,
    2.6788,
    2.6439,
    2.6007,
    2.4786,
    2.2469,
    2.1877,
    2.1392,
    2.0717,
    2.0107,
    1.9676,
    1.9140,
    1.7102,
    0.9101,
    0.7164,
]


_NORMALIZATION_EPSILON = 1e-4
_SILU_DIVISOR = 0.596
_RESIDUAL_BLEND = 0.3
_RESIDUAL_DIVISOR = math.hypot(1.0 - _RESIDUAL_BLEND, _RESIDUAL_BLEND)


def _rms_normalize(x: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    dims = (dim,) if isinstance(dim, int) else dim
    element_count = math.prod(x.shape[axis] for axis in dims)
    l2_norm = torch.linalg.vector_norm(x, dim=dims, keepdim=True, dtype=torch.float32)
    rms = torch.add(_NORMALIZATION_EPSILON, l2_norm, alpha=element_count**-0.5)
    return x / rms.to(x.dtype)


def _conv1d(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
    )


def _conv1d_with_gain(
    conv: nn.Conv1d, x: torch.Tensor, gain: torch.Tensor | float
) -> torch.Tensor:
    return F.conv1d(
        x,
        conv.weight * gain,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.empty_like(self.mean).normal_(generator=generator)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int | None = None,
        conv_shortcut: bool = False,
        kernel_size: int = 3,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim if out_dim is None else out_dim
        self.use_conv_shortcut = conv_shortcut
        self.use_norm = use_norm
        self.conv1 = _conv1d(in_dim, self.out_dim, kernel_size)
        self.conv2 = _conv1d(self.out_dim, self.out_dim, kernel_size)
        if self.in_dim != self.out_dim:
            if conv_shortcut:
                self.conv_shortcut = _conv1d(in_dim, self.out_dim, kernel_size)
            else:
                self.nin_shortcut = _conv1d(in_dim, self.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            x = _rms_normalize(x, dim=1)
        hidden = self.conv1(F.silu(x) / _SILU_DIVISOR)
        hidden = self.conv2(F.silu(hidden) / _SILU_DIVISOR)
        if self.in_dim != self.out_dim:
            shortcut = (
                self.conv_shortcut if self.use_conv_shortcut else self.nin_shortcut
            )
            x = shortcut(x)
        return torch.lerp(x, hidden, _RESIDUAL_BLEND) / _RESIDUAL_DIVISOR


class AttnBlock1D(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.qkv = _conv1d(in_channels, in_channels * 3, 1)
        self.proj_out = _conv1d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x).reshape(x.shape[0], self.num_heads, -1, 3, x.shape[-1])
        query, key, value = _rms_normalize(qkv, dim=2).unbind(3)
        query = rearrange(query, "b h c l -> b h l c")
        key = rearrange(key, "b h c l -> b h l c")
        value = rearrange(value, "b h c l -> b h l c")
        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = rearrange(hidden, "b h l c -> b (h c) l")
        return torch.lerp(x, self.proj_out(hidden), _RESIDUAL_BLEND) / _RESIDUAL_DIVISOR


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = _conv1d(in_channels, in_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest-exact")
        return self.conv(x) if self.with_conv else x


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv1 = _conv1d(in_channels, in_channels, 1)
            self.conv2 = _conv1d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = self.conv1(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return self.conv2(x) if self.with_conv else x


class Encoder1D(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        attn_layers: list[int],
        down_layers: list[int],
        in_dim: int,
        embed_dim: int,
        resamp_with_conv: bool = True,
        double_z: bool = True,
        kernel_size: int = 3,
        clip_act: float = 256.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = _conv1d(in_dim, dim, kernel_size)

        in_ch_mult = (1,) + ch_mult
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = dim * in_ch_mult[level]
            block_out = dim * ch_mult[level]
            for _ in range(num_res_blocks):
                block.append(
                    ResnetBlock1D(
                        in_dim=block_in,
                        out_dim=block_out,
                        kernel_size=kernel_size,
                        use_norm=True,
                    )
                )
                block_in = block_out
                if level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if level in down_layers:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_dim=block_in, out_dim=block_in, kernel_size=kernel_size, use_norm=True
        )
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(
            in_dim=block_in, out_dim=block_in, kernel_size=kernel_size, use_norm=True
        )
        output_dim = 2 * embed_dim if double_z else embed_dim
        self.conv_out = _conv1d(block_in, output_dim, kernel_size)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [self.conv_in(x)]
        for level in range(self.num_layers):
            for block_index in range(self.num_res_blocks):
                hidden = self.down[level].block[block_index](states[-1])
                if len(self.down[level].attn) > 0:
                    hidden = self.down[level].attn[block_index](hidden)
                states.append(hidden.clamp(-self.clip_act, self.clip_act))
            if level in self.down_layers:
                states.append(self.down[level].downsample(states[-1]))
        hidden = self.mid.block_1(states[-1])
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden).clamp(-self.clip_act, self.clip_act)
        return _conv1d_with_gain(
            self.conv_out, F.silu(hidden) / _SILU_DIVISOR, self.learnable_gain + 1
        )


class Decoder1D(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        out_dim: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        attn_layers: list[int],
        down_layers: list[int],
        in_dim: int,
        embed_dim: int,
        kernel_size: int = 3,
        resamp_with_conv: bool = True,
        clip_act: float = 256.0,
    ) -> None:
        super().__init__()
        self.ch = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = [level + 1 for level in down_layers]
        block_in = dim * ch_mult[-1]
        self.conv_in = _conv1d(embed_dim, block_in, kernel_size)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_dim=block_in, out_dim=block_in, use_norm=True
        )
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(
            in_dim=block_in, out_dim=block_in, use_norm=True
        )

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = dim * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                block.append(
                    ResnetBlock1D(in_dim=block_in, out_dim=block_out, use_norm=True)
                )
                block_in = block_out
                if level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level in self.down_layers:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up)

        self.conv_out = _conv1d(block_in, out_dim, kernel_size)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_in(z)
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden).clamp(-self.clip_act, self.clip_act)
        for level in reversed(range(self.num_layers)):
            for block_index in range(self.num_res_blocks + 1):
                hidden = self.up[level].block[block_index](hidden)
                if len(self.up[level].attn) > 0:
                    hidden = self.up[level].attn[block_index](hidden)
                hidden = hidden.clamp(-self.clip_act, self.clip_act)
            if level in self.down_layers:
                hidden = self.up[level].upsample(hidden)
        return _conv1d_with_gain(
            self.conv_out, F.silu(hidden) / _SILU_DIVISOR, self.learnable_gain + 1
        )


class MMAudioVAE(nn.Module):
    """MMAudio mel-spectrogram VAE for 16 kHz or 44.1 kHz audio."""

    def __init__(
        self, mode: str | dict[str, Any] = "44k", need_encoder: bool = False
    ) -> None:
        super().__init__()
        if isinstance(mode, dict):
            config = mode
            mode = config.get("mode", "44k")
            need_encoder = config.get("need_encoder", need_encoder)
        if mode == "16k":
            data_dim, embed_dim, hidden_dim = 80, 20, 384
            data_mean, data_std = DATA_MEAN_80D, DATA_STD_80D
        elif mode == "44k":
            data_dim, embed_dim, hidden_dim = 128, 40, 512
            data_mean, data_std = DATA_MEAN_128D, DATA_STD_128D
        else:
            raise ValueError(f"Unknown MMAudio VAE mode: {mode}")

        self.mode = mode
        self.embed_dim = embed_dim
        self._weights_normalized = False
        self.register_buffer(
            "data_mean", torch.tensor(data_mean, dtype=torch.float32).view(1, -1, 1)
        )
        self.register_buffer(
            "data_std", torch.tensor(data_std, dtype=torch.float32).view(1, -1, 1)
        )
        if need_encoder:
            self.encoder = Encoder1D(
                dim=hidden_dim,
                ch_mult=(1, 2, 4),
                num_res_blocks=2,
                attn_layers=[3],
                down_layers=[0],
                in_dim=data_dim,
                embed_dim=embed_dim,
            )
        self.decoder = Decoder1D(
            dim=hidden_dim,
            ch_mult=(1, 2, 4),
            num_res_blocks=2,
            attn_layers=[3],
            down_layers=[0],
            in_dim=data_dim,
            out_dim=data_dim,
            embed_dim=embed_dim,
        )

    def encode(
        self, mel: torch.Tensor, normalize_input: bool = True
    ) -> DiagonalGaussianDistribution:
        self._require_normalized_weights()
        if not hasattr(self, "encoder"):
            raise RuntimeError("This MMAudio VAE was loaded decoder-only")
        if normalize_input:
            mel = self.normalize(mel)
        return DiagonalGaussianDistribution(self.encoder(mel))

    def decode(
        self, latent: torch.Tensor, unnormalize_output: bool = True
    ) -> torch.Tensor:
        self._require_normalized_weights()
        mel = self.decoder(latent)
        return self.unnormalize(mel) if unnormalize_output else mel

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.data_mean) / self.data_std

    def unnormalize(self, mel: torch.Tensor) -> torch.Tensor:
        return mel * self.data_std + self.data_mean

    def _require_normalized_weights(self) -> None:
        if not self._weights_normalized:
            raise RuntimeError("call remove_weight_norm() before inference")

    @torch.no_grad()
    def remove_weight_norm(self) -> MMAudioVAE:
        # Idempotent: the diffusers reference calls this exactly once, in
        # ``MMAudioVAE.__init__``, BEFORE ``from_pretrained`` overwrites the
        # freshly-constructed (random) Conv1d weights with a checkpoint's
        # real, already-final values -- so a real checkpoint's own weights
        # are never renormalized. Mirroring that timing here (see
        # ``Kandinsky6AudioVAE.__init__``) means this method only ever runs
        # against pre-load weights too; the guard below additionally makes a
        # second, redundant call (e.g. the wrapper's own public
        # ``remove_weight_norm()``) a true no-op instead of re-mutating an
        # already-normalized (possibly checkpoint-loaded) weight a second
        # time.
        if self._weights_normalized:
            return self
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                weight = _rms_normalize(module.weight.to(torch.float32), dim=(1, 2))
                weight = weight / math.sqrt(weight[0].numel())
                module.weight.copy_(weight.to(module.weight.dtype))
                logger.debug("Removed weight norm from %s", name)
        self._weights_normalized = True
        return self
