# SPDX-License-Identifier: Apache-2.0
#
# Adapted from MMAudio (https://github.com/hkchengrex/MMAudio), MIT License,
# Copyright (c) 2024 Sony Research Inc., via FastVideo.
# Original license terms are retained in THIRD_PARTY_LICENSES.
"""Native 1D audio VAE used by MMAudio (mel-spectrogram <-> latent codec)."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


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

    def __init__(self, mode: str = "44k", need_encoder: bool = False) -> None:
        super().__init__()
        if mode == "16k":
            data_dim, embed_dim, hidden_dim = 80, 20, 384
        elif mode == "44k":
            data_dim, embed_dim, hidden_dim = 128, 40, 512
        else:
            raise ValueError(f"Unknown MMAudio VAE mode: {mode}")

        self.mode = mode
        self.embed_dim = embed_dim
        self._weights_normalized = False
        # checkpoint buffers are required by native VAELoader's strict load
        self.register_buffer(
            "data_mean", torch.zeros(1, data_dim, 1, dtype=torch.float32)
        )
        self.register_buffer(
            "data_std", torch.ones(1, data_dim, 1, dtype=torch.float32)
        )
        self.encoder = None
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
        if self.encoder is None:
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
        # normalize only random initialization; checkpoint weights are already final
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
