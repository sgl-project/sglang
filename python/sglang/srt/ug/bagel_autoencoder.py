# SPDX-License-Identifier: Apache-2.0

"""BAGEL autoencoder implementation used by SRT visual feature extractors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from safetensors.torch import load_file as load_sft
from torch import Tensor, nn


@dataclass
class BAGELAutoEncoderParams:
    resolution: int
    in_channels: int
    downsample: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class BAGELAttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        batch, channels, height, width = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        hidden_states = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(
            hidden_states,
            "b 1 (h w) c -> b c h w",
            h=height,
            w=width,
            c=channels,
            b=batch,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class BAGELResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32,
            num_channels=out_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x):
        hidden_states = self.norm1(x)
        hidden_states = _swish(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = _swish(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden_states


class BAGELDownsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def forward(self, x: Tensor):
        return self.conv(nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class BAGELUpsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class BAGELEncoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        num_resolutions = len(ch_mult)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for level in range(num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks):
                block.append(BAGELResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if level != num_resolutions - 1:
                down.downsample = BAGELDownsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = BAGELResnetBlock(block_in, block_in)
        self.mid.attn_1 = BAGELAttnBlock(block_in)
        self.mid.block_2 = BAGELResnetBlock(block_in, block_in)
        self.norm_out = nn.GroupNorm(
            num_groups=32,
            num_channels=block_in,
            eps=1e-6,
            affine=True,
        )
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for level, down in enumerate(self.down):
            for block in down.block:
                hidden_states = block(hs[-1])
                hs.append(hidden_states)
            if level != len(self.down) - 1:
                hs.append(down.downsample(hs[-1]))

        hidden_states = hs[-1]
        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = _swish(hidden_states)
        return self.conv_out(hidden_states)


class BAGELDecoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        del in_channels
        num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[num_resolutions - 1]
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        self.mid = nn.Module()
        self.mid.block_1 = BAGELResnetBlock(block_in, block_in)
        self.mid.attn_1 = BAGELAttnBlock(block_in)
        self.mid.block_2 = BAGELResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for level in reversed(range(num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                block.append(BAGELResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = BAGELUpsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(
            num_groups=32,
            num_channels=block_in,
            eps=1e-6,
            affine=True,
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        hidden_states = self.conv_in(z)
        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states)
        for level in reversed(range(len(self.up))):
            for block in self.up[level].block:
                hidden_states = block(hidden_states)
            if level != 0:
                hidden_states = self.up[level].upsample(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = _swish(hidden_states)
        return self.conv_out(hidden_states)


class BAGELDiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if not self.sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)


class BAGELAutoEncoder(nn.Module):
    def __init__(self, params: BAGELAutoEncoderParams):
        super().__init__()
        self.encoder = BAGELEncoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = BAGELDecoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = BAGELDiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        return self.scale_factor * (z - self.shift_factor)

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def _print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if missing:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    if unexpected:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_bagel_autoencoder(
    local_path: str | None,
) -> tuple[BAGELAutoEncoder, BAGELAutoEncoderParams]:
    params = BAGELAutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    autoencoder = BAGELAutoEncoder(params)
    if local_path is not None:
        state_dict = load_sft(local_path)
        missing, unexpected = autoencoder.load_state_dict(
            state_dict,
            strict=False,
            assign=True,
        )
        _print_load_warning(missing, unexpected)
    return autoencoder, params
