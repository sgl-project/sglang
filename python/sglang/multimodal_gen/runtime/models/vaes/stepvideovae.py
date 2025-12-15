# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.vaes import StepVideoVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE


def base_group_norm(x, norm_layer, act_silu=False, channel_last=False) -> torch.Tensor:
    if hasattr(base_group_norm, "spatial") and base_group_norm.spatial:
        assert channel_last
        x_shape = x.shape
        x = x.flatten(0, 1)
        if channel_last:
            # Permute to NCHW format
            x = x.permute(0, 3, 1, 2)

        out = F.group_norm(
            x.contiguous(),
            norm_layer.num_groups,
            norm_layer.weight,
            norm_layer.bias,
            norm_layer.eps,
        )
        if act_silu:
            out = F.silu(out)

        if channel_last:
            # Permute back to NHWC format
            out = out.permute(0, 2, 3, 1)

        out = out.view(x_shape)
    else:
        if channel_last:
            # Permute to NCHW format
            x = x.permute(0, 3, 1, 2)
        out = F.group_norm(
            x.contiguous(),
            norm_layer.num_groups,
            norm_layer.weight,
            norm_layer.bias,
            norm_layer.eps,
        )
        if act_silu:
            out = F.silu(out)
        if channel_last:
            # Permute back to NHWC format
            out = out.permute(0, 2, 3, 1)
    return out


def base_conv2d(x, conv_layer, channel_last=False, residual=None) -> torch.Tensor:
    if channel_last:
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
    out = F.conv2d(
        x,
        conv_layer.weight,
        conv_layer.bias,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
    )
    if residual is not None:
        if channel_last:
            residual = residual.permute(0, 3, 1, 2)  # NHWC to NCHW
        out += residual
    if channel_last:
        out = out.permute(0, 2, 3, 1)  # NCHW to NHWC
    return out


def base_conv3d(
    x, conv_layer, channel_last=False, residual=None, only_return_output=False
) -> torch.Tensor:
    if only_return_output:
        size = cal_outsize(
            x.shape, conv_layer.weight.shape, conv_layer.stride, conv_layer.padding
        )
        return torch.empty(size, device=x.device, dtype=x.dtype)
    if channel_last:
        x = x.permute(0, 4, 1, 2, 3)  # NDHWC to NCDHW
    out = F.conv3d(
        x,
        conv_layer.weight,
        conv_layer.bias,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
    )
    if residual is not None:
        if channel_last:
            residual = residual.permute(0, 4, 1, 2, 3)  # NDHWC to NCDHW
        out += residual
    if channel_last:
        out = out.permute(0, 2, 3, 4, 1)  # NCDHW to NDHWC
    return out


def cal_outsize(input_sizes, kernel_sizes, stride, padding) -> list:
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = 1, 1, 1

    in_d = input_sizes[1]
    in_h = input_sizes[2]
    in_w = input_sizes[3]

    kernel_d = kernel_sizes[2]
    kernel_h = kernel_sizes[3]
    kernel_w = kernel_sizes[4]
    out_channels = kernel_sizes[0]

    out_d = calc_out_(in_d, padding_d, dilation_d, kernel_d, stride_d)
    out_h = calc_out_(in_h, padding_h, dilation_h, kernel_h, stride_h)
    out_w = calc_out_(in_w, padding_w, dilation_w, kernel_w, stride_w)
    size = [input_sizes[0], out_d, out_h, out_w, out_channels]
    return size


def calc_out_(
    in_size: int, padding: int, dilation: int, kernel: int, stride: int
) -> int:
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def base_conv3d_channel_last(x, conv_layer, residual=None) -> torch.Tensor:
    in_numel = x.numel()
    out_numel = int(x.numel() * conv_layer.out_channels / conv_layer.in_channels)
    if (in_numel >= 2**30) or (out_numel >= 2**30):
        assert conv_layer.stride[0] == 1, "time split asks time stride = 1"

        B, T, H, W, C = x.shape
        K = conv_layer.kernel_size[0]

        chunks = 4
        chunk_size = T // chunks

        if residual is None:
            out_nhwc = base_conv3d(
                x,
                conv_layer,
                channel_last=True,
                residual=residual,
                only_return_output=True,
            )
        else:
            out_nhwc = residual

        assert B == 1
        for i in range(chunks):
            if i == chunks - 1:
                xi = x[:1, chunk_size * i :]
                out_nhwci = out_nhwc[:1, chunk_size * i :]
            else:
                xi = x[:1, chunk_size * i : chunk_size * (i + 1) + K - 1]
                out_nhwci = out_nhwc[:1, chunk_size * i : chunk_size * (i + 1)]
            if residual is not None:
                if i == chunks - 1:
                    ri = residual[:1, chunk_size * i :]
                else:
                    ri = residual[:1, chunk_size * i : chunk_size * (i + 1)]
            else:
                ri = None
            out_nhwci.copy_(base_conv3d(xi, conv_layer, channel_last=True, residual=ri))
    else:
        out_nhwc = base_conv3d(x, conv_layer, channel_last=True, residual=residual)
    return out_nhwc


class Upsample2D(nn.Module):

    def __init__(
        self, channels, use_conv=False, use_conv_transpose=False, out_channels=None
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        else:
            assert "Not Supported"
            self.conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)

    def forward(self, x, output_size=None) -> torch.Tensor:
        assert x.shape[-1] == self.channels

        if self.use_conv_transpose:
            return self.conv(x)

        if output_size is None:
            x = (
                F.interpolate(
                    x.permute(0, 3, 1, 2).to(memory_format=torch.channels_last),
                    scale_factor=2.0,
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        else:
            x = (
                F.interpolate(
                    x.permute(0, 3, 1, 2).to(memory_format=torch.channels_last),
                    size=output_size,
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        # x = self.conv(x)
        x = base_conv2d(x, self.conv, channel_last=True)
        return x


class Downsample2D(nn.Module):

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x) -> torch.Tensor:
        assert x.shape[-1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 0, 0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)

        assert x.shape[-1] == self.channels
        # x = self.conv(x)
        x = base_conv2d(x, self.conv, channel_last=True)
        return x


class CausalConv(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size, **kwargs) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (
                kernel_size if isinstance(kernel_size, tuple) else ((kernel_size,) * 3)
            )
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.dilation = kwargs.pop("dilation", 1)
        self.stride = kwargs.pop("stride", 1)
        if isinstance(self.stride, int):
            self.stride = (self.stride, 1, 1)
        time_pad = self.dilation * (time_kernel_size - 1) + max((1 - self.stride[0]), 0)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )
        self.time_uncausal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            0,
            0,
        )

        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            **kwargs,
        )
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.is_first_run = True

    def forward(self, x, is_init=True, residual=None) -> torch.Tensor:
        x = nn.functional.pad(
            x, self.time_causal_padding if is_init else self.time_uncausal_padding
        )
        x = self.conv(x)
        if residual is not None:
            x.add_(residual)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**3 % in_channels == 0
        self.repeats = out_channels * factor**3 // in_channels

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor,
            self.factor,
            self.factor,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor,
            x.size(4) * self.factor,
            x.size(6) * self.factor,
        )
        x = x[:, :, self.factor - 1 :, :, :]
        return x


class ConvPixelShuffleUpSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        self.conv = CausalConv(
            in_channels, out_channels * out_ratio, kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = self.conv(x, is_init)
        x = self.pixel_shuffle_3d(x, self.factor)
        return x

    @staticmethod
    def pixel_shuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.size()
        new_channels = channels // (factor**3)
        new_depth = depth * factor
        new_height = height * factor
        new_width = width * factor

        x = x.view(
            batch_size, new_channels, factor, factor, factor, depth, height, width
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, new_channels, new_depth, new_height, new_width)
        x = x[:, :, factor - 1 :, :, :]
        return x


class ConvPixelUnshuffleDownSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        assert out_channels % out_ratio == 0
        self.conv = CausalConv(
            in_channels, out_channels // out_ratio, kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = self.conv(x, is_init)
        x = self.pixel_unshuffle_3d(x, self.factor)
        return x

    @staticmethod
    def pixel_unshuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
        pad = (0, 0, 0, 0, factor - 1, 0)  # (left, right, top, bottom, front, back)
        x = F.pad(x, pad)
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // factor, factor, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * factor**3, D // factor, H // factor, W // factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**3 % out_channels == 0
        self.group_size = in_channels * factor**3 // out_channels

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        pad = (
            0,
            0,
            0,
            0,
            self.factor - 1,
            0,
        )  # (left, right, top, bottom, front, back)
        x = F.pad(x, pad)
        B, C, D, H, W = x.shape
        x = x.view(
            B,
            C,
            D // self.factor,
            self.factor,
            H // self.factor,
            self.factor,
            W // self.factor,
            self.factor,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B, C * self.factor**3, D // self.factor, H // self.factor, W // self.factor
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            D // self.factor,
            H // self.factor,
            W // self.factor,
        )
        x = x.mean(dim=2)
        return x


def base_group_norm_with_zero_pad(
    x, norm_layer, act_silu=True, pad_size=2
) -> torch.Tensor:
    out_shape = list(x.shape)
    out_shape[1] += pad_size
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out[:, pad_size:] = base_group_norm(
        x, norm_layer, act_silu=act_silu, channel_last=True
    )
    out[:, :pad_size] = 0
    return out


class CausalConvChannelLast(CausalConv):
    time_causal_padding: tuple[Any, ...]
    time_uncausal_padding: tuple[Any, ...]

    def __init__(self, chan_in, chan_out, kernel_size, **kwargs) -> None:
        super().__init__(chan_in, chan_out, kernel_size, **kwargs)

        self.time_causal_padding = (0, 0) + self.time_causal_padding
        self.time_uncausal_padding = (0, 0) + self.time_uncausal_padding

    def forward(self, x, is_init=True, residual=None) -> torch.Tensor:
        if self.is_first_run:
            self.is_first_run = False
            # self.conv.weight = nn.Parameter(self.conv.weight.permute(0,2,3,4,1).contiguous())

        x = nn.functional.pad(
            x, self.time_causal_padding if is_init else self.time_uncausal_padding
        )

        x = base_conv3d_channel_last(x, self.conv, residual=residual)
        return x


class CausalConvAfterNorm(CausalConv):

    def __init__(self, chan_in, chan_out, kernel_size, **kwargs) -> None:
        super().__init__(chan_in, chan_out, kernel_size, **kwargs)

        if self.time_causal_padding == (1, 1, 1, 1, 2, 0):
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=(0, 1, 1),
                **kwargs,
            )
        else:
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                **kwargs,
            )
        self.is_first_run = True

    def forward(self, x, is_init=True, residual=None) -> torch.Tensor:
        if self.is_first_run:
            self.is_first_run = False

        if self.time_causal_padding == (1, 1, 1, 1, 2, 0):
            pass
        else:
            x = nn.functional.pad(x, self.time_causal_padding).contiguous()

        x = base_conv3d_channel_last(x, self.conv, residual=residual)
        return x


class AttnBlock(nn.Module):

    def __init__(self, in_channels) -> None:
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.q = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.k = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.v = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.proj_out = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)

    def attention(self, x, is_init=True) -> torch.Tensor:
        x = base_group_norm(x, self.norm, act_silu=False, channel_last=True)
        q = self.q(x, is_init)
        k = self.k(x, is_init)
        v = self.v(x, is_init)

        b, t, h, w, c = q.shape
        q, k, v = map(lambda x: rearrange(x, "b t h w c -> b 1 (t h w) c"), (q, k, v))
        x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = rearrange(x, "b 1 (t h w) c -> b t h w c", t=t, h=h, w=w)

        return x

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        h = self.attention(x)
        x = self.proj_out(h, residual=x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class Resnet3DBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels=None,
        temb_channels=512,
        conv_shortcut=False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = CausalConvAfterNorm(in_channels, out_channels, kernel_size=3)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = CausalConvAfterNorm(out_channels, out_channels, kernel_size=3)

        assert conv_shortcut is False
        self.use_conv_shortcut = conv_shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConvAfterNorm(
                    in_channels, out_channels, kernel_size=3
                )
            else:
                self.nin_shortcut = CausalConvAfterNorm(
                    in_channels, out_channels, kernel_size=1
                )

    def forward(self, x, temb=None, is_init=True) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        h = base_group_norm_with_zero_pad(x, self.norm1, act_silu=True, pad_size=2)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nn.functional.silu(temb))[:, :, None, None]

        x = self.nin_shortcut(x) if self.in_channels != self.out_channels else x

        h = base_group_norm_with_zero_pad(h, self.norm2, act_silu=True, pad_size=2)
        x = self.conv2(h, residual=x)

        x = x.permute(0, 4, 1, 2, 3)
        return x


class Downsample3D(nn.Module):

    def __init__(self, in_channels, with_conv, stride) -> None:
        super().__init__()

        self.with_conv = with_conv
        if with_conv:
            self.conv = CausalConv(
                in_channels, in_channels, kernel_size=3, stride=stride
            )

    def forward(self, x, is_init=True) -> torch.Tensor:
        if self.with_conv:
            x = self.conv(x, is_init)
        else:
            x = nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class VideoEncoder(nn.Module):

    def __init__(
        self,
        ch=32,
        ch_mult=(4, 8, 16, 16),
        num_res_blocks=2,
        in_channels=3,
        z_channels=16,
        double_z=True,
        down_sampling_layer=(1, 2),
        resamp_with_conv=True,
        version=1,
    ) -> None:
        super().__init__()

        temb_ch = 0

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = CausalConv(in_channels, ch, kernel_size=3)
        self.down_sampling_layer = down_sampling_layer

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Resnet3DBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=temb_ch,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level in self.down_sampling_layer:
                    down.downsample = Downsample3D(
                        block_in, resamp_with_conv, stride=(2, 2, 2)
                    )
                else:
                    down.downsample = Downsample2D(
                        block_in, resamp_with_conv, padding=0
                    )  # DIFF
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch
        )

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in)
        self.version = version
        if version == 2:
            channels = 4 * z_channels * 2**3
            self.conv_patchify = ConvPixelUnshuffleDownSampleLayer3D(
                block_in, channels, kernel_size=3, factor=2
            )
            self.shortcut_pathify = PixelUnshuffleChannelAveragingDownSampleLayer3D(
                block_in, channels, 2
            )
            self.shortcut_out = PixelUnshuffleChannelAveragingDownSampleLayer3D(
                channels, 2 * z_channels if double_z else z_channels, 1
            )
            self.conv_out = CausalConvChannelLast(
                channels, 2 * z_channels if double_z else z_channels, kernel_size=3
            )
        else:
            self.conv_out = CausalConvAfterNorm(
                block_in, 2 * z_channels if double_z else z_channels, kernel_size=3
            )

    @torch.inference_mode()
    def forward(self, x, video_frame_num, is_init=True) -> torch.Tensor:
        # timestep embedding
        temb = None

        t = video_frame_num

        # downsampling
        h = self.conv_in(x, is_init)

        # make it real channel last, but behave like normal layout
        h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb, is_init)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                if isinstance(self.down[i_level].downsample, Downsample2D):
                    _, _, t, _, _ = h.shape
                    h = rearrange(h, "b c t h w -> (b t) h w c", t=t)
                    h = self.down[i_level].downsample(h)
                    h = rearrange(h, "(b t) h w c -> b c t h w", t=t)
                else:
                    h = self.down[i_level].downsample(h, is_init)

        h = self.mid.block_1(h, temb, is_init)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, is_init)

        h = h.permute(0, 2, 3, 4, 1).contiguous()  # b c l h w -> b l h w c
        if self.version == 2:
            h = base_group_norm(h, self.norm_out, act_silu=True, channel_last=True)
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            shortcut = self.shortcut_pathify(h, is_init)
            h = self.conv_patchify(h, is_init)
            h = h.add_(shortcut)
            shortcut = self.shortcut_out(h, is_init).permute(0, 2, 3, 4, 1)
            h = self.conv_out(h.permute(0, 2, 3, 4, 1).contiguous(), is_init)
            h = h.add_(shortcut)
        else:
            h = base_group_norm_with_zero_pad(
                h, self.norm_out, act_silu=True, pad_size=2
            )
            h = self.conv_out(h, is_init)
        h = h.permute(0, 4, 1, 2, 3)  # b l h w c -> b c l h w

        h = rearrange(h, "b c t h w -> b t c h w")
        return h


class Res3DBlockUpsample(nn.Module):

    def __init__(
        self, input_filters, num_filters, down_sampling_stride, down_sampling=False
    ) -> None:
        super().__init__()

        self.input_filters = input_filters
        self.num_filters = num_filters

        self.act_ = nn.SiLU(inplace=True)

        self.conv1 = CausalConvChannelLast(
            num_filters, num_filters, kernel_size=[3, 3, 3]
        )
        self.norm1 = nn.GroupNorm(32, num_filters)

        self.conv2 = CausalConvChannelLast(
            num_filters, num_filters, kernel_size=[3, 3, 3]
        )
        self.norm2 = nn.GroupNorm(32, num_filters)

        self.down_sampling = down_sampling
        if down_sampling:
            self.down_sampling_stride = down_sampling_stride
        else:
            self.down_sampling_stride = [1, 1, 1]

        if num_filters != input_filters or down_sampling:
            self.conv3 = CausalConvChannelLast(
                input_filters,
                num_filters,
                kernel_size=[1, 1, 1],
                stride=self.down_sampling_stride,
            )
            self.norm3 = nn.GroupNorm(32, num_filters)

    def forward(self, x, is_init=False) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        residual = x

        h = self.conv1(x, is_init)
        h = base_group_norm(h, self.norm1, act_silu=True, channel_last=True)

        h = self.conv2(h, is_init)
        h = base_group_norm(h, self.norm2, act_silu=False, channel_last=True)

        if self.down_sampling or self.num_filters != self.input_filters:
            x = self.conv3(x, is_init)
            x = base_group_norm(x, self.norm3, act_silu=False, channel_last=True)

        h.add_(x)
        h = self.act_(h)
        if residual is not None:
            h.add_(residual)

        h = h.permute(0, 4, 1, 2, 3)
        return h


class Upsample3D(nn.Module):

    def __init__(self, in_channels, scale_factor=2) -> None:
        super().__init__()

        self.scale_factor = scale_factor
        self.conv3d = Res3DBlockUpsample(
            input_filters=in_channels,
            num_filters=in_channels,
            down_sampling_stride=(1, 1, 1),
            down_sampling=False,
        )

    def forward(self, x, is_init=True, is_split=True) -> torch.Tensor:
        b, c, t, h, w = x.shape

        # x = x.permute(0,2,3,4,1).contiguous().permute(0,4,1,2,3).to(memory_format=torch.channels_last_3d)
        if is_split:
            split_size = c // 8
            x_slices = torch.split(x, split_size, dim=1)
            x = [
                nn.functional.interpolate(x, scale_factor=self.scale_factor)
                for x in x_slices
            ]
            x = torch.cat(x, dim=1)
        else:
            x = nn.functional.interpolate(x, scale_factor=self.scale_factor)

        x = self.conv3d(x, is_init)
        return x


class VideoDecoder(nn.Module):

    def __init__(
        self,
        ch=128,
        z_channels=16,
        out_channels=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temporal_up_layers=(2, 3),
        temporal_downsample=4,
        resamp_with_conv=True,
        version=1,
    ) -> None:
        super().__init__()

        temb_ch = 0

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temporal_downsample = temporal_downsample

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.version = version
        if version == 2:
            channels = 4 * z_channels * 2**3
            self.conv_in = CausalConv(z_channels, channels, kernel_size=3)
            self.shortcut_in = ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(
                z_channels, channels, 1
            )
            self.conv_unpatchify = ConvPixelShuffleUpSampleLayer3D(
                channels, block_in, kernel_size=3, factor=2
            )
            self.shortcut_unpathify = ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(
                channels, block_in, 2
            )
        else:
            self.conv_in = CausalConv(z_channels, block_in, kernel_size=3)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = Resnet3DBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=temb_ch
        )

        # upsampling
        self.up_id = len(temporal_up_layers)
        self.video_frame_num = 1
        self.cur_video_frame_num = self.video_frame_num // 2**self.up_id + 1
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Resnet3DBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=temb_ch,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level in temporal_up_layers:
                    up.upsample = Upsample3D(block_in)
                    self.cur_video_frame_num = self.cur_video_frame_num * 2
                else:
                    up.upsample = Upsample2D(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in)
        self.conv_out = CausalConvAfterNorm(block_in, out_channels, kernel_size=3)

    @torch.inference_mode()
    def forward(self, z, is_init=True) -> torch.Tensor:
        z = rearrange(z, "b t c h w -> b c t h w")
        h = self.conv_in(z, is_init=is_init)
        if self.version == 2:
            shortcut = self.shortcut_in(z, is_init=is_init)
            h = h.add_(shortcut)
            shortcut = self.shortcut_unpathify(h, is_init=is_init)
            h = self.conv_unpatchify(h, is_init=is_init)
            h = h.add_(shortcut)

        temb = None

        h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)
        h = self.mid.block_1(h, temb, is_init=is_init)
        h = self.mid.attn_1(h)
        h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)
        h = self.mid.block_2(h, temb, is_init=is_init)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)
                h = self.up[i_level].block[i_block](h, temb, is_init=is_init)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                if isinstance(self.up[i_level].upsample, Upsample2D):
                    B = h.size(0)
                    h = h.permute(0, 2, 3, 4, 1).flatten(0, 1)
                    h = self.up[i_level].upsample(h)
                    h = h.unflatten(0, (B, -1)).permute(0, 4, 1, 2, 3)
                else:
                    h = self.up[i_level].upsample(h, is_init=is_init)

        # end
        h = h.permute(0, 2, 3, 4, 1)  # b c l h w -> b l h w c
        h = base_group_norm_with_zero_pad(h, self.norm_out, act_silu=True, pad_size=2)
        h = self.conv_out(h)
        h = h.permute(0, 4, 1, 2, 3)

        if is_init:
            h = h[:, :, (self.temporal_downsample - 1) :]
        return h


def rms_norm(input, normalized_shape, eps=1e-6) -> torch.Tensor:
    dtype = input.dtype
    input = input.to(torch.float32)
    variance = (
        input.pow(2)
        .flatten(-len(normalized_shape))
        .mean(-1)[(...,) + (None,) * len(normalized_shape)]
    )
    input = input * torch.rsqrt(variance + eps)
    return input.to(dtype)


class DiagonalGaussianDistribution:

    def __init__(
        self,
        parameters,
        deterministic=False,
        rms_norm_mean=False,
        only_return_mean=False,
    ) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-3)  # N,[X],C,H,W
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.deterministic = deterministic
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )
        if rms_norm_mean:
            self.mean = rms_norm(self.mean, self.mean.size()[1:])
        self.only_return_mean = only_return_mean

    def sample(self, generator=None) -> torch.Tensor:
        # make sure sample is on the same device
        # as the parameters and has same dtype
        sample = torch.randn(
            self.mean.shape, generator=generator, device=self.parameters.device
        )
        sample = sample.to(dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        if self.only_return_mean:
            return self.mean
        else:
            return x


class AutoencoderKLStepvideo(nn.Module, ParallelTiledVAE):

    def __init__(
        self,
        config: StepVideoVAEConfig,
    ) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)

        self.frame_len = config.frame_len

        if config.version == 2:
            self.latent_len = 3
            base_group_norm.spatial = True  # type: ignore[attr-defined]
        else:
            self.latent_len = 5
            base_group_norm.spatial = False  # type: ignore[attr-defined]

        self.encoder = VideoEncoder(
            in_channels=config.in_channels,
            z_channels=config.z_channels,
            num_res_blocks=config.num_res_blocks,
            version=config.version,
        )

        self.decoder = VideoDecoder(
            z_channels=config.z_channels,
            out_channels=config.out_channels,
            num_res_blocks=config.num_res_blocks,
            version=config.version,
        )

        self.world_size = config.world_size
        # self.is_init = True

    def load_state_dict(self, state_dict, strict=True):
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("decoder.conv_out."):
                # move “decoder.conv_out.weight” → “decoder.conv_out.conv.weight”
                suffix = key[len("decoder.conv_out.") :]
                remapped[f"decoder.conv_out.conv.{suffix}"] = value
            else:
                remapped[key] = value
        super().load_state_dict(remapped, strict=strict)

    def _encode(self, x, is_init_image=True) -> torch.Tensor:
        # b, len, c, h, w = x.size()
        b, c, len, h, w = x.size()
        # x = rearrange(x, 'b l c h w -> b c l h w').contiguous()
        z = self.encoder(x, len, True)  # 下采样[1, 4, 8, 16, 16]
        return z

    @torch.inference_mode()
    def encode(self, x):
        # b (nc cf) c h w -> (b nc) cf c h w -> encode -> (b nc) cf c h w -> b (nc cf) c h w
        chunks = list(x.split(self.frame_len, dim=1))
        for i in range(len(chunks)):
            chunks[i] = self._encode(chunks[i], True)
        z = torch.cat(chunks, dim=1)

        posterior = DiagonalGaussianDistribution(z)
        return posterior.sample()

    def _decode(self, z) -> torch.Tensor:

        chunks = list(z.split(self.latent_len, dim=2))
        for i in range(len(chunks)):
            chunks[i] = chunks[i].permute(0, 2, 1, 3, 4)
            chunks[i] = chunks[i].to(next(self.decoder.parameters()).dtype)
            chunks[i] = self.decoder(chunks[i], is_init=True)
        x = torch.cat(chunks, dim=2)
        return x

    def decode(self, z) -> torch.Tensor:
        num_frames = z.size(2)
        dec = ParallelTiledVAE.decode(self, z).permute(0, 2, 1, 3, 4)
        dec = self.mix(dec).permute(0, 2, 1, 3, 4)
        num_sample_frames = num_frames // 3 * 17
        return dec[:, :, :num_sample_frames]

    def mix(self, x) -> torch.Tensor:
        remain_scale = 0.6
        mix_scale = 1.0 - remain_scale
        front = slice(self.frame_len - 1, x.size(1) - 1, self.frame_len)
        back = slice(self.frame_len, x.size(1), self.frame_len)
        x[:, back] = x[:, back] * remain_scale + x[:, front] * mix_scale
        x[:, front] = x[:, front] * remain_scale + x[:, back] * mix_scale
        return x

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec


EntryClass = AutoencoderKLStepvideo
