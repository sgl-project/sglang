# SPDX-License-Identifier: Apache-2.0
# 3D causal CNN encoder for the MiniMax H3 visual VAE (inference-only bundle).
import os

import torch.nn as nn
import torch.nn.functional as F

from .conv import BaseConv3d
from .norm import get_group_norm_3d, get_spatial_norm_3d

# ============================================================================
# 3D CNN Components
# ============================================================================


def norm_silu(x, norm, cond=None):
    if cond is None:
        return F.silu(norm(x), inplace=True)
    else:
        return F.silu(norm(x, cond), inplace=True)


class Downsample3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_stride=1,
        space_stride=2,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__()
        self.time_stride = time_stride
        self.space_stride = space_stride

        assert time_stride in [1, 2]
        assert space_stride in [1, 2, 3]

        self.conv = BaseConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )
        self.causal = self.conv.causal
        self.pad_mode = self.conv.pad_mode

    def forward(self, x):
        if self.space_stride == 2:
            pad = (0, 1, 0, 1, 0, 0)
            x = F.pad(x, pad, mode=self.pad_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        zq_ch=None,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
        use_t_isolated_gn=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.use_fused_norm = (
            os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true"
        )

        if zq_ch is None:
            self.norm1 = get_group_norm_3d(
                in_channels, use_t_isolated_gn=use_t_isolated_gn
            )
            self.norm2 = get_group_norm_3d(
                out_channels, use_t_isolated_gn=use_t_isolated_gn
            )
        else:
            self.norm1 = get_spatial_norm_3d(
                in_channels,
                zq_ch,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
                use_t_isolated_gn=use_t_isolated_gn,
            )
            self.norm2 = get_spatial_norm_3d(
                out_channels,
                zq_ch,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
                use_t_isolated_gn=use_t_isolated_gn,
            )

        self.conv1 = BaseConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

        self.conv2 = BaseConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = BaseConv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
            )

    def forward(self, x, zq=None):
        h = x

        if self.use_fused_norm:
            h = self.norm1(h, zq)
        else:
            h = norm_silu(h, self.norm1, zq)

        h = self.conv1(h)

        if self.use_fused_norm:
            h = self.norm2(h, zq)
        else:
            h = norm_silu(h, self.norm2, zq)

        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return h.add_(x)


class EncoderFCN3D(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult,
        space_down,
        time_down,
        num_res_blocks,
        in_channels,
        z_channels,
        double_z=False,
        zq_ch=None,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
        use_t_isolated_gn=False,
    ):
        super().__init__()
        self.ch = ch
        self.num_levels = len(ch_mult)

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * self.num_levels
        else:
            self.num_res_blocks = num_res_blocks

        self.space_down_factors = space_down
        self.time_down_factors = time_down
        self.in_channels = in_channels

        self.use_fused_norm = (
            os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true"
        )

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        conv_kwargs = dict(
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

        self.conv_in = BaseConv3d(
            in_channels, block_in[0], kernel_size=3, padding=1, **conv_kwargs
        )

        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()

            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(
                    ResnetBlock3D(
                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                        out_channels=block_mid[i_level],
                        zq_ch=zq_ch,
                        use_t_isolated_gn=use_t_isolated_gn,
                        **conv_kwargs,
                    )
                )

            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(
                    block_mid[i_level],
                    block_out[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                    **conv_kwargs,
                )
            else:
                if block_out[i_level] != block_mid[i_level]:
                    down.downsample = BaseConv3d(
                        block_mid[i_level],
                        block_out[i_level],
                        kernel_size=1,
                        **conv_kwargs,
                    )

            self.down.append(down)

        if zq_ch is None:
            self.norm_out = get_group_norm_3d(
                block_out[-1], use_t_isolated_gn=use_t_isolated_gn
            )
        else:
            self.norm_out = get_spatial_norm_3d(
                block_out[-1],
                zq_ch,
                use_t_isolated_gn=use_t_isolated_gn,
                **conv_kwargs,
            )

        self.conv_out = BaseConv3d(
            block_out[-1],
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=1,
            **conv_kwargs,
        )

    def forward(self, x, zq=None):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, zq)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        if self.use_fused_norm:
            h = self.norm_out(h, zq)
        else:
            h = norm_silu(h, self.norm_out, zq)

        h = self.conv_out(h)
        return h
