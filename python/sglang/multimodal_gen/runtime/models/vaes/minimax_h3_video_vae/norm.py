# SPDX-License-Identifier: Apache-2.0
# Torch-native normalization for the MiniMax H3 visual VAE.
import math

import torch.nn as nn
import torch.nn.functional as F

from .conv import BaseConv3d


def _merge_time_to_batch(x):
    batch, channels, depth, height, width = x.shape
    return (
        x.permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(batch * depth, channels, 1, height, width)
    )


def _split_time_from_batch(x, batch):
    batch_depth, channels, _, height, width = x.shape
    depth = batch_depth // batch
    return (
        x.view(batch, depth, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    def forward(self, input):
        if input.dim() == 5:
            batch = input.shape[0]
            input = _merge_time_to_batch(input)
            output = super().forward(input)
            return _split_time_from_batch(output, batch)
        return super().forward(input)


class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
        use_t_isolated_gn=False,
    ):
        super().__init__()
        norm_cls = TemporalIsolatedGroupNorm if use_t_isolated_gn else nn.GroupNorm
        self.norm_layer = norm_cls(
            num_groups=32, num_channels=f_channels, eps=1e-6, affine=True
        )

        self.conv_y = BaseConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )
        self.conv_b = BaseConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

    def forward(self, f, zq):
        f_size = f.shape[-3:]
        norm_f = self.norm_layer(f)
        scale = self.conv_y(zq)
        bias = self.conv_b(zq)

        if math.prod(scale.shape[-3:]) * math.prod(bias.shape[-3:]) > 1:
            scale = F.interpolate(scale, size=f_size, mode="nearest")
            bias = F.interpolate(bias, size=f_size, mode="nearest")

        return norm_f * scale + bias


def get_spatial_norm_3d(
    num_channels,
    cond_channels,
    *,
    padding_mode="zeros",
    padding_mode_t=None,
    causal=True,
    use_t_isolated_gn=False,
):
    return SpatialNorm3D(
        num_channels,
        cond_channels,
        padding_mode=padding_mode,
        padding_mode_t=padding_mode_t,
        causal=causal,
        use_t_isolated_gn=use_t_isolated_gn,
    )


def get_group_norm_3d(num_channels, use_t_isolated_gn=False):
    norm_cls = TemporalIsolatedGroupNorm if use_t_isolated_gn else nn.GroupNorm
    return norm_cls(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)
