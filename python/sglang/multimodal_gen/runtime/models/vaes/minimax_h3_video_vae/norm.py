# SPDX-License-Identifier: Apache-2.0
# Torch-native normalization for the MiniMax H3 visual VAE.
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import BaseConv3d


def _validate_activation(activation):
    valid_activations = {"identity", "silu", "relu"}
    if activation not in valid_activations:
        raise ValueError(
            f"Unsupported activation: {activation}. Supported: {valid_activations}"
        )


def _apply_activation(x, activation):
    _validate_activation(activation)
    if activation == "identity":
        return x
    if activation == "silu":
        return F.silu(x)
    return F.relu(x)


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


def fused_group_norm(x, num_groups, weight, bias, eps=1e-5, activation="silu"):
    out = F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)
    return _apply_activation(out, activation)


def fused_spatial_norm(
    f,
    num_groups,
    norm_weight,
    norm_bias,
    dynamic_scale,
    dynamic_bias,
    eps=1e-5,
    activation="silu",
):
    norm_f = F.group_norm(
        f,
        num_groups,
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
    )
    out = norm_f * dynamic_scale + dynamic_bias
    return _apply_activation(out, activation)


class DummyAffine(torch.nn.Module):
    def __init__(self, num_channels, affine=True):
        super().__init__()
        if affine:
            self.weight = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input):
        if self.weight is None:
            return input
        shape = [1, -1] + [1] * (input.dim() - 2)
        return input * self.weight.view(*shape) + self.bias.view(*shape)


class FusedGroupNorm3D(torch.nn.Module):
    """Compatibility wrapper implemented with native PyTorch ops."""

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True,
        activation="silu",
        cond_channels=None,
        use_t_isolated_gn=False,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__()
        _validate_activation(activation)
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.activation = activation
        self.use_t_isolated_gn = use_t_isolated_gn

        if cond_channels is not None:
            self.use_spatial_affine = True
            self.norm_layer = DummyAffine(num_channels, affine=affine)
            self.conv_y = BaseConv3d(
                cond_channels,
                num_channels,
                kernel_size=1,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
            )
            self.conv_b = BaseConv3d(
                cond_channels,
                num_channels,
                kernel_size=1,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
            )
        else:
            self.use_spatial_affine = False
            if self.affine:
                self.weight = torch.nn.Parameter(torch.ones(num_channels))
                self.bias = torch.nn.Parameter(torch.zeros(num_channels))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

    def forward(self, f, cond=None):
        need_reshape = self.use_t_isolated_gn and f.dim() == 5
        batch = f.shape[0] if need_reshape else None
        f_size = f.shape[-3:]
        if need_reshape:
            f = _merge_time_to_batch(f)

        if self.use_spatial_affine:
            scale = self.conv_y(cond)
            bias = self.conv_b(cond)
            if math.prod(scale.shape[-3:]) * math.prod(bias.shape[-3:]) > 1:
                scale = F.interpolate(scale, size=f_size, mode="nearest")
                bias = F.interpolate(bias, size=f_size, mode="nearest")
            if need_reshape:
                scale = _merge_time_to_batch(scale)
                bias = _merge_time_to_batch(bias)
            out = fused_spatial_norm(
                f,
                self.num_groups,
                self.norm_layer.weight,
                self.norm_layer.bias,
                scale,
                bias,
                self.eps,
                self.activation,
            )
        else:
            if cond is not None:
                raise NotImplementedError("Dynamic affine is not defined")
            weight = self.weight if self.affine else None
            bias = self.bias if self.affine else None
            out = fused_group_norm(
                f, self.num_groups, weight, bias, self.eps, self.activation
            )

        if need_reshape:
            out = _split_time_from_batch(out, batch)
        return out


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
    if os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true":
        return FusedGroupNorm3D(
            num_groups=32,
            num_channels=num_channels,
            eps=1e-6,
            affine=True,
            cond_channels=cond_channels,
            use_t_isolated_gn=use_t_isolated_gn,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )
    return SpatialNorm3D(
        num_channels,
        cond_channels,
        padding_mode=padding_mode,
        padding_mode_t=padding_mode_t,
        causal=causal,
        use_t_isolated_gn=use_t_isolated_gn,
    )


def get_group_norm_3d(num_channels, use_t_isolated_gn=False):
    if os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true":
        return FusedGroupNorm3D(
            num_groups=32,
            num_channels=num_channels,
            eps=1e-6,
            affine=True,
            use_t_isolated_gn=use_t_isolated_gn,
        )

    norm_cls = TemporalIsolatedGroupNorm if use_t_isolated_gn else nn.GroupNorm
    return norm_cls(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)
