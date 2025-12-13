# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.models.vaes.common import (
    DiagonalGaussianDistribution,
    ParallelTiledVAE,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

CACHE_T = 2


class AvgDown3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels
        self.first_chunk = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )

        if self.first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
            self.first_chunk = False

        return x


class CausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pad_t = self.padding[0] * 2 if self.padding[0] > 0 else 1
        self.padding = (0, *self.padding[1:])
        self.register_buffer("prev_cache", None, False)

    def clear_cache(self) -> None:
        if isinstance(self.prev_cache, torch.Tensor):
            self.prev_cache = None


class WanCausalConv3d(CausalConv3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if self.prev_cache is None:
            self.prev_cache = x.new_zeros((b, c, self.pad_t, h, w))
        x_with_cache = torch.cat([self.prev_cache, x], dim=2)
        x_with_cache = (
            x_with_cache.to(self.weight.dtype)
            if current_platform.is_mps()
            else x_with_cache
        )  # casting needed for mps since amp isn't supported
        x = super().forward(x_with_cache)
        self.prev_cache = x_with_cache.narrow(2, t, self.pad_t)
        return x


class WanCausalEncodeTimeConv3d(CausalConv3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape

        if self.prev_cache is None:
            self.prev_cache = x
            return x
        x_with_cache = torch.cat([self.prev_cache, x], dim=2)
        x_with_cache = (
            x_with_cache.to(self.weight.dtype)
            if current_platform.is_mps()
            else x_with_cache
        )  # casting needed for mps since amp isn't supported
        x = super().forward(x_with_cache)
        self.prev_cache = x_with_cache.narrow(2, t, self.pad_t)
        return x


class WanCausalDecodeTimeConv3d(CausalConv3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if self.prev_cache is None:
            self.prev_cache = x.new_zeros((b, c, self.pad_t, h, w))
            return x
        x_with_cache = torch.cat([self.prev_cache, x], dim=2)
        x_with_cache = (
            x_with_cache.to(self.weight.dtype)
            if current_platform.is_mps()
            else x_with_cache
        )  # casting needed for mps since amp isn't supported
        x = super().forward(x_with_cache)
        x = rearrange(x, "b (r c) t h w -> b c (t r) h w", r=2)
        self.prev_cache = x_with_cache.narrow(2, t, self.pad_t)
        return x


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class WanUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalDecodeTimeConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
            )

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = WanCausalEncodeTimeConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            x = self.time_conv(x)
        t = x.size(2)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d":
            x = self.time_conv(x)
        return x


class WanResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_act_fn(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        # First conv
        x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        # Second conv
        x = self.conv2(x)

        # Add residual connection
        return x + h


class WanAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        # layers
        self.norm = WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = (
            x.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch_size * time, channels, height, width)
        )

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x):
        # First residual block
        x = self.resnets[0](x)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            if attn is not None:
                x = attn(x)

            x = resnet(x)

        return x


class WanResidualDownBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        num_res_blocks,
        temperal_downsample=False,
        down_flag=False,
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)

        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        dim_mult = list(dim_mult)
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)
        self.nonlinearity = get_act_fn(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=(
                            temperal_downsample[i] if i != len(dim_mult) - 1 else False
                        ),
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        ## conv_in
        x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            x = layer(x)

        ## middle
        x = self.mid_block(x)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        ## conv_out
        x = self.conv_out(x)
        return x


# adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
class WanResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x):
        """
        Forward pass through the upsampling block.
        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if self.avg_shortcut is not None:
            x_copy = x.clone()

        for resnet in self.resnets:
            x = resnet(x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy)

        return x


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class WanDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=(False, True, True),
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        dim_mult = list(dim_mult)
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_upsample = list(temperal_upsample)

        self.nonlinearity = get_act_fn(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            # Create and add the upsampling block
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        ## conv1
        x = self.conv_in(x)

        ## middle
        x = self.mid_block(x)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        # conv_out
        x = self.conv_out(x)
        return x


def patchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )

    return x


class AutoencoderKLWan(nn.Module, ParallelTiledVAE):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].
    """

    _supports_gradient_checkpointing = False

    def __init__(
        self,
        config: WanVAEConfig,
    ) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)

        self.z_dim = config.z_dim
        self.temperal_downsample = list(config.temperal_downsample)
        self.temperal_upsample = list(config.temperal_downsample)[::-1]

        if config.decoder_base_dim is None:
            decoder_base_dim = config.base_dim
        else:
            decoder_base_dim = config.decoder_base_dim

        self.latents_mean = list(config.latents_mean)
        self.latents_std = list(config.latents_std)
        self.shift_factor = config.shift_factor

        if config.load_encoder:
            self.encoder = WanEncoder3d(
                in_channels=config.in_channels,
                dim=config.base_dim,
                z_dim=self.z_dim * 2,
                dim_mult=config.dim_mult,
                num_res_blocks=config.num_res_blocks,
                attn_scales=config.attn_scales,
                temperal_downsample=self.temperal_downsample,
                dropout=config.dropout,
                is_residual=config.is_residual,
            )
        self.quant_conv = nn.Conv3d(self.z_dim * 2, self.z_dim * 2, 1)
        self.post_quant_conv = nn.Conv3d(self.z_dim, self.z_dim, 1)

        if config.load_decoder:
            self.decoder = WanDecoder3d(
                dim=decoder_base_dim,
                z_dim=self.z_dim,
                dim_mult=config.dim_mult,
                num_res_blocks=config.num_res_blocks,
                attn_scales=config.attn_scales,
                temperal_upsample=self.temperal_upsample,
                dropout=config.dropout,
                out_channels=config.out_channels,
                is_residual=config.is_residual,
            )

        self.use_feature_cache = config.use_feature_cache

    def clear_cache(self) -> None:
        # cache decode
        if self.config.load_decoder:
            for m in self.decoder.modules():
                if isinstance(m, CausalConv3d):
                    m.clear_cache()
                elif isinstance(m, DupUp3D):
                    m.first_chunk = True

        # cache encode
        if self.config.load_encoder:
            for m in self.encoder.modules():
                if isinstance(m, CausalConv3d):
                    m.clear_cache()
                elif isinstance(m, DupUp3D):
                    m.first_chunk = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_feature_cache:
            self.clear_cache()
            if self.config.patch_size is not None:
                x = patchify(x, patch_size=self.config.patch_size)
            t = x.shape[2]
            iter_ = 1 + (t - 1) // 4
            out = []
            for i in range(iter_):
                if i == 0:
                    out_ = self.encoder(x[:, :, :1, :, :])
                else:
                    out_ = self.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :])
                out.append(out_)
            out = torch.cat(out, 2)
            enc = self.quant_conv(out)
            mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
            enc = torch.cat([mu, logvar], dim=1)
            enc = DiagonalGaussianDistribution(enc)
        else:
            enc = ParallelTiledVAE.encode(self, x)

        return enc

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        self.clear_cache()
        out = self.encoder(x)
        enc = self.quant_conv(out)
        mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
        enc = torch.cat([mu, logvar], dim=1)
        return enc

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = ParallelTiledVAE.tiled_encode(self, x)
        return enc

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = ParallelTiledVAE.spatial_tiled_encode(self, x)
        return enc

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_feature_cache:
            self.clear_cache()
            iter_ = z.shape[2]
            x = self.post_quant_conv(z)
            out = []
            for i in range(iter_):
                if i == 0:
                    out_ = self.decoder(x[:, :, i : i + 1, :, :])
                else:
                    out_ = self.decoder(x[:, :, i : i + 1, :, :])
                out.append(out_)
            out = torch.cat(out, 2)
            if self.config.patch_size is not None:
                out = unpatchify(out, patch_size=self.config.patch_size)

            out = out.float()
            out = torch.clamp(out, min=-1.0, max=1.0)
        else:
            out = ParallelTiledVAE.decode(self, z)

        return out

    def _decode(self, z: torch.Tensor, first_frame=False) -> torch.Tensor:
        x = self.post_quant_conv(z)
        out = self.decoder(x)
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        self.blend_num_frames *= 2
        dec = ParallelTiledVAE.tiled_decode(self, z)
        start_frame_idx = self.temporal_compression_ratio - 1
        dec = dec[:, :, start_frame_idx:]
        return dec

    def spatial_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        dec = ParallelTiledVAE.spatial_tiled_decode(self, z)
        start_frame_idx = self.temporal_compression_ratio - 1
        dec = dec[:, :, start_frame_idx:]
        return dec

    def parallel_tiled_decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        self.blend_num_frames *= 2
        dec = ParallelTiledVAE.parallel_tiled_decode(self, z)
        start_frame_idx = self.temporal_compression_ratio - 1
        dec = dec[:, :, start_frame_idx:]
        return dec

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


EntryClass = AutoencoderKLWan
