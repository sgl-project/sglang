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

import contextvars
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange

from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.models.vaes.common import (
    DiagonalGaussianDistribution,
    ParallelTiledVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import (
    AvgDown3D,
    DupUp3D,
    WanCausalConv3d,
    WanRMS_norm,
    WanUpsample,
    attention_block_forward,
    bind_context,
    mid_block_forward,
    resample_forward,
    residual_block_forward,
    residual_down_block_forward,
    residual_up_block_forward,
    up_block_forward,
)
from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_dist_utils import (
    WanDistAttentionBlock,
    WanDistCausalConv3d,
    WanDistMidBlock,
    WanDistResample,
    WanDistResidualBlock,
    WanDistResidualDownBlock,
    WanDistResidualUpBlock,
    WanDistUpBlock,
    ensure_local_height,
    gather_and_trim_height,
    split_for_parallel_decode,
    split_for_parallel_encode,
)

CACHE_T = 2

is_first_frame = contextvars.ContextVar("is_first_frame", default=False)
feat_cache = contextvars.ContextVar("feat_cache", default=None)
feat_idx = contextvars.ContextVar("feat_idx", default=0)
first_chunk = contextvars.ContextVar("first_chunk", default=None)

bind_context(is_first_frame, feat_cache, feat_idx, CACHE_T, first_chunk)


@contextmanager
def forward_context(
    first_frame_arg=False, feat_cache_arg=None, feat_idx_arg=None, first_chunk_arg=None
):
    is_first_frame_token = is_first_frame.set(first_frame_arg)
    feat_cache_token = feat_cache.set(feat_cache_arg)
    feat_idx_token = feat_idx.set(feat_idx_arg)
    first_chunk_token = first_chunk.set(first_chunk_arg)
    try:
        yield
    finally:
        is_first_frame.reset(is_first_frame_token)
        feat_cache.reset(feat_cache_token)
        feat_idx.reset(feat_idx_token)
        first_chunk.reset(first_chunk_token)


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
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = WanCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return resample_forward(self, x)


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
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        return residual_block_forward(self, x)


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
        return attention_block_forward(self, x)


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
        return mid_block_forward(self, x)


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
        return residual_down_block_forward(self, x)


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
        use_parallel_encode: bool = False,
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
        self.use_parallel_encode = use_parallel_encode
        self.downsample_count = max(len(dim_mult) - 1, 0)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        world_size = 1
        if dist.is_initialized():
            world_size = get_sp_world_size()

        if use_parallel_encode and world_size > 1:
            CausalConv3d = WanDistCausalConv3d
            ResidualDownBlock = WanDistResidualDownBlock
            ResidualBlock = WanDistResidualBlock
            AttentionBlock = WanDistAttentionBlock
            Resample = WanDistResample
            MidBlock = WanDistMidBlock
        else:
            CausalConv3d = WanCausalConv3d
            ResidualDownBlock = WanResidualDownBlock
            ResidualBlock = WanResidualBlock
            AttentionBlock = WanAttentionBlock
            Resample = WanResample
            MidBlock = WanMidBlock

        # init block
        self.conv_in = CausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    ResidualDownBlock(
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
                    self.down_blocks.append(ResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(AttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(Resample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = MidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = CausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False
        self.world_size = 1
        self.rank = 0
        if dist.is_initialized():
            self.world_size = get_sp_world_size()
            self.rank = get_sp_parallel_rank()

    def forward(self, x):
        expected_local_height = None
        expected_height = None
        if self.use_parallel_encode and self.world_size > 1:
            x, expected_height, expected_local_height = split_for_parallel_encode(
                x, self.downsample_count, self.world_size, self.rank
            )

        _feat_cache = feat_cache.get()
        _feat_idx = feat_idx.get()
        if _feat_cache is not None:
            idx = _feat_idx
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and _feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        _feat_cache[idx][:, :, -1, :, :]
                        .unsqueeze(2)
                        .to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, _feat_cache[idx])
            _feat_cache[idx] = cache_x
            _feat_idx += 1
            feat_cache.set(_feat_cache)
            feat_idx.set(_feat_idx)
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            x = layer(x)

        ## middle
        if self.use_parallel_encode and self.world_size > 1:
            x = ensure_local_height(x, expected_local_height)
        x = self.mid_block(x)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        _feat_cache = feat_cache.get()
        _feat_idx = feat_idx.get()
        if _feat_cache is not None:
            idx = _feat_idx
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and _feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        _feat_cache[idx][:, :, -1, :, :]
                        .unsqueeze(2)
                        .to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, _feat_cache[idx])
            _feat_cache[idx] = cache_x
            _feat_idx += 1
            feat_cache.set(_feat_cache)
            feat_idx.set(_feat_idx)
        else:
            x = self.conv_out(x)

        if self.use_parallel_encode and self.world_size > 1:
            x = gather_and_trim_height(x, expected_height)
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
        return residual_up_block_forward(self, x)


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
        return up_block_forward(self, x)


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
        use_parallel_decode: bool = False,
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
        self.use_parallel_decode = use_parallel_decode
        self.upsample_count = 0

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        world_size = 1
        if dist.is_initialized():
            world_size = get_sp_world_size()

        if use_parallel_decode and world_size > 1:
            CausalConv3d = WanDistCausalConv3d
            MidBlock = WanDistMidBlock
            ResidualUpBlock = WanDistResidualUpBlock
            UpBlock = WanDistUpBlock
        else:
            CausalConv3d = WanCausalConv3d
            MidBlock = WanMidBlock
            ResidualUpBlock = WanResidualUpBlock
            UpBlock = WanUpBlock

        # init block
        self.conv_in = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = MidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.upsample_count = 0
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
                up_block = ResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = UpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)
            if up_flag:
                self.upsample_count += 1

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = CausalConv3d(out_dim, out_channels, 3, padding=1)

        self.gradient_checkpointing = False
        self.world_size = 1
        self.rank = 0
        if dist.is_initialized():
            self.world_size = get_sp_world_size()
            self.rank = get_sp_parallel_rank()

    def forward(self, x):
        expected_height = None
        if self.use_parallel_decode and self.world_size > 1:
            x, expected_height = split_for_parallel_decode(
                x, self.upsample_count, self.world_size, self.rank
            )

        ## conv1
        _feat_cache = feat_cache.get()
        _feat_idx = feat_idx.get()
        if _feat_cache is not None:
            idx = _feat_idx
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and _feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        _feat_cache[idx][:, :, -1, :, :]
                        .unsqueeze(2)
                        .to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, _feat_cache[idx])
            _feat_cache[idx] = cache_x
            _feat_idx += 1
            feat_cache.set(_feat_cache)
            feat_idx.set(_feat_idx)
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        _feat_cache = feat_cache.get()
        _feat_idx = feat_idx.get()
        if _feat_cache is not None:
            idx = _feat_idx
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and _feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        _feat_cache[idx][:, :, -1, :, :]
                        .unsqueeze(2)
                        .to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, _feat_cache[idx])
            _feat_cache[idx] = cache_x
            _feat_idx += 1
            feat_cache.set(_feat_cache)
            feat_idx.set(_feat_idx)
        else:
            x = self.conv_out(x)

        if self.use_parallel_decode and self.world_size > 1:
            x = gather_and_trim_height(x, expected_height)
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


class AutoencoderKLWan(ParallelTiledVAE):
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
        self.use_parallel_encode = getattr(config, "use_parallel_encode", False)
        self.use_parallel_decode = getattr(config, "use_parallel_decode", False)

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
                use_parallel_encode=self.use_parallel_encode,
            )
        self.quant_conv = WanCausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(self.z_dim, self.z_dim, 1)

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
                use_parallel_decode=self.use_parallel_decode,
            )

        self.use_feature_cache = config.use_feature_cache

    def clear_cache(self) -> None:

        def _count_conv3d(model) -> int:
            count = 0
            for m in model.modules():
                if isinstance(m, WanCausalConv3d) or isinstance(m, WanDistCausalConv3d):
                    count += 1
            return count

        if self.config.load_decoder:
            self._conv_num = _count_conv3d(self.decoder)
            self._conv_idx = 0
            self._feat_map = [None] * self._conv_num
        # cache encode
        if self.config.load_encoder:
            self._enc_conv_num = _count_conv3d(self.encoder)
            self._enc_conv_idx = 0
            self._enc_feat_map = [None] * self._enc_conv_num

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_feature_cache:
            self.clear_cache()
            if self.config.patch_size is not None:
                x = patchify(x, patch_size=self.config.patch_size)
            with forward_context(
                feat_cache_arg=self._enc_feat_map, feat_idx_arg=self._enc_conv_idx
            ):
                t = x.shape[2]
                iter_ = 1 + (t - 1) // 4
                for i in range(iter_):
                    feat_idx.set(0)
                    if i == 0:
                        out = self.encoder(x[:, :, :1, :, :])
                    else:
                        out_ = self.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :])
                        out = torch.cat([out, out_], 2)
            enc = self.quant_conv(out)
            mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
            enc = torch.cat([mu, logvar], dim=1)
            enc = DiagonalGaussianDistribution(enc)
            self.clear_cache()
        else:
            for block in self.encoder.down_blocks:
                if isinstance(block, WanResample) and block.mode == "downsample3d":
                    _padding = list(block.time_conv._padding)
                    _padding[4] = 2
                    block.time_conv._padding = tuple(_padding)
            enc = ParallelTiledVAE.encode(self, x)

        return enc

    def _encode(self, x: torch.Tensor, first_frame=False) -> torch.Tensor:
        with forward_context(first_frame_arg=first_frame):
            out = self.encoder(x)
        enc = self.quant_conv(out)
        mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
        enc = torch.cat([mu, logvar], dim=1)
        return enc

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        first_frame = x[:, :, 0, :, :].unsqueeze(2)
        first_frame = self._encode(first_frame, first_frame=True)

        enc = ParallelTiledVAE.tiled_encode(self, x)
        enc = enc[:, :, 1:]
        enc = torch.cat([first_frame, enc], dim=2)
        return enc

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        first_frame = x[:, :, 0, :, :].unsqueeze(2)
        first_frame = self._encode(first_frame, first_frame=True)

        enc = ParallelTiledVAE.spatial_tiled_encode(self, x)
        enc = enc[:, :, 1:]
        enc = torch.cat([first_frame, enc], dim=2)
        return enc

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_feature_cache:
            self.clear_cache()
            iter_ = z.shape[2]
            x = self.post_quant_conv(z)
            with forward_context(
                feat_cache_arg=self._feat_map, feat_idx_arg=self._conv_idx
            ):
                for i in range(iter_):
                    feat_idx.set(0)
                    if i == 0:
                        first_chunk.set(True)
                        out = self.decoder(x[:, :, i : i + 1, :, :])
                    else:
                        first_chunk.set(False)
                        out_ = self.decoder(x[:, :, i : i + 1, :, :])
                        out = torch.cat([out, out_], 2)

            if self.config.patch_size is not None:
                out = unpatchify(out, patch_size=self.config.patch_size)

            out = out.float()
            out = torch.clamp(out, min=-1.0, max=1.0)
            self.clear_cache()
        else:
            out = ParallelTiledVAE.decode(self, z)

        return out

    def _decode(self, z: torch.Tensor, first_frame=False) -> torch.Tensor:
        x = self.post_quant_conv(z)
        with forward_context(first_frame_arg=first_frame):
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
