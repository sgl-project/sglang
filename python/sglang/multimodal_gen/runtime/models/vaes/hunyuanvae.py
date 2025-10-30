# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers

# Copyright 2024 The Hunyuan Team, The HuggingFace Team and The sgl-diffusion Team. All rights reserved.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.vaes import HunyuanVAEConfig
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE


def prepare_causal_attention_mask(
    num_frames: int,
    height_width: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int | None = None,
) -> torch.Tensor:
    indices = torch.arange(1, num_frames + 1, dtype=torch.int32, device=device)
    indices_blocks = indices.repeat_interleave(height_width)
    x, y = torch.meshgrid(indices_blocks, indices_blocks, indexing="xy")
    mask = torch.where(x <= y, 0, -float("inf")).to(dtype=dtype)

    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class HunyuanVAEAttention(nn.Module):

    def __init__(
        self, in_channels, heads, dim_head, eps, norm_num_groups, bias
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.norm_num_groups = norm_num_groups
        self.bias = bias

        inner_dim = heads * dim_head

        # Define the projection layers
        self.to_q = nn.Linear(in_channels, inner_dim, bias=bias)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=bias)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, in_channels, bias=bias))

        # Optional normalization layers
        self.group_norm = nn.GroupNorm(
            norm_num_groups, in_channels, eps=eps, affine=True
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Project to query, key, value
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape for multi-head attention
        head_dim = self.dim_head

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Perform scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = self.to_out(hidden_states)

        # Residual connection and rescale
        hidden_states = hidden_states + residual

        return hidden_states


class HunyuanVideoCausalConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(
            hidden_states, self.time_causal_padding, mode=self.pad_mode
        )
        return self.conv(hidden_states)


class HunyuanVideoUpsampleCausal3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        upsample_factor: tuple[int, ...] = (2, 2, 2),
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        self.upsample_factor = upsample_factor

        self.conv = HunyuanVideoCausalConv3d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_frames = hidden_states.size(2)

        first_frame, other_frames = hidden_states.split((1, num_frames - 1), dim=2)
        first_frame = F.interpolate(
            first_frame.squeeze(2),
            scale_factor=self.upsample_factor[1:],
            mode="nearest",
        ).unsqueeze(2)

        if num_frames > 1:
            # See: https://github.com/pytorch/pytorch/issues/81665
            # Unless you have a version of pytorch where non-contiguous implementation of F.interpolate
            # is fixed, this will raise either a runtime error, or fail silently with bad outputs.
            # If you are encountering an error here, make sure to try running encoding/decoding with
            # `vae.enable_tiling()` first. If that doesn't work, open an issue at:
            # https://github.com/huggingface/diffusers/issues
            other_frames = other_frames.contiguous()
            other_frames = F.interpolate(
                other_frames, scale_factor=self.upsample_factor, mode="nearest"
            )
            hidden_states = torch.cat((first_frame, other_frames), dim=2)
        else:
            hidden_states = first_frame

        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoDownsampleCausal3D(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        padding: int = 1,
        kernel_size: int = 3,
        bias: bool = True,
        stride=2,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels

        self.conv = HunyuanVideoCausalConv3d(
            channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoResnetBlockCausal3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_act_fn(non_linearity)

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = HunyuanVideoCausalConv3d(in_channels, out_channels, 3, 1, 0)

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = HunyuanVideoCausalConv3d(out_channels, out_channels, 3, 1, 0)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = HunyuanVideoCausalConv3d(
                in_channels, out_channels, 1, 1, 0
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual
        return hidden_states


class HunyuanVideoMidBlock3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        # There is always at least one resnet
        resnets = [
            HunyuanVideoResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
        ]
        attentions: list[HunyuanVAEAttention | None] = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    HunyuanVAEAttention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        bias=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                self.resnets[0], hidden_states
            )

            for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
                if attn is not None:
                    batch_size, num_channels, num_frames, height, width = (
                        hidden_states.shape
                    )
                    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                    attention_mask = prepare_causal_attention_mask(
                        num_frames,
                        height * width,
                        hidden_states.dtype,
                        hidden_states.device,
                        batch_size=batch_size,
                    )
                    hidden_states = attn(hidden_states, attention_mask=attention_mask)
                    hidden_states = hidden_states.unflatten(
                        1, (num_frames, height, width)
                    ).permute(0, 4, 1, 2, 3)

                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)

        else:
            hidden_states = self.resnets[0](hidden_states)

            for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
                if attn is not None:
                    batch_size, num_channels, num_frames, height, width = (
                        hidden_states.shape
                    )
                    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                    attention_mask = prepare_causal_attention_mask(
                        num_frames,
                        height * width,
                        hidden_states.dtype,
                        hidden_states.device,
                        batch_size=batch_size,
                    )
                    hidden_states = attn(hidden_states, attention_mask=attention_mask)
                    hidden_states = hidden_states.unflatten(
                        1, (num_frames, height, width)
                    ).permute(0, 4, 1, 2, 3)

                hidden_states = resnet(hidden_states)

        return hidden_states


class HunyuanVideoDownBlock3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_stride: tuple[int, ...] | int = 2,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    HunyuanVideoDownsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        stride=downsample_stride,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for resnet in self.resnets:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class HunyuanVideoUpBlock3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        upsample_scale_factor: tuple[int, ...] = (2, 2, 2),
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    HunyuanVideoUpsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        upsample_factor=upsample_scale_factor,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for resnet in self.resnets:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)

        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class HunyuanVideoEncoder3D(nn.Module):
    r"""
    Causal encoder for 3D video-like data introduced in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.conv_in = HunyuanVideoCausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1
        )
        self.mid_block: HunyuanVideoMidBlock3D | None = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            if down_block_type != "HunyuanVideoDownBlock3D":
                raise ValueError(f"Unsupported down_block_type: {down_block_type}")

            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(temporal_compression_ratio))

            if temporal_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers)
                    and not is_final_block
                )
            elif temporal_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {temporal_compression_ratio}"
                )

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)

            down_block = HunyuanVideoDownBlock3D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_stride=downsample_stride,
                downsample_padding=0,
            )

            self.down_blocks.append(down_block)

        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = HunyuanVideoCausalConv3d(
            block_out_channels[-1], conv_out_channels, kernel_size=3
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    down_block, hidden_states
                )

            hidden_states = self._gradient_checkpointing_func(
                self.mid_block, hidden_states
            )
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)
            assert self.mid_block is not None
            hidden_states = self.mid_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class HunyuanVideoDecoder3D(nn.Module):
    r"""
    Causal decoder for 3D video-like data introduced in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = HunyuanVideoCausalConv3d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1
        )
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            if up_block_type != "HunyuanVideoUpBlock3D":
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(
                    i >= len(block_out_channels) - 1 - num_time_upsample_layers
                    and not is_final_block
                )
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {time_compression_ratio}"
                )

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(
                upsample_scale_factor_T + upsample_scale_factor_HW
            )

            up_block = HunyuanVideoUpBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = HunyuanVideoCausalConv3d(
            block_out_channels[0], out_channels, kernel_size=3
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                self.mid_block, hidden_states
            )

            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    up_block, hidden_states
                )
        else:
            hidden_states = self.mid_block(hidden_states)

            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states)

        # post-process
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class AutoencoderKLHunyuanVideo(nn.Module, ParallelTiledVAE):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [HunyuanVideo](https://huggingface.co/papers/2412.03603).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        config: HunyuanVAEConfig,
    ) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)

        # TODO(will): only pass in config. We do this by manually defining a
        # config for hunyuan vae
        self.block_out_channels = config.block_out_channels

        if config.load_encoder:
            self.encoder = HunyuanVideoEncoder3D(
                in_channels=config.in_channels,
                out_channels=config.latent_channels,
                down_block_types=config.down_block_types,
                block_out_channels=config.block_out_channels,
                layers_per_block=config.layers_per_block,
                norm_num_groups=config.norm_num_groups,
                act_fn=config.act_fn,
                double_z=True,
                mid_block_add_attention=config.mid_block_add_attention,
                temporal_compression_ratio=config.temporal_compression_ratio,
                spatial_compression_ratio=config.spatial_compression_ratio,
            )
            self.quant_conv = nn.Conv3d(
                2 * config.latent_channels, 2 * config.latent_channels, kernel_size=1
            )

        if config.load_decoder:
            self.decoder = HunyuanVideoDecoder3D(
                in_channels=config.latent_channels,
                out_channels=config.out_channels,
                up_block_types=config.up_block_types,
                block_out_channels=config.block_out_channels,
                layers_per_block=config.layers_per_block,
                norm_num_groups=config.norm_num_groups,
                act_fn=config.act_fn,
                time_compression_ratio=config.temporal_compression_ratio,
                spatial_compression_ratio=config.spatial_compression_ratio,
                mid_block_add_attention=config.mid_block_add_attention,
            )
            self.post_quant_conv = nn.Conv3d(
                config.latent_channels, config.latent_channels, kernel_size=1
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        enc = self.quant_conv(x)
        return enc

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec


EntryClass = AutoencoderKLHunyuanVideo
