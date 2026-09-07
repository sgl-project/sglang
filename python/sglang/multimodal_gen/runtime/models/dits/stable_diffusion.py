# SPDX-License-Identifier: Apache-2.0
"""Native Stable Diffusion 2.1 UNet blocks used by Hunyuan3D texture models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)


@dataclass(frozen=True)
class StableDiffusionUNetConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    center_input_sample: bool
    flip_sin_to_cos: bool
    freq_shift: float
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    downsample_padding: int
    dropout: float
    norm_num_groups: int
    norm_eps: float
    cross_attention_dim: int
    attention_head_dim: tuple[int, ...]
    transformer_layers_per_block: int
    use_linear_projection: bool

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> StableDiffusionUNetConfig:
        block_out_channels = tuple(config["block_out_channels"])
        attention_head_dim_value = config["attention_head_dim"]
        attention_head_dim = (
            (attention_head_dim_value,) * len(block_out_channels)
            if isinstance(attention_head_dim_value, int)
            else tuple(attention_head_dim_value)
        )
        parsed = cls(
            sample_size=int(config["sample_size"]),
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            center_input_sample=bool(config.get("center_input_sample", False)),
            flip_sin_to_cos=bool(config.get("flip_sin_to_cos", True)),
            freq_shift=float(config.get("freq_shift", 0.0)),
            down_block_types=tuple(config["down_block_types"]),
            up_block_types=tuple(config["up_block_types"]),
            block_out_channels=block_out_channels,
            layers_per_block=int(config["layers_per_block"]),
            downsample_padding=int(config.get("downsample_padding", 1)),
            dropout=float(config.get("dropout", 0.0)),
            norm_num_groups=int(config["norm_num_groups"]),
            norm_eps=float(config["norm_eps"]),
            cross_attention_dim=int(config["cross_attention_dim"]),
            attention_head_dim=attention_head_dim,
            transformer_layers_per_block=int(
                config.get("transformer_layers_per_block", 1)
            ),
            use_linear_projection=bool(config.get("use_linear_projection", False)),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        expected_down = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        )
        expected_up = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        )
        if self.down_block_types != expected_down or self.up_block_types != expected_up:
            raise ValueError(
                "The native SD2 UNet currently supports only the Hunyuan3D "
                "four-level SD2.1 block layout."
            )
        if len(self.block_out_channels) != 4 or len(self.attention_head_dim) != 4:
            raise ValueError("Hunyuan3D SD2.1 UNet requires four channel stages.")
        if self.layers_per_block != 2 or self.transformer_layers_per_block != 1:
            raise ValueError(
                "Hunyuan3D SD2.1 UNet requires two ResNet layers and one "
                "transformer layer per block."
            )
        if not self.use_linear_projection:
            raise ValueError("Hunyuan3D SD2.1 checkpoints require linear projection.")


@dataclass
class StableDiffusionUNetOutput:
    sample: torch.Tensor


def timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    embedding = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat(
            [embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1
        )
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, embedding_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class StableDiffusionAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        context_dim = cross_attention_dim or query_dim
        self.heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(0.0)])

    def _prepare_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.ndim == 2:
            return attention_mask[:, None, None, :]
        if attention_mask.ndim == 3:
            return attention_mask[:, None, :, :]
        if attention_mask.ndim == 4:
            return attention_mask
        raise ValueError(
            f"Expected a 2D, 3D, or 4D attention mask, got {attention_mask.ndim}D."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states).view(batch_size, -1, self.heads, self.head_dim)
        key = self.to_k(context).view(batch_size, -1, self.heads, self.head_dim)
        value = self.to_v(context).view(batch_size, -1, self.heads, self.head_dim)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=self._prepare_mask(attention_mask),
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(
            batch_size, -1, self.heads * self.head_dim
        )
        return self.to_out[1](self.to_out[0](output))


class GEGLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner_dim = dim * 4
        self.net = nn.ModuleList(
            [GEGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_heads
        self.attention_head_dim = head_dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn1 = StableDiffusionAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.attn2 = StableDiffusionAttention(
            dim, num_heads, head_dim, cross_attention_dim
        )
        self.norm3 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None and cross_attention_kwargs:
            unsupported = set(cross_attention_kwargs) - {"scale"}
            if unsupported:
                raise ValueError(
                    "Unsupported native SD2 cross-attention arguments: "
                    f"{sorted(unsupported)}"
                )
        hidden_states = hidden_states + self.attn1(
            self.norm1(hidden_states), attention_mask=attention_mask
        )
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        return hidden_states + self.ff(self.norm3(hidden_states))


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        cross_attention_dim: int,
        norm_num_groups: int,
    ) -> None:
        super().__init__()
        head_dim = channels // num_heads
        self.norm = nn.GroupNorm(norm_num_groups, channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, num_heads, head_dim, cross_attention_dim)]
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        batch_size, channels, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels
        )
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, height, width, channels
        ).permute(0, 3, 1, 2)
        return hidden_states.contiguous() + residual


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        norm_num_groups: int,
        norm_eps: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(
            norm_num_groups, in_channels, eps=norm_eps, affine=True
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(
            norm_num_groups, out_channels, eps=norm_eps, affine=True
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(
        self, hidden_states: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.nonlinearity(self.norm1(hidden_states)))
        time_states = self.time_emb_proj(self.nonlinearity(time_embedding))
        hidden_states = hidden_states + time_states[:, :, None, None]
        hidden_states = self.conv2(
            self.dropout(self.nonlinearity(self.norm2(hidden_states)))
        )
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.conv.padding == (0, 0):
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1))
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self, hidden_states: torch.Tensor, output_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        if output_size is None:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=2.0, mode="nearest"
            )
        else:
            hidden_states = F.interpolate(
                hidden_states, size=output_size, mode="nearest"
            )
        return self.conv(hidden_states)


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        norm_num_groups: int,
        norm_eps: float,
        dropout: float,
        add_downsample: bool,
        downsample_padding: int,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    time_embedding_dim,
                    norm_num_groups,
                    norm_eps,
                    dropout,
                )
                for index in range(2)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels, downsample_padding)])
            if add_downsample
            else None
        )

    def forward(
        self, hidden_states: torch.Tensor, time_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, time_embedding)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_heads: int,
        cross_attention_dim: int,
        norm_num_groups: int,
        norm_eps: float,
        dropout: float,
        add_downsample: bool,
        downsample_padding: int,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    time_embedding_dim,
                    norm_num_groups,
                    norm_eps,
                    dropout,
                )
                for index in range(2)
            ]
        )
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(
                    out_channels,
                    num_heads,
                    cross_attention_dim,
                    norm_num_groups,
                )
                for _ in range(2)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels, downsample_padding)])
            if add_downsample
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        cross_attention_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states: tuple[torch.Tensor, ...] = ()
        for resnet, attention in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, time_embedding)
            hidden_states = attention(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        num_heads: int,
        cross_attention_dim: int,
        norm_num_groups: int,
        norm_eps: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    channels,
                    channels,
                    time_embedding_dim,
                    norm_num_groups,
                    norm_eps,
                    dropout,
                )
                for _ in range(2)
            ]
        )
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(
                    channels,
                    num_heads,
                    cross_attention_dim,
                    norm_num_groups,
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        cross_attention_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, time_embedding)
        hidden_states = self.attentions[0](
            hidden_states,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return self.resnets[1](hidden_states, time_embedding)


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        previous_output_channels: int,
        time_embedding_dim: int,
        norm_num_groups: int,
        norm_eps: float,
        dropout: float,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        resnets = []
        for index in range(3):
            skip_channels = in_channels if index == 2 else out_channels
            hidden_channels = previous_output_channels if index == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    hidden_channels + skip_channels,
                    out_channels,
                    time_embedding_dim,
                    norm_num_groups,
                    norm_eps,
                    dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual_states: tuple[torch.Tensor, ...],
        time_embedding: torch.Tensor,
        upsample_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            residual = residual_states[-1]
            residual_states = residual_states[:-1]
            hidden_states = resnet(
                torch.cat([hidden_states, residual], dim=1), time_embedding
            )
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states, upsample_size)
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        previous_output_channels: int,
        time_embedding_dim: int,
        num_heads: int,
        cross_attention_dim: int,
        norm_num_groups: int,
        norm_eps: float,
        dropout: float,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        resnets = []
        for index in range(3):
            skip_channels = in_channels if index == 2 else out_channels
            hidden_channels = previous_output_channels if index == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    hidden_channels + skip_channels,
                    out_channels,
                    time_embedding_dim,
                    norm_num_groups,
                    norm_eps,
                    dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(
                    out_channels,
                    num_heads,
                    cross_attention_dim,
                    norm_num_groups,
                )
                for _ in range(3)
            ]
        )
        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual_states: tuple[torch.Tensor, ...],
        time_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        upsample_size: tuple[int, int] | None,
        attention_mask: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        cross_attention_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        for resnet, attention in zip(self.resnets, self.attentions):
            residual = residual_states[-1]
            residual_states = residual_states[:-1]
            hidden_states = resnet(
                torch.cat([hidden_states, residual], dim=1), time_embedding
            )
            hidden_states = attention(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states, upsample_size)
        return hidden_states


DownBlock = DownBlock2D | CrossAttnDownBlock2D
UpBlock = UpBlock2D | CrossAttnUpBlock2D


class StableDiffusionUNet2DConditionModel(nn.Module, LayerwiseOffloadableModuleMixin):
    """Native SD2.1 UNet with Diffusers-compatible parameter names."""

    layer_names = ["down_blocks", "up_blocks"]
    layerwise_offload_dit_group_enabled = True

    def __init__(self, config: StableDiffusionUNetConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.block_out_channels
        time_embedding_dim = channels[0] * 4
        self.conv_in = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(channels[0], time_embedding_dim)
        self.class_embedding: nn.Embedding | None = None

        down_blocks: list[DownBlock] = []
        output_channels = channels[0]
        for index, block_type in enumerate(config.down_block_types):
            input_channels = output_channels
            output_channels = channels[index]
            common = dict(
                in_channels=input_channels,
                out_channels=output_channels,
                time_embedding_dim=time_embedding_dim,
                norm_num_groups=config.norm_num_groups,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
                add_downsample=index != len(channels) - 1,
                downsample_padding=config.downsample_padding,
            )
            if block_type == "CrossAttnDownBlock2D":
                down_blocks.append(
                    CrossAttnDownBlock2D(
                        **common,
                        num_heads=config.attention_head_dim[index],
                        cross_attention_dim=config.cross_attention_dim,
                    )
                )
            else:
                down_blocks.append(DownBlock2D(**common))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_block = UNetMidBlock2DCrossAttn(
            channels[-1],
            time_embedding_dim,
            config.attention_head_dim[-1],
            config.cross_attention_dim,
            config.norm_num_groups,
            config.norm_eps,
            config.dropout,
        )

        reversed_channels = tuple(reversed(channels))
        reversed_heads = tuple(reversed(config.attention_head_dim))
        up_blocks: list[UpBlock] = []
        output_channels = reversed_channels[0]
        for index, block_type in enumerate(config.up_block_types):
            previous_output_channels = output_channels
            output_channels = reversed_channels[index]
            input_channels = reversed_channels[min(index + 1, len(channels) - 1)]
            common = dict(
                in_channels=input_channels,
                out_channels=output_channels,
                previous_output_channels=previous_output_channels,
                time_embedding_dim=time_embedding_dim,
                norm_num_groups=config.norm_num_groups,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
                add_upsample=index != len(channels) - 1,
            )
            if block_type == "CrossAttnUpBlock2D":
                up_blocks.append(
                    CrossAttnUpBlock2D(
                        **common,
                        num_heads=reversed_heads[index],
                        cross_attention_dim=config.cross_attention_dim,
                    )
                )
            else:
                up_blocks.append(UpBlock2D(**common))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_norm_out = nn.GroupNorm(
            config.norm_num_groups, channels[0], eps=config.norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[0], config.out_channels, 3, padding=1)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv_in.weight.dtype

    def _time_embedding(
        self, sample: torch.Tensor, timestep: torch.Tensor | float | int
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        else:
            timestep = timestep.to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        projected = timestep_embedding(
            timestep,
            self.config.block_out_channels[0],
            flip_sin_to_cos=self.config.flip_sin_to_cos,
            downscale_freq_shift=self.config.freq_shift,
        ).to(dtype=sample.dtype)
        return self.time_embedding(projected)

    @staticmethod
    def _attention_bias(
        mask: torch.Tensor | None, dtype: torch.dtype
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 2:
            return ((1 - mask.to(dtype)) * -10000.0).unsqueeze(1)
        return mask

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        timestep_cond: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        down_block_additional_residuals: tuple[torch.Tensor, ...] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        down_intrablock_additional_residuals: tuple[torch.Tensor, ...] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> StableDiffusionUNetOutput | tuple[torch.Tensor]:
        if timestep_cond is not None or added_cond_kwargs is not None:
            raise ValueError("The Hunyuan3D SD2.1 UNet has no added conditioning.")
        if down_intrablock_additional_residuals is not None:
            raise ValueError("T2I adapter residuals are not supported by Hunyuan3D.")
        if (down_block_additional_residuals is None) != (
            mid_block_additional_residual is None
        ):
            raise ValueError(
                "ControlNet down and mid residuals must be provided together."
            )

        attention_mask = self._attention_bias(attention_mask, sample.dtype)
        encoder_attention_mask = self._attention_bias(
            encoder_attention_mask, sample.dtype
        )
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        time_embedding = self._time_embedding(sample, timestep)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels are required by this UNet.")
            time_embedding = time_embedding + self.class_embedding(class_labels).to(
                sample.dtype
            )

        forward_upsample_size = any(
            dimension % 8 != 0 for dimension in sample.shape[-2:]
        )
        sample = self.conv_in(sample)
        down_residuals = (sample,)
        for block in self.down_blocks:
            if isinstance(block, CrossAttnDownBlock2D):
                sample, residuals = block(
                    sample,
                    time_embedding,
                    encoder_hidden_states,
                    attention_mask,
                    encoder_attention_mask,
                    cross_attention_kwargs,
                )
            else:
                sample, residuals = block(sample, time_embedding)
            down_residuals += residuals

        if down_block_additional_residuals is not None:
            down_residuals = tuple(
                residual + additional
                for residual, additional in zip(
                    down_residuals, down_block_additional_residuals
                )
            )

        sample = self.mid_block(
            sample,
            time_embedding,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            cross_attention_kwargs,
        )
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        for index, block in enumerate(self.up_blocks):
            residuals = down_residuals[-len(block.resnets) :]
            down_residuals = down_residuals[: -len(block.resnets)]
            upsample_size = (
                down_residuals[-1].shape[-2:]
                if index != len(self.up_blocks) - 1 and forward_upsample_size
                else None
            )
            if isinstance(block, CrossAttnUpBlock2D):
                sample = block(
                    sample,
                    residuals,
                    time_embedding,
                    encoder_hidden_states,
                    upsample_size,
                    attention_mask,
                    encoder_attention_mask,
                    cross_attention_kwargs,
                )
            else:
                sample = block(sample, residuals, time_embedding, upsample_size)

        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        if not return_dict:
            return (sample,)
        return StableDiffusionUNetOutput(sample=sample)


EntryClass = StableDiffusionUNet2DConditionModel
