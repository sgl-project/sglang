# Copyright 2026 Liquid AI. All rights reserved.
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
#
# Adapted from vLLM's implementation of Siglip2VisionModel
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/lfm2_siglip2.py
#
# Siglip2 is a vision encoder that supports variable-resolution images via NaFlex.
# Unlike Siglip v1 which uses fixed-size images, Siglip2 handles images of different
# sizes by packing them into sequences and using cu_seqlens for attention.

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Siglip2VisionConfig

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


class Siglip2VisionEmbeddings(nn.Module):
    """Siglip2 vision embeddings with NaFlex variable-resolution support."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        # Siglip2 uses Linear instead of Conv2d for patch embedding
        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """Embed patchified pixel values in packed (unpadded) form.

        Args:
            pixel_values_packed: (1, total_tokens, patch_dim) or
                (total_tokens, patch_dim), packed in tile order.
            spatial_shapes: (num_tiles, 2) on CPU (height, width) per tile.

        Returns:
            (1, total_tokens, embed_dim) packed embeddings.
        """
        assert spatial_shapes.device.type == "cpu", (
            "Expected `spatial_shapes` on CPU to avoid device-to-host sync in "
            "variable-length packing."
        )

        if pixel_values_packed.dim() == 3:
            assert pixel_values_packed.shape[0] == 1
            pixel_values_flat = pixel_values_packed[0]
        else:
            pixel_values_flat = pixel_values_packed

        lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(dtype=torch.int64)
        lengths_list = lengths.tolist()
        total_tokens = int(sum(lengths_list))
        if total_tokens != pixel_values_flat.shape[0]:
            raise ValueError(
                "Packed pixel_values token count does not match spatial_shapes: "
                f"{pixel_values_flat.shape[0]} vs {total_tokens}."
            )

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values_flat.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        packed_pos_embeds = self.resize_positional_embeddings_packed(
            positional_embeddings,
            spatial_shapes,
            lengths_list=lengths_list,
        )

        embeddings = patch_embeds + packed_pos_embeds
        return embeddings.unsqueeze(0)

    @staticmethod
    def resize_positional_embeddings_packed(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        lengths_list: list[int],
    ) -> torch.Tensor:
        """Resize positional embeddings per image and return a packed tensor.

        Args:
            positional_embeddings: (height, width, embed_dim) base grid.
            spatial_shapes: (batch_size, 2) on CPU, (height, width) per image.
            lengths_list: flattened token length per image (height * width).

        Returns:
            (total_tokens, embed_dim) packed positional embeddings.
        """
        assert spatial_shapes.device.type == "cpu"

        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        total_tokens = int(sum(lengths_list))
        packed_pos_embeds = torch.empty(
            (total_tokens, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width)
        pos_4d = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for
        # bfloat16/float16 on CPU.
        if pos_4d.device.type == "cpu":
            pos_4d = pos_4d.to(torch.float32)

        offset = 0
        for i, length in enumerate(lengths_list):
            if length <= 0:
                continue
            height, width = spatial_shapes[i].tolist()
            resized = F.interpolate(
                pos_4d,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape(embed_dim, height * width).transpose(0, 1)
            resized = resized.to(source_dtype)
            packed_pos_embeds[offset : offset + length] = resized
            offset += length

        return packed_pos_embeds


class Siglip2Attention(nn.Module):
    """Multi-headed attention for Siglip2 using optimized VisionAttention backend."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Use SGLang's optimized VisionAttention with automatic backend selection
        self.attn = VisionAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            projection_size=self.embed_dim,
            use_qkv_parallel=True,
            dropout=config.attention_dropout,
            flatten_batch=True,  # For variable-length sequence support
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with variable-length attention.

        Args:
            hidden_states: (1, total_tokens, embed_dim) packed hidden states
            cu_seqlens: Cumulative sequence lengths for variable-length attention
            max_seqlen: Maximum sequence length (unused, VisionAttention computes internally)

        Returns:
            (1, total_tokens, embed_dim) attention output
        """
        return self.attn(hidden_states, cu_seqlens=cu_seqlens)


class Siglip2MLP(nn.Module):
    """MLP for Siglip2 encoder layers."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    """Single encoder layer for Siglip2."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for encoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, embed_dim).
            cu_seqlens: Cumulative sequence lengths tensor.
            max_seqlen: Maximum sequence length.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    """Transformer encoder for Siglip2."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                )
                for idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


def resolve_visual_encoder_outputs(
    encoder_outputs: torch.Tensor | list[torch.Tensor],
    post_layer_norm: Optional[nn.LayerNorm],
    select_layers: Optional[list[int]] = None,
    max_possible_layers: Optional[int] = None,
) -> torch.Tensor:
    """Resolve outputs from visual encoder based on select_layers."""
    if select_layers is None:
        if isinstance(encoder_outputs, list):
            encoder_outputs = encoder_outputs[-1]
        if post_layer_norm is not None:
            encoder_outputs = post_layer_norm(encoder_outputs)
        return encoder_outputs

    if max_possible_layers is None:
        raise ValueError(
            "`max_possible_layers` must be provided alongside `select_layers`"
        )

    if not isinstance(encoder_outputs, list):
        raise ValueError(
            "Expected encoder_outputs to be a list when select_layers is provided"
        )

    # Get the hidden states corresponding to the layer indices
    num_loaded_layers = len(encoder_outputs) - 1
    offset = max_possible_layers - num_loaded_layers
    hs_pool = [
        (
            encoder_outputs[layer_idx]
            if layer_idx >= 0
            else encoder_outputs[layer_idx + offset]
        )
        for layer_idx in select_layers
    ]

    uses_last_layer = select_layers[-1] in (max_possible_layers - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(hs_pool[-1])

    return torch.cat(hs_pool, dim=-1)


class Siglip2VisionTransformer(nn.Module):
    """Siglip2 Vision Transformer with NaFlex variable-resolution support."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_dim = config.hidden_size
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=add_prefix("encoder", prefix),
        )
        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        select_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Forward pass through the vision transformer.

        Args:
            pixel_values_packed: Packed pixel values
            spatial_shapes: (batch_size, 2) tensor with (height, width) per image
            cu_seqlens: Cumulative sequence lengths
            max_seqlen: Maximum sequence length
            select_layers: Optional layer indices to select hidden states from

        Returns:
            Vision features tensor
        """
        hidden_states = self.embeddings(pixel_values_packed, spatial_shapes)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            return_all_hidden_states=select_layers is not None,
        )

        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs,
            self.post_layernorm,
            select_layers=select_layers,
            max_possible_layers=self.config.num_hidden_layers,
        )

        return encoder_outputs


class Siglip2Model(nn.Module):
    """Siglip2 Vision Model for use in vision-language models."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=add_prefix("vision_model", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        select_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Forward pass through the vision model."""
        return self.vision_model(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            select_layers=select_layers,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # VisionAttention uses attn.qkv_proj for fused Q/K/V
            ("attn.qkv_proj", "q_proj", "q"),
            ("attn.qkv_proj", "k_proj", "k"),
            ("attn.qkv_proj", "v_proj", "v"),
        ]
        # VisionAttention uses attn.proj instead of out_proj
        params_rename_mapping = {
            "out_proj": "attn.proj",
        }
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is optional in Siglip2Model
            if (
                name.startswith("vision_model.post_layernorm")
                and self.vision_model.post_layernorm is None
            ):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Apply rename mappings (e.g., out_proj -> attn.proj)
                for old_name, new_name in params_rename_mapping.items():
                    if old_name in name:
                        name = name.replace(old_name, new_name)
                        break

                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
