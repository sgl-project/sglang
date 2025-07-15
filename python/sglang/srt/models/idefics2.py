# Copyright 2023 The SGLang team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix


class Idefics2VisionMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Idefics2EncoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.self_attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=self.num_heads,
            projection_size=config.intermediate_size,
            use_qkv_parallel=True,
            quant_config=quant_config,
            dropout=config.attention_dropout,
            qkv_backend="sdpa",
            softmax_in_single_precision=True,
            flatten_batch=False,
            prefix=add_prefix("self_attn", prefix),
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Idefics2VisionMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.

        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Idefics2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention
    layers. Each layer is a
    [`Idefics2EncoderLayer`].

    Args:
        config: Idefics2Config
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [
                Idefics2EncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds (torch.Tensor):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation.
                This is useful if you want more control over how to convert
                `input_ids` indices into associated vectorsthan the model's
                internal embedding lookup matrix.
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
            )
            hidden_states = layer_outputs
        return hidden_states


class Idefics2VisionEmbeddings(nn.Module):
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings
    ` to enable images of variable
    resolution.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision
    Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
    which allows treating images in their native aspect ratio and without the
    need to resize them to the same fixed size. In particular, we start from the
    original pre-trained SigLIP model(which uses images of fixed-size square
    images) and adapt it by training on images of variable resolutions.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def get_position_ids(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
        tgt_sizes: Optional[torch.IntTensor] = None,
    ):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):

            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)
            bucket_coords_h = torch.bucketize(
                fractional_coords_h, boundaries, right=True
            )
            bucket_coords_w = torch.bucketize(
                fractional_coords_w, boundaries, right=True
            )
            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        position_ids = position_ids.to(self.position_embedding.weight.device)
        return position_ids

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
        tgt_sizes: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(
            device=self.patch_embedding.weight.device, dtype=target_dtype
        )
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        position_ids = self.get_position_ids(
            pixel_values, patch_attention_mask, tgt_sizes
        )

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Idefics2VisionTransformer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        require_post_norm: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()

        embed_dim = config.hidden_size
        self.config = config
        self.embeddings = Idefics2VisionEmbeddings(config)
        self.encoder = Idefics2Encoder(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.post_layernorm = (
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            if require_post_norm
            else nn.Identity()
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def compute_cu_seqlens(
        self,
        tgt_sizes: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # shape: (batch_size,)
        if tgt_sizes is not None:
            seqlen = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        elif input_embeds is not None:
            seqlen = torch.full(
                size=(input_embeds.shape[0],),
                fill_value=input_embeds.shape[1],
                dtype=torch.int32,
                device=input_embeds.device,
            )
        else:
            raise ValueError(
                "Either `tgt_sizes` or `input_embeds` must be provided to compute cu_seqlens."
            )

        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=seqlen.device, dtype=torch.int32),
                torch.cumsum(seqlen, dim=0, dtype=torch.int32),
            ],
            dim=0,
        ).to(seqlen.device)
        return cu_seqlens

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        tgt_sizes: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )
        cu_seqlens = self.compute_cu_seqlens(tgt_sizes, hidden_states)
        encoder_outputs = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
        )
        last_hidden_state = self.post_layernorm(encoder_outputs)
        return last_hidden_state
