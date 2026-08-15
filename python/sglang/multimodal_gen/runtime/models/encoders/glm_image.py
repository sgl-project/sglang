# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_image/modeling_glm_image.py
# Copyright 2025 The ZhipuAI Team
# Copyright 2025 The HuggingFace Team
# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Cache, DynamicCache, GenerationMixin
from transformers.activations import ACT2FN
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.glm_image.configuration_glm_image import (
    GlmImageConfig,
    GlmImageTextConfig,
    GlmImageVisionConfig,
    GlmImageVQVAEConfig,
)
from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids

from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class GlmImagePreTrainedModel(PreTrainedModel):
    config_class = GlmImageConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["GlmImageTextDecoderLayer", "GlmImageVisionBlock"]
    _supports_sdpa = True


class GlmImageVisionMLP(nn.Module):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class GlmImageVisionAttention(nn.Module):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=config.attention_bias,
        )
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn = LocalAttention(
            self.num_heads,
            self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
            allow_cudnn_sdp=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .unbind(1)
        )

        bounds = tuple(int(item) for item in cu_seqlens.tolist())
        max_seqlen = max(
            (stop - start for start, stop in zip(bounds[:-1], bounds[1:])),
            default=0,
        )
        cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32)
        output = self.attn.attn_impl.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=bounds,
        )
        return self.proj(output.reshape(seq_length, -1).contiguous())


class GlmImageVisionPatchEmbed(nn.Module):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(
            -1, self.embed_dim
        )


class GlmImageVisionEmbeddings(nn.Module):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()
        self.position_embedding = nn.Embedding(
            (config.image_size // config.patch_size) ** 2,
            config.hidden_size,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor,
        image_shapes: torch.Tensor,
        h_coords: torch.Tensor,
        w_coords: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.position_embedding.weight
        source_size = int(weight.shape[0] ** 0.5)
        position_embedding = (
            weight.view(source_size, source_size, weight.shape[1])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )

        token_positions = torch.arange(embeddings.shape[0], device=embeddings.device)
        sequence_ids = (
            token_positions.unsqueeze(0) >= lengths.cumsum(0).unsqueeze(1)
        ).sum(0)
        target_h = image_shapes[sequence_ids, 1].float()
        target_w = image_shapes[sequence_ids, 2].float()
        grid = torch.stack(
            (
                ((w_coords + 0.5) / target_w) * 2 - 1,
                ((h_coords + 0.5) / target_h) * 2 - 1,
            ),
            dim=-1,
        ).view(1, -1, 1, 2)
        adapted = F.grid_sample(
            position_embedding,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        adapted = adapted.squeeze(0).squeeze(-1).permute(1, 0)
        return embeddings + adapted.to(device=embeddings.device, dtype=weight.dtype)


class GlmImageVisionBlock(nn.Module):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GlmImageVisionAttention(config)
        self.mlp = GlmImageVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class GlmImageVisionModel(nn.Module):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = GlmImageVisionPatchEmbed(config)
        self.embeddings = GlmImageVisionEmbeddings(config)
        self.blocks = nn.ModuleList(
            [GlmImageVisionBlock(config) for _ in range(config.depth)]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> BaseModelOutputWithPooling:
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        hidden_states = self.patch_embed(pixel_values)
        position_ids = position_ids.to(hidden_states.device)
        grid_thw = grid_thw.to(hidden_states.device)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(hidden_states.device)
        hidden_states = self.embeddings(
            hidden_states,
            lengths,
            grid_thw,
            position_ids[:, 0],
            position_ids[:, 1],
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class GlmImageVQVAEVectorQuantizer(nn.Module):
    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__()
        self.embedding_dim = config.embed_dim
        self.embedding = nn.Embedding(config.num_embeddings, config.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = F.normalize(
            hidden_states.view(-1, self.embedding_dim), p=2, dim=-1
        )
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        distances = (
            hidden_states.square().sum(dim=1, keepdim=True)
            + embedding.square().sum(dim=1)
            - 2 * hidden_states @ embedding.t()
        )
        return distances.argmin(dim=1)


class GlmImageVQVAE(nn.Module):
    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__()
        self.quantize = GlmImageVQVAEVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.latent_channels, 1)

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.quant_conv(hidden_states))


class GlmImageRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query = query_rot * cos + _rotate_half(query_rot) * sin
    key = key_rot * cos + _rotate_half(key_rot) * sin
    return torch.cat((query, query_pass), dim=-1), torch.cat((key, key_pass), dim=-1)


class GlmImageTextRotaryEmbedding(nn.Module):
    def __init__(self, config: GlmImageTextConfig):
        super().__init__()
        rope_parameters = config.rope_parameters
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_dim = int(
            head_dim * rope_parameters.get("partial_rotary_factor", 1.0)
        )
        self.rope_theta = rope_parameters["rope_theta"]
        self.mrope_section = rope_parameters.get("mrope_section", [8, 12, 12])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # keep inverse frequencies in FP32 instead of a BF16-rounded buffer
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    device=position_ids.device,
                    dtype=torch.float32,
                )
                / self.rotary_dim
            )
        )
        frequencies = (
            inv_freq[None, None, :, None] @ position_ids[:, :, None, :].float()
        ).transpose(2, 3)
        chunks = frequencies.split(self.mrope_section, dim=-1)
        frequencies = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(chunks)], dim=-1
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(
            hidden_states.dtype
        )


class GlmImageTextAttention(nn.Module):
    def __init__(self, config: GlmImageTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.attn = LocalAttention(
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            softmax_scale=self.scaling,
            causal=True,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
            allow_cudnn_sdp=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).view(
            *input_shape, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            *input_shape, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            *input_shape, self.num_key_value_heads, self.head_dim
        )
        cos, sin = position_embeddings
        query, key = _apply_rotary_pos_emb(
            query.transpose(1, 2),
            key.transpose(1, 2),
            cos,
            sin,
        )
        if past_key_values is not None:
            key, value = past_key_values.update(
                key,
                value.transpose(1, 2),
                self.layer_idx,
            )
        else:
            value = value.transpose(1, 2)
        # single-token decode is faster with SDPA than FlashAttention
        if (
            query.shape[2] == 1
            and attention_mask is None
            and query.device.type == "cuda"
        ):
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=self.num_heads != self.num_key_value_heads,
            ).transpose(1, 2)
        else:
            output = self.attn(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
            )
        return self.o_proj(output.reshape(*input_shape, -1).contiguous())


class GlmImageTextMLP(nn.Module):
    def __init__(self, config: GlmImageTextConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(up * self.activation_fn(gate))


class GlmImageTextDecoderLayer(nn.Module):
    def __init__(self, config: GlmImageTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GlmImageTextAttention(config, layer_idx)
        self.mlp = GlmImageTextMLP(config)
        self.input_layernorm = GlmImageRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GlmImageRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_self_attn_layernorm = GlmImageRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_mlp_layernorm = GlmImageRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
            past_key_values,
        )
        hidden_states = residual + self.post_self_attn_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.post_mlp_layernorm(hidden_states)


def _can_skip_causal_mask(
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
) -> bool:
    if past_key_values is not None:
        return False
    if attention_mask is None:
        return True
    return attention_mask.ndim == 2 and bool(torch.all(attention_mask > 0).item())


class GlmImageTextModel(nn.Module):
    def __init__(self, config: GlmImageTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                GlmImageTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GlmImageRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = GlmImageTextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        del kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_length = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = torch.arange(
                past_length,
                past_length + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).view(1, 1, -1)
            position_ids = position_ids.expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if _can_skip_causal_mask(attention_mask, past_key_values):
            causal_mask = None
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                causal_mask,
                past_key_values,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@dataclass
class GlmImageModelOutputWithPast(BaseModelOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


class GlmImageModel(nn.Module):
    def __init__(self, config: GlmImageConfig):
        super().__init__()
        self.config = config
        self.visual = GlmImageVisionModel(config.vision_config)
        self.language_model = GlmImageTextModel(config.text_config)
        self.vqmodel = GlmImageVQVAE(config.vq_config)
        self.rope_deltas = None
        self._cached_decode_position_ids = None
        self._prefill_len = None

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    @staticmethod
    def get_vision_position_ids(
        start_position: int,
        grid_thw: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
        temporal = torch.arange(grid_t, device=device).repeat_interleave(
            grid_h * grid_w
        )
        height = (
            torch.arange(grid_h, device=device).repeat_interleave(grid_w).repeat(grid_t)
        )
        width = torch.arange(grid_w, device=device).repeat(grid_h * grid_t)
        return torch.stack((temporal, height, width), dim=0) + start_position

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        images_per_sample: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device
        position_ids = torch.ones(
            3,
            batch_size,
            sequence_length,
            dtype=input_ids.dtype,
            device=device,
        )
        text_positions = torch.arange(sequence_length, device=device).expand(3, -1)
        if image_grid_thw is None:
            grids_per_sample = [None] * batch_size
        elif images_per_sample is None:
            grids_per_sample = [image_grid_thw] * batch_size
        else:
            grids_per_sample = torch.split(image_grid_thw, images_per_sample.tolist())

        decode_positions = []
        for batch_idx in range(batch_size):
            current_ids = input_ids[batch_idx]
            current_grids = grids_per_sample[batch_idx]
            if (
                attention_mask is not None
                and attention_mask.shape[1] == sequence_length
            ):
                valid_mask = attention_mask[batch_idx] == 1
                current_ids = current_ids[valid_mask]
            else:
                valid_mask = None

            image_ends = torch.where(current_ids == self.config.image_end_token_id)[0]
            image_starts = (
                torch.where(current_ids == self.config.image_start_token_id)[0] + 1
            )
            current_position = 0
            previous_image_end = 0
            current_parts = []
            if current_grids is not None:
                for image_idx, (start, end) in enumerate(zip(image_starts, image_ends)):
                    if image_idx >= len(current_grids):
                        break
                    text_length = int(start - previous_image_end)
                    current_parts.append(
                        text_positions[
                            :, current_position : current_position + text_length
                        ]
                    )
                    current_position += text_length
                    vision_positions = self.get_vision_position_ids(
                        current_position,
                        current_grids[image_idx],
                        device,
                    )
                    current_parts.append(vision_positions)
                    current_position += int(
                        max(current_grids[image_idx][1], current_grids[image_idx][2])
                    )
                    previous_image_end = int(end)

            remaining_length = len(current_ids) - previous_image_end
            current_parts.append(
                text_positions[
                    :, current_position : current_position + remaining_length
                ]
            )
            current_position += remaining_length
            current_positions = torch.cat(current_parts, dim=-1)
            if valid_mask is None:
                position_ids[:, batch_idx] = current_positions
            else:
                position_ids[:, batch_idx, valid_mask] = current_positions

            if current_grids is None:
                continue
            decode_grid_count = max(len(current_grids) - len(image_ends), 0)
            temporal_parts = []
            height_parts = []
            width_parts = []
            decode_position = current_position
            decode_grids = (
                current_grids[-decode_grid_count:] if decode_grid_count else ()
            )
            for grid in decode_grids:
                height, width = int(grid[1]), int(grid[2])
                token_count = height * width
                temporal_parts.append(
                    torch.full(
                        (token_count,),
                        decode_position,
                        device=device,
                        dtype=torch.long,
                    )
                )
                height_parts.append(
                    decode_position
                    + torch.arange(height, device=device)
                    .unsqueeze(1)
                    .expand(height, width)
                    .flatten()
                )
                width_parts.append(
                    decode_position
                    + torch.arange(width, device=device)
                    .unsqueeze(0)
                    .expand(height, width)
                    .flatten()
                )
                decode_position += max(height, width)
            end_position = torch.tensor(
                [decode_position], device=device, dtype=torch.long
            )
            temporal_parts.append(end_position)
            height_parts.append(end_position)
            width_parts.append(end_position)
            decode_positions.append(
                torch.stack(
                    (
                        torch.cat(temporal_parts),
                        torch.cat(height_parts),
                        torch.cat(width_parts),
                    )
                )
            )

        self._prefill_len = sequence_length
        if decode_positions:
            max_length = max(item.shape[1] for item in decode_positions)
            self._cached_decode_position_ids = torch.stack(
                [
                    F.pad(item, (0, max_length - item.shape[1]), mode="replicate")
                    for item in decode_positions
                ]
            )
        else:
            self._cached_decode_position_ids = None
        deltas = torch.zeros(
            batch_size, 1, dtype=input_ids.dtype, device=input_ids.device
        )
        return position_ids, deltas

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> BaseModelOutputWithPooling:
        outputs = self.visual(
            pixel_values.to(self.visual.dtype),
            image_grid_thw,
        )
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        outputs.pooler_output = torch.split(outputs.last_hidden_state, split_sizes)
        return outputs

    def get_image_tokens(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        split_sizes = image_grid_thw.prod(dim=-1).tolist()
        tokens = []
        for item, grid in zip(
            torch.split(hidden_states, split_sizes, dim=0), image_grid_thw
        ):
            grid_t, grid_h, grid_w = (int(value) for value in grid.tolist())
            item = item.view(grid_t, grid_h, grid_w, hidden_size)
            item = item.permute(0, 3, 1, 2).contiguous()
            tokens.append(self.vqmodel.encode(item))
        return torch.cat(tokens)

    def _compute_position_ids(
        self,
        input_ids: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        images_per_sample: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor | None:
        past_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if past_length == 0:
            self.rope_deltas = None
            self._cached_decode_position_ids = None
            self._prefill_len = None
        if input_ids is not None and image_grid_thw is not None and past_length == 0:
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                images_per_sample,
                attention_mask,
            )
            return position_ids
        if self.rope_deltas is None or past_length == 0:
            return None
        batch_size, sequence_length = inputs_embeds.shape[:2]
        if self._cached_decode_position_ids is not None:
            step = past_length - self._prefill_len
            return self._cached_decode_position_ids[
                :, :, step : step + sequence_length
            ].permute(1, 0, 2)
        positions = torch.arange(
            past_length,
            past_length + sequence_length,
            device=inputs_embeds.device,
        )
        return positions.view(1, 1, -1).expand(3, batch_size, -1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        images_per_sample: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> GlmImageModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None:
            if images_per_sample is None:
                source_grids = image_grid_thw[:-1]
            else:
                grids_per_sample = torch.split(
                    image_grid_thw, images_per_sample.tolist()
                )
                non_padding = (
                    attention_mask == 1
                    if attention_mask is not None
                    else torch.ones_like(input_ids, dtype=torch.bool)
                )
                source_counts = (
                    (input_ids == self.config.image_end_token_id) & non_padding
                ).sum(dim=1)
                source_grid_parts = [
                    grids[: int(count)]
                    for grids, count in zip(grids_per_sample, source_counts)
                    if int(count) > 0
                ]
                if not source_grid_parts:
                    raise ValueError(
                        "pixel_values were provided without source image tokens"
                    )
                source_grids = torch.cat(source_grid_parts)
            image_features = self.get_image_features(pixel_values, source_grids)
            image_ids = self.get_image_tokens(
                torch.cat(image_features.pooler_output), source_grids
            ).to(input_ids.device)
            placeholder_mask = input_ids == self.config.image_token_id
            if placeholder_mask.sum() != image_ids.numel():
                raise ValueError(
                    "Image placeholder count does not match encoded image tokens"
                )
            input_ids = input_ids.masked_scatter(placeholder_mask, image_ids)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if position_ids is None:
            position_ids = self._compute_position_ids(
                input_ids,
                image_grid_thw,
                images_per_sample,
                inputs_embeds,
                attention_mask,
                past_key_values,
            )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return GlmImageModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


@dataclass
class GlmImageCausalLMOutputWithPast(CausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


class GlmImageForConditionalGeneration(
    GlmImagePreTrainedModel, GenerationMixin, LayerwiseOffloadableModuleMixin
):
    _tied_weights_keys = {}
    layerwise_offload_dit_group_enabled = False
    layer_names = ["model.language_model.layers", "model.visual.blocks"]

    def __init__(self, config: GlmImageConfig):
        super().__init__(config)
        self.model = GlmImageModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vision_vocab_size,
            bias=False,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        del kwargs
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_image_tokens(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.get_image_tokens(hidden_states, image_grid_thw)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        images_per_sample: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> GlmImageCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            images_per_sample=images_per_sample,
            use_cache=use_cache,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )
        return GlmImageCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        images_per_sample=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        model_inputs["position_ids"] = None
        model_inputs["images_per_sample"] = images_per_sample
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
        return model_inputs


EntryClass = GlmImageForConditionalGeneration
