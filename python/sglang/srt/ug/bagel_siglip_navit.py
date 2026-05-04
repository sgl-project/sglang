# SPDX-License-Identifier: Apache-2.0

"""BAGEL packed SigLIP vision encoder used by SRT visual feature extractors."""

from __future__ import annotations

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.siglip.configuration_siglip import (
    SiglipVisionConfig as _SiglipVisionConfig,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipPreTrainedModel,
)

try:  # pragma: no cover - exercised in GPU integration tests when available.
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover
    flash_attn_varlen_func = None


class BAGELSiglipVisionConfig(_SiglipVisionConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        rope=True,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.rope = rope


class BAGELRotaryEmbedding2D(torch.nn.Module):
    def __init__(self, dim: int, max_h: int, max_w: int, base: int = 10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base**freq)

        grid_h = torch.arange(0, max_h).to(inv_freq.dtype)
        grid_h = grid_h[:, None].repeat(1, max_w)
        grid_w = torch.arange(0, max_w).to(inv_freq.dtype)
        grid_w = grid_w[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)
        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    @staticmethod
    def _forward_one_side(grid, inv_freq):
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class BAGELSiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: BAGELSiglipVisionConfig):
        super().__init__()
        self.config = config
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
        num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = num_patches_per_side**2
        self.num_positions = self.num_patches
        if not config.rope:
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def convert_conv2d_to_linear(self, config, meta: bool = False):
        device = "meta" if meta else None
        linear_patch_embedding = nn.Linear(
            config.num_channels * self.patch_size**2,
            self.embed_dim,
            bias=True,
            device=device,
        )
        if not meta:
            weight = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
                self.embed_dim,
                config.num_channels * self.patch_size**2,
            )
            linear_patch_embedding.weight.data = weight
            linear_patch_embedding.bias.data = self.patch_embedding.bias.data
        del self.patch_embedding
        self.patch_embedding = linear_patch_embedding

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(packed_pixel_values)
        if not self.config.rope:
            patch_embeds = patch_embeds + self.position_embedding(
                packed_flattened_position_ids
            )
        return patch_embeds


class BAGELSiglipAttention(SiglipAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos_h: torch.Tensor | None = None,
        sin_h: torch.Tensor | None = None,
        cos_w: torch.Tensor | None = None,
        sin_w: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs, max_seqlen
        total_q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(total_q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = (
                query_states[:, :, : self.head_dim // 2],
                query_states[:, :, self.head_dim // 2 :],
            )
            kh, kw = (
                key_states[:, :, : self.head_dim // 2],
                key_states[:, :, self.head_dim // 2 :],
            )
            qh, kh = _apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = _apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        if flash_attn_varlen_func is not None and query_states.is_cuda:
            attn_output = flash_attn_varlen_func(
                query_states.to(torch.bfloat16),
                key_states.to(torch.bfloat16),
                value_states.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=int(torch.diff(cu_seqlens).max().item()),
                max_seqlen_k=int(torch.diff(cu_seqlens).max().item()),
                causal=False,
            )
        else:
            attn_output = self._dense_varlen_attention(
                query_states,
                key_states,
                value_states,
                cu_seqlens,
            )
        return self.out_proj(attn_output.reshape(total_q_len, -1))

    @staticmethod
    def _dense_varlen_attention(query_states, key_states, value_states, cu_seqlens):
        outputs = []
        scale = query_states.shape[-1] ** -0.5
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            q = query_states[start:end].transpose(0, 1)
            k = key_states[start:end].transpose(0, 1)
            v = value_states[start:end].transpose(0, 1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(probs, v).transpose(0, 1))
        return torch.cat(outputs, dim=0)


class BAGELSiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class BAGELSiglipEncoderLayer(nn.Module):
    def __init__(self, config: BAGELSiglipVisionConfig):
        super().__init__()
        self.self_attn = BAGELSiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = BAGELSiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos_h: torch.Tensor | None = None,
        sin_h: torch.Tensor | None = None,
        cos_w: torch.Tensor | None = None,
        sin_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cos_h=cos_h,
            sin_h=sin_h,
            cos_w=cos_w,
            sin_w=sin_w,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class BAGELSiglipEncoder(nn.Module):
    def __init__(self, config: BAGELSiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BAGELSiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos_h: torch.Tensor | None = None,
        sin_h: torch.Tensor | None = None,
        cos_w: torch.Tensor | None = None,
        sin_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                cos_h=cos_h,
                sin_h=sin_h,
                cos_w=cos_w,
                sin_w=sin_w,
            )
        return hidden_states


class BAGELSiglipVisionTransformer(nn.Module):
    def __init__(self, config: BAGELSiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = BAGELSiglipVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = BAGELRotaryEmbedding2D(dim_head // 2, max_size, max_size)
        self.encoder = BAGELSiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
        )
        extra_inputs = {}
        if self.config.rope:
            extra_inputs = {
                "cos_h": self.rope.cos_h[packed_flattened_position_ids],
                "sin_h": self.rope.sin_h[packed_flattened_position_ids],
                "cos_w": self.rope.cos_w[packed_flattened_position_ids],
                "sin_w": self.rope.sin_w[packed_flattened_position_ids],
            }
        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **extra_inputs,
        )
        return self.post_layernorm(hidden_states)


class BAGELSiglipVisionModel(SiglipPreTrainedModel):
    config_class = BAGELSiglipVisionConfig
    main_input_name = "packed_pixel_values"

    def __init__(self, config: BAGELSiglipVisionConfig):
        super().__init__(config)
        self.vision_model = BAGELSiglipVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        return self.vision_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
