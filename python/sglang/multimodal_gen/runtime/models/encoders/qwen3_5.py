# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Adapted from the Transformers Qwen3.5 text model (Apache-2.0).

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.qwen3_5 import Qwen3_5EncoderConfig
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.loader.weight_utils import load_stacked_weight
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLTextMLP
from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_rope import (
    apply_qwen_vl_text_rope,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope


def _chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, use_qk_l2norm=False
):
    """FP32 chunked prefill using the current Qwen3.5 triangular solve."""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    batch, heads, length, key_dim = key.shape
    value_dim = value.shape[-1]
    if use_qk_l2norm:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query * (key_dim**-0.5)
    padding = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, padding)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, padding)) for x in (beta, g)]
    num_chunks = (length + padding) // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(batch, heads, num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    g = g.reshape(batch, heads, num_chunks, chunk_size).cumsum(dim=3)
    upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    decay = (g.unsqueeze(4) - g.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    system = (k_beta @ key.transpose(-1, -2)) * decay
    intra = (query @ key.transpose(-1, -2)) * decay
    decayed_k_beta = k_beta * g.exp().unsqueeze(-1)
    values = torch.linalg.solve_triangular(
        system, v_beta, upper=False, unitriangular=True
    )
    k_cumdecay = torch.linalg.solve_triangular(
        system, decayed_k_beta, upper=False, unitriangular=True
    )
    state = torch.zeros(
        batch, heads, key_dim, value_dim, dtype=values.dtype, device=values.device
    )
    output = torch.zeros_like(values)
    query = query * g.exp().unsqueeze(-1)
    key = key * (g[..., -1:] - g).exp().unsqueeze(-1)
    chunk_decay = g[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = values[:, :, i] - k_cumdecay[:, :, i] @ state
        inter = query[:, :, i] @ state
        output[:, :, i] = inter + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    output = output.reshape(batch, heads, -1, value_dim)[:, :, :length]
    return output.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )


class Qwen3_5Linear(ReplicatedLinear):
    def forward(self, x):
        return super().forward(x)[0]


class Qwen3_5RMSNorm(GemmaRMSNorm):
    """Preserve the reference's FP32 reduction and offset-weight rounding."""

    def forward(self, x):
        return self.forward_native(x)


class Qwen3_5RMSNormGated(RMSNorm):
    def __init__(self, hidden_size, eps):
        super().__init__(
            hidden_size, eps=eps, cast_x_before_out_mul=True, force_native=True
        )

    def forward(self, x, gate):
        # Keep the reference's BF16 rounding before applying the FP32 gate.
        return (super().forward(x) * F.silu(gate.float())).to(x.dtype)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            config.linear_conv_kernel_dim,
            groups=conv_dim,
            padding=config.linear_conv_kernel_dim - 1,
            bias=False,
        )
        self.in_proj_qkv = Qwen3_5Linear(config.hidden_size, conv_dim, bias=False)
        self.in_proj_z = Qwen3_5Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = Qwen3_5Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = Qwen3_5Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = Qwen3_5Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, padding_mask):
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :, None].to(hidden_states)
        batch, length, _ = hidden_states.shape
        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        qkv = F.silu(self.conv1d(qkv)[:, :, :length]).transpose(1, 2)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        repeats = self.num_v_heads // self.num_k_heads
        if repeats != 1:
            q = q.repeat_interleave(repeats, dim=2)
            k = k.repeat_interleave(repeats, dim=2)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )
        output = _chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm=True)
        z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
        output = self.norm(output.reshape(-1, self.head_v_dim), z)
        return self.out_proj(output.reshape(batch, length, self.value_dim))


class Qwen3_5Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.q_proj = Qwen3_5Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = Qwen3_5Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = Qwen3_5Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = Qwen3_5Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn = LocalAttention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
            causal=True,
            supported_attention_backends=Qwen3_5Model._supported_attention_backends,
            default_attention_backend=AttentionBackendEnum.TORCH_SDPA,
        )

    def forward(self, hidden_states, position_ids, rotary_emb, attention_mask):
        batch, length, _ = hidden_states.shape
        q, gate = (
            self.q_proj(hidden_states)
            .view(batch, length, self.num_heads, 2 * self.head_dim)
            .chunk(2, dim=-1)
        )
        k = self.k_proj(hidden_states).view(
            batch, length, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch, length, self.num_kv_heads, self.head_dim
        )
        q, k = apply_qwen_vl_text_rope(
            rotary_emb,
            position_ids,
            self.q_norm(q).transpose(1, 2),
            self.k_norm(k).transpose(1, 2),
        )
        output = self.attn(
            q.transpose(1, 2), k.transpose(1, 2), v, attn_mask=attention_mask
        )
        output = (
            output.reshape(batch, length, -1)
            * gate.reshape(batch, length, -1).sigmoid()
        )
        return self.o_proj(output)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self, hidden_states, position_ids, rotary_emb, attention_mask, padding_mask
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, padding_mask)
        else:
            hidden_states = self.self_attn(
                hidden_states, position_ids, rotary_emb, attention_mask
            )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("Qwen3.5 text encoding requires SiLU activation")
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope = dict(config.rope_parameters)
        rope.setdefault("mrope_interleaved", True)
        self.rotary_emb = get_rope(
            head_size=config.head_dim,
            rotary_dim=int(config.head_dim * rope.get("partial_rotary_factor", 1.0)),
            max_position=config.max_position_embeddings,
            base=rope["rope_theta"],
            is_neox_style=True,
            rope_scaling=rope,
        )


class Qwen3_5Model(TextEncoder):
    """Qwen3.5 text-only prefill for diffusion conditioning."""

    layer_names = ["language_model.layers"]
    host_resident_table_names = ["language_model.embed_tokens"]

    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(self, config: Qwen3_5EncoderConfig):
        super().__init__(config)
        if config.quant_config is not None:
            raise NotImplementedError(
                "Qwen3.5 diffusion encoder quantization is not supported"
            )
        self.language_model = Qwen3_5TextModel(config.arch_config.text_config)

    @staticmethod
    def should_materialize_checkpoint_weight(name: str) -> bool:
        return not name.startswith("visual.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if not self.should_materialize_checkpoint_weight(name):
                continue
            if name not in params:
                raise KeyError(name)
            loaded.add(load_stacked_weight(name, weight, params, ()))
        return loaded

    @torch.no_grad()
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        use_cache=False,
        return_dict=None,
        past_key_values=None,
        **kwargs,
    ):
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict
        if use_cache or past_key_values is not None:
            raise ValueError(
                "Qwen3.5 diffusion encoding does not use an autoregressive cache"
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if any(
            kwargs.get(name) is not None
            for name in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            )
        ):
            raise ValueError("Qwen3.5 diffusion encoder accepts text inputs only")
        model = self.language_model
        hidden_states = (
            model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        batch, length, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device)[
                None
            ].expand(batch, -1)
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        elif position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        padding_mask = attention_mask
        if padding_mask is not None and bool(torch.all(padding_mask == 1)):
            padding_mask = None
        causal_mask = None
        if padding_mask is not None:
            causal_mask = torch.ones(
                length, length, dtype=torch.bool, device=hidden_states.device
            ).tril()
            causal_mask = (
                causal_mask[None, None] & padding_mask[:, None, None, :].bool()
            )
        all_hidden = () if output_hidden_states else None
        for layer in model.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            hidden_states = layer(
                hidden_states, position_ids, model.rotary_emb, causal_mask, padding_mask
            )
        hidden_states = model.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        if not return_dict:
            return (
                (hidden_states, all_hidden)
                if output_hidden_states
                else (hidden_states,)
            )
        return BaseEncoderOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden
        )


EntryClass = Qwen3_5Model
