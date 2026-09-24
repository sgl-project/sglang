# Copyright 2025 The Qwen Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn

from sglang.multimodal_gen.runtime.loader.mlx_loader import (
    create_model,
    load_quantized_weights,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision_mlx import (
    Qwen3VLVisionEncoder,
)


def interleaved_rope(position_ids, head_dim, theta, sections, dtype):
    frequency = theta ** (-mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    angles = position_ids[..., None].astype(mx.float32) * frequency
    selected = mx.array(angles[0])
    selected[..., 1 : sections[1] * 3 : 3] = angles[1, ..., 1 : sections[1] * 3 : 3]
    selected[..., 2 : sections[2] * 3 : 3] = angles[2, ..., 2 : sections[2] * 3 : 3]
    return mx.cos(selected).astype(dtype), mx.sin(selected).astype(dtype)


def apply_rope(x, rope):
    first, second = mx.split(x, 2, axis=-1)
    cos, sin = (part[:, :, None] for part in rope)
    return mx.concatenate(
        (first * cos - second * sin, second * cos + first * sin), axis=-1
    )


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)

    def __call__(self, x, rope, mask):
        q = self.q_norm(
            self.q_proj(x).reshape(*x.shape[:2], self.num_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(x).reshape(*x.shape[:2], self.num_kv_heads, self.head_dim)
        )
        v = self.v_proj(x).reshape(*x.shape[:2], self.num_kv_heads, self.head_dim)
        attended = mx.fast.scaled_dot_product_attention(
            apply_rope(q, rope).transpose(0, 2, 1, 3),
            apply_rope(k, rope).transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
            mask=mask,
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1))


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        gate = self.gate_proj(x)
        gate = nn.silu(gate.astype(mx.float32)).astype(gate.dtype)
        return self.down_proj(gate * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, eps
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.self_attn = Attention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def __call__(self, x, rope, mask):
        x = x + self.self_attn(self.input_layernorm(x), rope, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3VLTextEncoder(nn.Module):
    """Qwen-Image conditioning uses the last decoder state before final RMSNorm."""

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=5000000,
        rms_norm_eps=1e-6,
        mrope_section=(24, 20, 20),
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            DecoderLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        visual_positions=None,
        deepstack_visual_embeds=(),
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = x.shape
        if position_ids is None:
            position_ids = mx.broadcast_to(mx.arange(length), (3, batch, length))
        rope = interleaved_rope(
            position_ids, self.head_dim, self.rope_theta, self.mrope_section, x.dtype
        )
        mask = "causal"
        if attention_mask is not None:
            indices = mx.arange(length)
            causal = indices[:, None] >= indices[None, :]
            mask = causal[None, None] & attention_mask[:, None, None, :].astype(
                mx.bool_
            )
        for index, layer in enumerate(self.layers):
            x = layer(x, rope, mask)
            if index < len(deepstack_visual_embeds):
                flat = x.reshape(-1, x.shape[-1])
                flat[visual_positions] = (
                    flat[visual_positions] + deepstack_visual_embeds[index]
                )
                x = flat.reshape(x.shape)
        return x


def load_encoders(path, config, quantization, with_images):
    weights = mx.load(path)
    text_weights, vision_weights = {}, {}
    for name, value in weights.items():
        if name.startswith("model.visual."):
            vision_weights[name.removeprefix("model.visual.")] = value
        elif name.startswith("model.language_model."):
            text_weights[name.removeprefix("model.language_model.")] = value
        elif name.startswith("model."):
            text_weights[name.removeprefix("model.")] = value
        elif not name.startswith("lm_head."):
            raise ValueError(f"unrecognized Qwen3-VL weight: {name}")
    text_config = dict(config["text_config"])
    text_config["mrope_section"] = text_config["rope_scaling"]["mrope_section"]
    text_encoder = load_quantized_weights(
        create_model(Qwen3VLTextEncoder, text_config), text_weights, quantization
    )
    vision_encoder = None
    if with_images:
        name = "patch_embed.proj.weight"
        vision_weights[name] = vision_weights[name].transpose(0, 2, 3, 4, 1)
        vision_encoder = load_quantized_weights(
            create_model(Qwen3VLVisionEncoder, config["vision_config"]),
            vision_weights,
            quantization,
        )
    return text_encoder, vision_encoder
