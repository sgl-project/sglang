# Copyright 2026 Qwen-Image Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class PrefixLayout:
    text_indices: mx.array
    image_indices: mx.array
    prefix_rope: tuple[mx.array, mx.array]
    target_rope: tuple[mx.array, mx.array]
    segments: tuple[tuple[int, int, bool], ...]


def build_layout(image_slots, image_shapes, axes_dims=(16, 56, 56)):
    indices, image_indices, positions, segments = [], [], [], []
    cursor = position = image_index = 0
    for text_index, is_image in enumerate(image_slots):
        if not is_image:
            indices.append(text_index)
            positions.append((position, position, position))
            position += 1
            continue
        start = len(indices)
        if start > cursor:
            segments.append((cursor, start, False))
        _, height, width = image_shapes[image_index]
        for h in range(-(height - height // 2), height // 2):
            for w in range(-(width - width // 2), width // 2):
                indices.append(text_index)
                image_indices.append(len(indices) - 1)
                positions.append((position, h, w))
        segments.append((start, len(indices), True))
        cursor = len(indices)
        position += max(height, width)
        image_index += 1
    if image_index != len(image_shapes) - 1:
        raise ValueError("condition-image slots do not match image_shapes")
    if len(indices) > cursor:
        segments.append((cursor, len(indices), False))
    prefix_len = len(indices)
    _, height, width = image_shapes[-1]
    for h in range(-(height - height // 2), height // 2):
        for w in range(-(width - width // 2), width // 2):
            positions.append((position, h, w))
    pos = mx.array(positions, dtype=mx.float32)
    angles = mx.concatenate(
        [
            pos[:, axis : axis + 1]
            * (10000.0 ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim))
            for axis, dim in enumerate(axes_dims)
        ],
        axis=-1,
    )
    cos, sin = mx.cos(angles), mx.sin(angles)
    return PrefixLayout(
        text_indices=mx.array(indices, dtype=mx.int32),
        image_indices=mx.array(image_indices, dtype=mx.int32),
        prefix_rope=(cos[:prefix_len], sin[:prefix_len]),
        target_rope=(cos[prefix_len:], sin[prefix_len:]),
        segments=tuple(segments),
    )


def apply_rope(x, rope):
    value = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    real, imag = value[..., 0], value[..., 1]
    cos, sin = (r[None, :, None, :] for r in rope)
    return (
        mx.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1)
        .reshape(x.shape)
        .astype(x.dtype)
    )


class ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.zeros(dim)
        self.eps = eps

    def __call__(self, x):
        value = x.astype(mx.float32)
        scale = self.weight.astype(mx.float32) + 1
        return (
            value
            * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
            * scale
        ).astype(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, context_dim, dim, eps):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        x = self.in_layer(self.text_norm(x))
        return self.out_layer(nn.gelu_approx(x.astype(mx.float32)).astype(x.dtype))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = {
            "linear_1": nn.Linear(256, dim, bias=False),
            "linear_2": nn.Linear(dim, dim, bias=False),
        }

    def __call__(self, timestep, dtype):
        # preserve the checkpoint's BF16 division before the FP32 embedding
        timestep = (timestep.astype(dtype).astype(mx.float32) / 1000).astype(dtype)
        freq = mx.exp(-math.log(10000) * mx.arange(128, dtype=mx.float32) / 128)
        angles = timestep.astype(mx.float32)[:, None] * 1000 * freq
        x = mx.concatenate((mx.cos(angles), mx.sin(angles)), axis=-1).astype(dtype)
        x = self.timestep_embedder["linear_1"](x)
        return self.timestep_embedder["linear_2"](
            nn.silu(x.astype(mx.float32)).astype(x.dtype)
        )


class FeedForward(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.proj = nn.Linear(dim, dim * ratio, bias=False)
        self.gate_layer = nn.Linear(dim, dim * ratio, bias=False)
        self.out = nn.Linear(dim * ratio, dim, bias=False)

    def __call__(self, x):
        gate = self.gate_layer(x)
        gate = nn.silu(gate.astype(mx.float32)).astype(gate.dtype)
        return self.out(gate * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)

    def qkv(self, x, rope):
        shape = (*x.shape[:-1], self.heads, self.head_dim)
        q = apply_rope(self.norm_q(self.to_q(x).reshape(shape)), rope)
        k = apply_rope(self.norm_k(self.to_k(x).reshape(shape)), rope)
        v = self.to_v(x).reshape(shape)
        return tuple(value.transpose(0, 2, 1, 3) for value in (q, k, v))

    def output(self, x):
        return self.to_out[0](
            x.transpose(0, 2, 1, 3).reshape(x.shape[0], -1, self.heads * self.head_dim)
        )

    def prefill(self, x, rope, segments):
        q, k, v = self.qkv(x, rope)
        outputs = []
        for start, end, is_image in segments:
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q[:, :, start:end],
                    k[:, :, :end],
                    v[:, :, :end],
                    scale=self.head_dim**-0.5,
                    mask=None if is_image else "causal",
                )
            )
        return self.output(mx.concatenate(outputs, axis=2)), (k, v)

    def __call__(self, x, rope, cache):
        q, k, v = self.qkv(x, rope)
        prefix_k, prefix_v = cache
        out = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate((prefix_k, k), axis=2),
            mx.concatenate((prefix_v, v), axis=2),
            scale=self.head_dim**-0.5,
        )
        return self.output(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, ratio, eps):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = Attention(dim, heads, head_dim, eps)
        self.img_mlp = FeedForward(dim, ratio)

    def prefill(self, x, modulation, layout):
        scale1, gate1, scale2, gate2 = modulation
        out, cache = self.attn.prefill(
            self.img_norm1(x) * (1 + scale1), layout.prefix_rope, layout.segments
        )
        x = x + out * gate1
        return x + self.img_mlp(self.img_norm2(x) * (1 + scale2)) * gate2, cache

    def __call__(self, x, modulation, rope, cache):
        scale1, gate1, scale2, gate2 = modulation
        x = x + self.attn(self.img_norm1(x) * (1 + scale1), rope, cache) * gate1
        return x + self.img_mlp(self.img_norm2(x) * (1 + scale2)) * gate2


class OutputNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)

    def __call__(self, x, temb):
        activation = nn.silu(temb.astype(mx.float32)).astype(temb.dtype)
        return self.norm(x) * (1 + self.linear(activation)[:, None])


class QwenImage21Transformer(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        eps=1e-6,
    ):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.img_in = nn.Linear(in_channels, dim, bias=False)
        self.txt_in = TextProjection(context_in_dim, dim, eps)
        self.time_text_embed = TimeEmbedding(dim)
        self.modulation = [nn.SiLU(), nn.Linear(dim, dim * 4, bias=False)]
        self.transformer_blocks = [
            TransformerBlock(
                dim, num_attention_heads, attention_head_dim, mlp_ratio, eps
            )
            for _ in range(num_layers)
        ]
        self.norm_out = OutputNorm(dim, eps)
        self.proj_out = nn.Linear(dim, out_channels, bias=False)

    def prepare_modulation(self, temb):
        activation = self.modulation[0](temb.astype(mx.float32)).astype(temb.dtype)
        scale1, gate1, scale2, gate2 = mx.split(
            self.modulation[1](activation)[:, None], 4, axis=-1
        )
        return scale1, mx.tanh(gate1), scale2, mx.tanh(gate2)

    def prepare_conditioning(self, embeddings, layout, condition_latents=None):
        """Return request-owned KV tensors; target denoising never mutates them."""
        prefix = self.txt_in(embeddings)[:, layout.text_indices]
        if condition_latents is not None:
            prefix[:, layout.image_indices] = self.img_in(condition_latents)
        temb = self.time_text_embed(mx.zeros((1,)), prefix.dtype)
        modulation = self.prepare_modulation(temb)
        caches = []
        for block in self.transformer_blocks:
            prefix, cache = block.prefill(prefix, modulation, layout)
            caches.append(cache)
        return caches

    def _denoise(self, hidden_states, temb, target_rope, prefix_caches):
        images = self.img_in(hidden_states)
        modulation = self.prepare_modulation(temb)
        for block, cache in zip(self.transformer_blocks, prefix_caches, strict=True):
            images = block(images, modulation, target_rope, cache)
        return self.proj_out(self.norm_out(images, temb))

    def __call__(self, hidden_states, timestep, target_rope, prefix_caches):
        temb = self.time_text_embed(timestep, hidden_states.dtype)
        return self._denoise(hidden_states, temb, target_rope, prefix_caches)

    def compile_denoise(self):
        # keep the sinusoidal embedding outside fusion to preserve BF16 rounding
        compiled = mx.compile(self._denoise)

        def denoise(hidden_states, timestep, target_rope, prefix_caches):
            temb = self.time_text_embed(timestep, hidden_states.dtype)
            return compiled(hidden_states, temb, target_rope, prefix_caches)

        return denoise
