# Copyright 2025 The Qwen Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn


def position_metadata(grid_thw, merge_size, grid_side, head_dim):
    positions, indices, weights, bounds = [], [], [], [0]
    for frames, height, width in grid_thw:
        rows, cols = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
        order = (
            (rows * width + cols)
            .reshape(height // merge_size, merge_size, width // merge_size, merge_size)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        )
        order = mx.tile(order, frames)
        positions.append(
            mx.stack((rows.flatten()[order], cols.flatten()[order]), axis=-1)
        )
        h = mx.linspace(0, grid_side - 1, height)
        w = mx.linspace(0, grid_side - 1, width)
        hf, wf = h.astype(mx.int32), w.astype(mx.int32)
        hc, wc = mx.minimum(hf + 1, grid_side - 1), mx.minimum(wf + 1, grid_side - 1)
        dh, dw = h - hf, w - wf
        corners = mx.stack(
            [
                (a[:, None] * grid_side + b[None]).flatten()
                for a, b in [(hf, wf), (hf, wc), (hc, wf), (hc, wc)]
            ]
        )
        blend = mx.stack(
            [
                (a[:, None] * b[None]).flatten()
                for a, b in [(1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw)]
            ]
        )
        indices.append(corners[:, order])
        weights.append(blend[:, order])
        for _ in range(frames):
            bounds.append(bounds[-1] + height * width)
    positions = mx.concatenate(positions)
    frequency = 10000.0 ** (
        -mx.arange(0, head_dim // 2, 2, dtype=mx.float32) / (head_dim // 2)
    )
    angles = (positions[..., None].astype(mx.float32) * frequency).reshape(
        -1, head_dim // 2
    )
    return (
        mx.concatenate(indices, axis=1),
        mx.concatenate(weights, axis=1),
        (mx.cos(angles), mx.sin(angles)),
        tuple(zip(bounds[:-1], bounds[1:])),
    )


def apply_rope(x, rope):
    first, second = mx.split(x.astype(mx.float32), 2, axis=-1)
    cos, sin = (part[:, None] for part in rope)
    return mx.concatenate(
        (first * cos - second * sin, second * cos + first * sin), axis=-1
    ).astype(x.dtype)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_size, temporal_patch_size, patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, self.kernel, stride=self.kernel)

    def __call__(self, pixels):
        x = pixels.reshape(-1, self.in_channels, *self.kernel).transpose(0, 2, 3, 4, 1)
        dtype = self.proj.weight.dtype
        x = x.astype(dtype).astype(mx.float32)
        x = mx.conv3d(x, self.proj.weight.astype(mx.float32), stride=self.kernel)
        return (
            (x + self.proj.bias.astype(mx.float32))
            .astype(dtype)
            .reshape(x.shape[0], -1)
        )


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, rope, segments):
        qkv = self.qkv(x).reshape(x.shape[0], 3, self.heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        parts = []
        for start, end in segments:
            attended = mx.fast.scaled_dot_product_attention(
                q[start:end].transpose(1, 0, 2)[None],
                k[start:end].transpose(1, 0, 2)[None],
                v[start:end].transpose(1, 0, 2)[None],
                scale=self.head_dim**-0.5,
            )
            parts.append(attended[0].transpose(1, 0, 2).reshape(end - start, -1))
        return self.proj(mx.concatenate(parts))


class MLP(nn.Module):
    def __init__(self, dim, intermediate_size):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, intermediate_size)
        self.linear_fc2 = nn.Linear(intermediate_size, dim)

    def __call__(self, x):
        x = self.linear_fc1(x)
        return self.linear_fc2(nn.gelu_approx(x.astype(mx.float32)).astype(x.dtype))


class Block(nn.Module):
    def __init__(self, dim, heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.mlp = MLP(dim, intermediate_size)

    def __call__(self, x, rope, segments):
        x = x + self.attn(self.norm1(x), rope, segments)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, dim, output_dim, merge_size, post_shuffle):
        super().__init__()
        self.hidden_size = dim * merge_size**2
        self.post_shuffle = post_shuffle
        self.norm = nn.LayerNorm(self.hidden_size if post_shuffle else dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, output_dim)

    def __call__(self, x):
        x = (
            self.norm(x.reshape(-1, self.hidden_size))
            if self.post_shuffle
            else self.norm(x).reshape(-1, self.hidden_size)
        )
        x = self.linear_fc1(x)
        return self.linear_fc2(nn.gelu(x.astype(mx.float32)).astype(x.dtype))


class Qwen3VLVisionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        out_hidden_size=4096,
        num_heads=16,
        depth=27,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        deepstack_visual_indexes=(8, 16, 24),
    ):
        super().__init__()
        self.merge_size = spatial_merge_size
        self.grid_side = int(num_position_embeddings**0.5)
        self.head_dim = hidden_size // num_heads
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.patch_embed = PatchEmbedding(
            in_channels, hidden_size, temporal_patch_size, patch_size
        )
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.blocks = [
            Block(hidden_size, num_heads, intermediate_size) for _ in range(depth)
        ]
        self.merger = PatchMerger(
            hidden_size, out_hidden_size, spatial_merge_size, False
        )
        self.deepstack_merger_list = [
            PatchMerger(hidden_size, out_hidden_size, spatial_merge_size, True)
            for _ in deepstack_visual_indexes
        ]

    def __call__(self, pixels, grid_thw):
        indices, weights, rope, segments = position_metadata(
            grid_thw, self.merge_size, self.grid_side, self.head_dim
        )
        x = self.patch_embed(pixels)
        # match Transformers 4.57: round each corner and each sum in the weight dtype
        corners = self.pos_embed(indices)
        corners = corners * weights[..., None].astype(corners.dtype)
        x = x + (corners[0] + corners[1] + corners[2] + corners[3])
        deepstack = []
        for index, block in enumerate(self.blocks):
            x = block(x, rope, segments)
            if index in self.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[len(deepstack)](x))
        return self.merger(x), deepstack
