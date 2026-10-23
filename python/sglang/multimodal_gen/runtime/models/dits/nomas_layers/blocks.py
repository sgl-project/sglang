# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Tencent-Hunyuan/Hy-Nomas
# Original PatchEmbed: Copyright 2020 Ross Wightman.
"""Nomas DiT building blocks; upstream options Nomas never sets are removed, parameter names are unchanged."""

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.diffusion import modulate_scale_shift, residual_gate_add


def attention(q, k, v, attn_mask):
    # Match upstream unpad/FA2/pad: padded query rows have zero attention.
    result = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask[:, None, None, :],
    )
    return result.transpose(1, 2).flatten(2) * attn_mask[:, :, None]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(
            x
        ) * self.weight


class ModulateDiT(nn.Module):
    def __init__(self, hidden_size: int, factor: int, act_layer):
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size)

    def forward(self, x):
        return self.linear(self.act(x))


def modulate(x, shift, scale):
    return modulate_scale_shift(x, scale, shift)


def apply_gate(x, gate):
    return x * gate.unsqueeze(1)


def gated_residual(residual, update, gate):
    # The shared kernel preserves the multiply's storage-dtype rounding before
    # the residual addition, and falls back for mixed dtypes (camera residuals).
    return residual_gate_add(residual, update, gate.unsqueeze(1))


class PatchEmbed(nn.Module):
    """Conv3d patchify; [B, C, T, H, W] -> [B, T*H*W, D]."""

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TextProjection(nn.Module):
    def __init__(self, in_channels, hidden_size, act_layer):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, caption):
        return self.linear_2(self.act_1(self.linear_1(caption)))


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of fractional timesteps, (*t.shape, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t.flatten()[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.view(*t.shape, dim)


class TimestepEmbedder(nn.Module):
    def __init__(
        self, hidden_size, act_layer, frequency_embedding_size=256, max_period=10000
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            act_layer(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(t_freq.type(self.mlp[0].weight.dtype))


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, act_layer):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, in_channels)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class FinalLayerPerBlock(nn.Module):
    """adaLN final layer; ``c`` is the per-request timestep vector [B, D]."""

    def __init__(self, hidden_size, patch_size, out_channels, act_layer):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)
        self.adaLN_modulation = nn.Sequential(
            act_layer(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size, heads_num):
        super().__init__()
        self.heads_num = heads_num
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(hidden_size, 4 * hidden_size, nn.SiLU)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c, attn_mask):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        qkv = self.self_attn_qkv(self.norm1(x))
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        x = x + apply_gate(self.self_attn_proj(attention(q, k, v, attn_mask)), gate_msa)
        return x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)


class IndividualTokenRefiner(nn.Module):
    def __init__(self, hidden_size, heads_num, depth):
        super().__init__()
        self.blocks = nn.ModuleList(
            [IndividualTokenRefinerBlock(hidden_size, heads_num) for _ in range(depth)]
        )

    def forward(self, x, c, mask):
        mask = mask.clone().bool()
        mask[:, 0] = True  # avoid an all-masked row producing NaN
        for block in self.blocks:
            x = block(x, c, mask)
        return x
