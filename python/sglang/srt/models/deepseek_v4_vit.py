# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Replicated DeepSeek-V4 ViT and aligner, matching inference/vision.py in
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp. The wrapper applies the processor's
permutation and inserts sentinel embeddings after alignment.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import add_prefix


@lru_cache(16)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device: torch.device
):
    """2D RoPE cos/sin tables for an n_h x n_w patch grid, fp32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1).to(device), freqs.sin().unsqueeze(1).to(device)


def apply_vision_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    """RMSNorm computed in fp32 with an fp32 weight, matching the reference."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4VisionPatchEmbed(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        patch_size = config.vision_patch_size
        self.proj = ReplicatedLinear(
            3 * patch_size**2,
            config.vision_dim,
            bias=True,
            quant_config=None,
            prefix=add_prefix("proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_patches, 3, p, p] -> [n_patches, vision_dim]
        out, _ = self.proj(x.flatten(1))
        return out


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        dim = config.vision_dim
        self.n_heads = config.vision_n_heads
        self.head_dim = dim // self.n_heads
        self.wqkv = ReplicatedLinear(
            dim,
            3 * dim,
            bias=True,
            quant_config=None,
            prefix=add_prefix("wqkv", prefix),
        )
        self.wo = ReplicatedLinear(
            dim, dim, bias=True, quant_config=None, prefix=add_prefix("wo", prefix)
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        qkv, _ = self.wqkv(x)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in qkv.chunk(3, dim=-1))
        q = apply_vision_rotary(q, cos, sin)
        k = apply_vision_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        out, _ = self.wo(o.transpose(0, 1).reshape(n, -1))
        return out


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        self.w1 = ReplicatedLinear(
            config.vision_dim,
            2 * config.vision_inter_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("w1", prefix),
        )
        self.w2 = ReplicatedLinear(
            config.vision_inter_dim,
            config.vision_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w1(x)
        gate, up = gate_up.chunk(2, dim=-1)
        out, _ = self.w2(F.silu(gate) * up)
        return out


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        dim = config.vision_dim
        self.norm1 = DeepseekV4VisionRMSNorm(dim)
        self.attn = DeepseekV4VisionAttention(config, prefix=add_prefix("attn", prefix))
        self.norm2 = DeepseekV4VisionRMSNorm(dim)
        self.mlp = DeepseekV4VisionMLP(config, prefix=add_prefix("mlp", prefix))

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionTower(nn.Module):
    """DeepSeek-V4 ViT: full bidirectional attention over one image with 2D RoPE.

    Maps to the `vision.*` weights in the HF checkpoint.
    """

    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4VisionPatchEmbed(
            config, prefix=add_prefix("patch_embed", prefix)
        )
        self.blocks = nn.ModuleList(
            DeepseekV4VisionBlock(config, prefix=add_prefix(f"blocks.{i}", prefix))
            for i in range(config.vision_n_layers)
        )
        self.norm = DeepseekV4VisionRMSNorm(config.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, x.device
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4VisionAligner(nn.Module):
    """Pixel-unshuffle (3x3) + 2-layer GELU projector to the LLM hidden size.

    Maps to the `aligner.*` weights in the HF checkpoint.
    """

    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = ReplicatedLinear(
            in_dim,
            config.hidden_size,
            bias=True,
            quant_config=None,
            prefix=add_prefix("w1", prefix),
        )
        self.w2 = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=None,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        h, _ = self.w1(x)
        out, _ = self.w2(F.gelu(h))
        return out


class DeepseekV4VisionEncoder(nn.Module):
    """Convenience wrapper: ViT + aligner, one call per image.

    Returns aligner outputs in row-major grid order ([n_llm_tokens, hidden]).
    The caller applies the `perm` reordering and merges with the learned
    image sentinel embeddings on the language-model side.
    """

    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        self.vision = DeepseekV4VisionTower(config, prefix=add_prefix("vision", prefix))
        self.aligner = DeepseekV4VisionAligner(
            config, prefix=add_prefix("aligner", prefix)
        )

    def forward(
        self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int
    ) -> torch.Tensor:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    # alias matching the reference model's naming
    def encode_image(
        self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int
    ) -> torch.Tensor:
        return self.forward(patches, n_vit_h, n_vit_w)
