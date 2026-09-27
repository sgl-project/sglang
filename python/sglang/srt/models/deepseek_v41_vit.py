"""Vision tower and aligner from the 260903 reference implementation."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def apply_vision_rotary(q, k, position_embeddings, x_shape):
    # The reference pairs the two halves of each head, with FP32 arithmetic.
    cos, sin = position_embeddings
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


class Attention(VisionAttention):
    def __init__(self, args):
        super().__init__(
            embed_dim=args.vision_dim,
            num_heads=args.vision_n_heads,
            projection_size=args.vision_dim,
            use_qkv_parallel=True,
            use_data_parallel=True,
            customized_position_embedding_applier=apply_vision_rotary,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        return (
            super()
            .forward(
                x,
                position_embeddings=(cos, sin),
                forward_metadata=metadata,
                max_seqlen=x.shape[0],
            )
            .squeeze(0)
        )


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, metadata)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        # One image is one full, bidirectional sequence. Supplying its known
        # length avoids device-to-host length discovery in every encoder layer.
        metadata = prepare_vision_attention_metadata(
            torch.tensor([0, x.shape[0]], dtype=torch.int32),
            x.device,
            max_seqlen=x.shape[0],
        )
        for block in self.blocks:
            x = block(x, cos, sin, metadata)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
