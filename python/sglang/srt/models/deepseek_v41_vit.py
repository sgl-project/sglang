"""DeepSeek-V4.1 vision tower and aligner."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.layernorm import RMSNorm


def _rms_norm(dim: int) -> RMSNorm:
    # The fused CUDA kernels do not take an fp32 weight with a bf16 input.
    return RMSNorm(dim, eps=1e-6, weight_dtype=torch.float32, force_native=True)


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
        self.norm1 = _rms_norm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = _rms_norm(args.vision_dim)
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
    """DeepSeek ViT: full bidirectional attention within each image, with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = _rms_norm(args.vision_dim)

    @property
    def attention_backend(self) -> str:
        return self.blocks[0].attn.qkv_backend_name

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        return self.forward_packed(patches, grids=[(n_h, n_w)])

    def forward_packed(
        self, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Encode images whose patches are concatenated row-wise in ``grids`` order."""
        x = self.patch_embed(patches)
        cos_sin = [
            get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
            for n_h, n_w in grids
        ]
        cos = torch.cat([c for c, _ in cos_sin]).to(x.device)
        sin = torch.cat([s for _, s in cos_sin]).to(x.device)
        lengths = [n_h * n_w for n_h, n_w in grids]
        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
        # Passing the known length avoids device-to-host length discovery per layer.
        metadata = prepare_vision_attention_metadata(
            cu_seqlens, x.device, max_seqlen=max(lengths)
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
        return self.forward_packed(x, grids=[(n_h, n_w)])[0]

    def forward_packed(
        self, x: torch.Tensor, grids: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        """Merge and project each image's ViT rows; returns one tensor per image."""
        rows = x.split([n_h * n_w for n_h, n_w in grids])
        merged = [
            self._merge(rows=image_rows, n_h=n_h, n_w=n_w)
            for image_rows, (n_h, n_w) in zip(rows, grids, strict=True)
        ]
        out = self.w2(F.gelu(self.w1(torch.cat(merged))))
        return list(out.split([len(m) for m in merged]))

    def _merge(self, rows: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = rows.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        return F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)


# Kernels that split attention by forward_metadata.cu_seqlens. SDPA ignores that
# metadata, so packed images would attend across each other.
_PACKED_ATTENTION_BACKENDS = frozenset({"fa3", "fa4", "triton_attn"})
# Bounds peak memory on many-image calls; measured on H100,
# small-image speedup levels off by 8 images per pass.
_MAX_IMAGES_PER_PASS = 8


def encode_images(
    vision: ViT,
    aligner: Aligner,
    patches: list[torch.Tensor],
    grids: list[tuple[int, int]],
) -> list[torch.Tensor]:
    """Return aligner rows per image, sharing ViT passes when the backend allows.

    Packed output may differ from per-image output by rounding, since each
    matmul sees more rows.
    """
    if vision.attention_backend not in _PACKED_ATTENTION_BACKENDS:
        return [
            aligner(vision(p, n_h, n_w), n_h, n_w)
            for p, (n_h, n_w) in zip(patches, grids, strict=True)
        ]
    features = []
    for start in range(0, len(grids), _MAX_IMAGES_PER_PASS):
        chunk = grids[start : start + _MAX_IMAGES_PER_PASS]
        chunk_patches = torch.cat(patches[start : start + _MAX_IMAGES_PER_PASS])
        hidden = vision.forward_packed(chunk_patches, grids=chunk)
        features.extend(aligner.forward_packed(hidden, grids=chunk))
    return features
