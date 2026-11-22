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


class _VisionRMSNorm(RMSNorm):
    """``forward_native`` with an fp32 weight; on CUDA it runs as three launches
    (``rmsnorm_native``) instead of eight, with the same output bits."""

    def __init__(self, dim: int):
        # The fused CUDA kernels do not take an fp32 weight with a bf16 input.
        super().__init__(dim, eps=1e-6, weight_dtype=torch.float32, force_native=True)
        self._forward_method = self._forward_vision

    def _fused(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and torch.version.hip is None
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and self.weight.dtype == torch.float32
        )

    def _forward_vision(self, x, residual=None, *args, **kwargs):
        if residual is None and self._fused(x):
            from sglang.kernels.ops.layernorm.rmsnorm_native import rmsnorm_native

            return rmsnorm_native(x, self.weight, self.variance_epsilon)
        return self.forward_native(x, residual, *args, **kwargs)

    def add_and_norm(
        self, x: torch.Tensor, delta: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x = x + delta`` (skipped for None), then ``(self(x), x)``."""
        if delta is not None and self._fused(x) and delta.dtype == x.dtype:
            from sglang.kernels.ops.layernorm.rmsnorm_native import rmsnorm_native

            return rmsnorm_native(delta, self.weight, self.variance_epsilon, x)
        if delta is not None:
            x = x + delta
        return self(x), x


def _rms_norm(dim: int) -> RMSNorm:
    return _VisionRMSNorm(dim)


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


@lru_cache(32)
def _vision_cos_sin_on(device: torch.device, n_h: int, n_w: int, dim, theta):
    # One host-to-device copy per grid, not a blocking copy per image.
    cos, sin = get_vision_cos_sin(n_h, n_w, dim, theta)
    return cos.to(device), sin.to(device)


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
    if (
        q.is_cuda
        and torch.version.hip is None
        and q.ndim == k.ndim == 3
        and q.shape == k.shape
        and q.shape[0] > 0
        and q.shape[1] > 0
        and not any(t.requires_grad for t in (q, k, cos, sin))
        and q.device == k.device == cos.device == sin.device
        and q.shape[-1] == 64
        and q.stride(-1) == k.stride(-1) == 1
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.dtype == k.dtype
        and cos.dtype == sin.dtype == torch.float32
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.shape == sin.shape == (q.shape[0], 1, 32)
    ):
        from sglang.srt.models.dsv41_vision_rope import apply_fused_vision_rope

        return apply_fused_vision_rope(q, k, cos, sin)
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
        h = self.w1(x)
        if h.is_cuda and torch.version.hip is None:
            # silu(gate) rounded to the dtype, then times up: the eager bits.
            from sglang.kernels.ops.activation.activation import (
                silu_and_mul_with_activation_rounding,
            )

            return self.w2(silu_and_mul_with_activation_rounding(h))
        gate, up = h.chunk(2, dim=-1)
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
        delta: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (x, delta) with the block output x + delta; each residual add
        # runs inside the norm that reads it, in the same order and rounding.
        h, x = self.norm1.add_and_norm(x, delta)
        h, x = self.norm2.add_and_norm(x, self.attn(h, cos, sin, metadata))
        return x, self.mlp(h)


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
            _vision_cos_sin_on(x.device, n_h, n_w, self.rope_dim, self.rope_theta)
            for n_h, n_w in grids
        ]
        cos = torch.cat([c for c, _ in cos_sin])
        sin = torch.cat([s for _, s in cos_sin])
        lengths = [n_h * n_w for n_h, n_w in grids]
        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
        # Passing the known length avoids device-to-host length discovery per layer.
        metadata = prepare_vision_attention_metadata(
            cu_seqlens, x.device, max_seqlen=max(lengths)
        )
        delta = None
        for block in self.blocks:
            x, delta = block(x, delta, cos, sin, metadata)
        return self.norm.add_and_norm(x, delta)[0]


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
