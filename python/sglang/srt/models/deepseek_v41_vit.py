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
from sglang.srt.multimodal.deepseek_v41_image_processing import image_token_types
from sglang.srt.true_on_policy import is_true_on_policy_enabled
from sglang.srt.true_on_policy import vision as top_vision


def _rms_norm(dim: int) -> RMSNorm:
    # The fused CUDA kernels do not take an fp32 weight with a bf16 input.
    return RMSNorm(dim, eps=1e-6, weight_dtype=torch.float32, force_native=True)


def _norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    # One function on bare weights, the one the trainer's tower calls (true_on_policy.vision.rms_norm).
    if is_true_on_policy_enabled():
        return top_vision.rms_norm(x, norm.weight, norm.variance_epsilon)
    return norm(x)


def _linear(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    if is_true_on_policy_enabled():
        return top_vision.linear(x, layer.weight, layer.bias)
    return layer(x)


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
        return _linear(self.proj, x.flatten(1))


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
        gate, up = _linear(self.w1, x).chunk(2, dim=-1)
        return _linear(self.w2, F.silu(gate) * up)


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
        x = x + self.attn(_norm(self.norm1, x), cos, sin, metadata)
        return x + self.mlp(_norm(self.norm2, x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = _rms_norm(args.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        # Passing the known length avoids device-to-host length discovery per layer.
        metadata = prepare_vision_attention_metadata(
            torch.tensor([0, x.shape[0]], dtype=torch.int32),
            x.device,
            max_seqlen=x.shape[0],
        )
        for block in self.blocks:
            x = block(x, cos, sin, metadata)
        return _norm(self.norm, x)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = downsample(x, n_h=n_h, n_w=n_w, ratio=self.downsample_ratio)
        return _linear(self.w2, F.gelu(_linear(self.w1, x)))


def downsample(x: torch.Tensor, *, n_h: int, n_w: int, ratio: int) -> torch.Tensor:
    """ViT rows [n_h * n_w, D] -> [ceil(n_h / ratio) * ceil(n_w / ratio), D * ratio**2], zero-padded windows."""
    x = x.view(n_h, n_w, -1).permute(2, 0, 1)
    x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
    return F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)


def image_span(
    features: torch.Tensor,
    *,
    n_vit_h: int,
    n_vit_w: int,
    ratio: int,
    start: torch.Tensor,
    newline: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """One image's LLM rows: aligner rows in reading order between the start, newline and end embeddings."""
    types = image_token_types(-(-n_vit_h // ratio), -(-n_vit_w // ratio)).to(
        features.device
    )
    span = features.new_empty((len(types), start.shape[0]), dtype=start.dtype)
    span[types == 0] = start
    span[types == 1] = features.to(start.dtype)
    span[types == 2] = newline
    span[types == 3] = end
    return span
