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
        return self.norm(x)


# Query tile for CPU attention. A full 16×8649² fp32 score matrix is ~4.8 GiB.
_CPU_QUERY_CHUNK = 256


def _as_fp32_cpu(module: nn.Module) -> nn.Module:
    return module.to(device="cpu", dtype=torch.float32)


class CpuAttention(nn.Module):
    """Checkpoint names (`wqkv`, `wo`). No TP linears and no CUDA attention."""

    def __init__(self, args):
        super().__init__()
        dim = args.vision_dim
        self.n_heads = args.vision_n_heads
        self.head_dim = dim // self.n_heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        qkv = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = apply_rotary(q.view(-1, self.n_heads, self.head_dim), cos, sin)
        k = apply_rotary(k.view(-1, self.n_heads, self.head_dim), cos, sin)
        v = v.view(-1, self.n_heads, self.head_dim)
        out = _chunked_sdpa(q, k, v)
        return self.wo(out.reshape(x.shape[0], -1))


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """One GEMM. The master weight stays on CPU; this copy dies with the op."""
    w = weight.detach().to(device=x.device, dtype=x.dtype)
    b = None if bias is None else bias.detach().to(device=x.device, dtype=x.dtype)
    return F.linear(x, w, b)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = weight.detach().to(device=x.device, dtype=torch.float32)
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * w).to(dtype=x.dtype)


def _streaming_block(block, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = _rms(x, block.norm1.weight)
    qkv = _linear(h, block.attn.wqkv.weight, block.attn.wqkv.bias)
    q, k, v = qkv.chunk(3, dim=-1)
    heads, head_dim = block.attn.n_heads, block.attn.head_dim
    q = apply_rotary(q.view(-1, heads, head_dim), cos, sin)
    k = apply_rotary(k.view(-1, heads, head_dim), cos, sin)
    v = v.view(-1, heads, head_dim)
    attn = _chunked_sdpa(q, k, v).reshape(x.shape[0], -1)
    x = x + _linear(attn, block.attn.wo.weight, block.attn.wo.bias)
    h = _rms(x, block.norm2.weight)
    gate, up = _linear(h, block.mlp.w1.weight, None).chunk(2, dim=-1)
    return x + _linear(F.silu(gate) * up, block.mlp.w2.weight, None)


def _chunked_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q/k/v: [seq, heads, dim]. Score one query tile so the matrix stays small.
    scale = q.shape[-1] ** -0.5
    pieces = []
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    for start in range(0, q.shape[0], _CPU_QUERY_CHUNK):
        q_b = q[start : start + _CPU_QUERY_CHUNK].transpose(0, 1).unsqueeze(0)
        piece = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=scale)
        pieces.append(piece.squeeze(0).transpose(0, 1))
    return torch.cat(pieces, dim=0)


class CpuBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = _rms_norm(args.vision_dim)
        self.attn = CpuAttention(args)
        self.norm2 = _rms_norm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class CpuViT(nn.Module):
    """One fp32 tower. Parameter names match the checkpoint."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList(
            [CpuBlock(args) for _ in range(args.vision_n_layers)]
        )
        self.norm = _rms_norm(args.vision_dim)
        _as_fp32_cpu(self)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches.float())
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def streaming_gpu(self, patches: torch.Tensor, n_h: int, n_w: int, device: torch.device) -> torch.Tensor:
        """Same tower, fp16 on one GPU. Weights are copied per GEMM and not kept."""
        x = patches.to(device=device, dtype=torch.float16).flatten(1)
        x = _linear(x, self.patch_embed.proj.weight, self.patch_embed.proj.bias)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=device)
        sin = sin.to(device=device)
        for block in self.blocks:
            x = _streaming_block(block, x, cos, sin)
        return _rms(x, self.norm.weight)


class CpuAligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        hidden = args.hidden_size
        self.w1 = nn.Linear(in_dim, hidden, bias=True)
        self.w2 = nn.Linear(hidden, hidden, bias=True)
        _as_fp32_cpu(self)

    def _project(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        return F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self._project(x, n_h, n_w)
        return self.w2(F.gelu(self.w1(x)))

    def streaming_gpu(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self._project(x, n_h, n_w)
        hidden = F.gelu(_linear(x, self.w1.weight, self.w1.bias))
        return _linear(hidden, self.w2.weight, self.w2.bias)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        hidden = getattr(args, "hidden_size", None)
        if hidden is None:
            hidden = args.dim
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
