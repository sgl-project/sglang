"""Vision attention used by the native Dots Omni vision encoder."""

from __future__ import annotations

from typing import Any

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """2D vision RoPE frequency table with optional caching."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        cache_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_seq_len = cache_seq_len
        if cache_seq_len is not None:
            self.register_buffer(
                "freqs_cache", self._compute_freqs(cache_seq_len), persistent=False
            )

    def _compute_freqs(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(seq, self.inv_freq)

    def forward(self, seqlen: int) -> torch.Tensor:
        if self._cache_seq_len is None:
            return self._compute_freqs(seqlen)
        if seqlen > self.freqs_cache.shape[0]:
            self.freqs_cache = self._compute_freqs(seqlen)
        return self.freqs_cache[:seqlen]


class _RMSNorm(nn.Module):
    """Q/K norm matching Dots ViT's fp32 reduction semantics."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class VisionAttention(nn.Module):
    """QKV projection followed by SGLang varlen flash attention."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.is_causal = config.is_causal
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        bias = getattr(config, "use_bias", True)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        if self.use_qk_norm:
            head_dim = dim // self.num_heads
            self.q_norm = _RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = _RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=self.is_causal,
        )
        return self.proj(attn_output.reshape(seq_length, -1))


def apply_vision_attention_residual(
    attn: nn.Module,
    norm: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_pos_emb: torch.Tensor,
) -> torch.Tensor:
    return hidden_states + attn(
        norm(hidden_states),
        cu_seqlens,
        max_seqlen,
        rotary_pos_emb,
    )


__all__ = [
    "VisionAttention",
    "VisionRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
    "apply_vision_attention_residual",
    "rotate_half",
]
