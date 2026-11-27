from __future__ import annotations

import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def indexer_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rope(x: torch.Tensor) -> torch.Tensor:
        rope_dim = cache.shape[1]
        half = rope_dim // 2
        cos_sin = cache[positions.long()].float()
        cos, sin = cos_sin[:, None, :half], cos_sin[:, None, half:]
        first, second = x[..., :half], x[..., half:rope_dim]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), -1
        )
        return torch.cat((rotated, x[..., rope_dim:]), -1).bfloat16()

    xq = q.float()
    q_norm = xq * torch.rsqrt(xq.pow(2).mean(-1, keepdim=True) + eps)
    q_norm = (q_norm * (q_weight.float() + 1)).bfloat16().float()
    xk = k.float()
    mean = xk.mean(-1, keepdim=True)
    variance = xk.pow(2).mean(-1, keepdim=True) - mean * mean
    k_norm = (xk - mean) * torch.rsqrt(variance + eps)
    k_norm = (k_norm * k_weight.float() + k_bias.float()).bfloat16().float()
    return rope(q_norm), rope(k_norm)


def compress_region(index_keys: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logits.float(), dim=0)
    key = (weights * index_keys.float()).sum(0).bfloat16().float()
    return key.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def block_scores(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    q = q.to(torch.float8_e4m3fn).float()
    products = torch.relu(q @ keys.float().T)
    return (products * weights.float()[:, None]).sum(0)


def sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    token_slots: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    k = k_cache[token_slots, 0].float()
    v = v_cache[token_slots, 0].float()
    probs = torch.softmax((q.float() @ k.T) * scale, dim=-1)
    return (probs @ v).to(q.dtype)
