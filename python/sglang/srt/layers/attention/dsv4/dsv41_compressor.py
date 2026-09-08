"""Pure-torch DeepSeek V4.1 low-ratio compressor and token-pairing helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    """fp32 statistics and fp32 weight multiply, cast back at the very end."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def last_token_per_request(mask: torch.Tensor, req: torch.Tensor) -> torch.Tensor:
    """Among tokens selected by mask, keep only the last one of each request."""
    idx = mask.nonzero().squeeze(1)
    if idx.numel() == 0:
        return mask
    order = torch.argsort(idx)
    idx = idx[order]
    r = req[idx]
    is_last = torch.ones_like(idx, dtype=torch.bool)
    is_last[:-1] = r[:-1] != r[1:]
    out = torch.zeros_like(mask)
    out[idx[is_last]] = True
    return out


def pair_partners_decode(
    kv: torch.Tensor,
    score: torch.Tensor,
    odd: torch.Tensor,
    req: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ratio-2 pairing for decode, one token per request: an odd position reads
    its partner from the per-request state, an even one parks itself there.
    `req` must be unique per row (padded rows go to a spare row). A partner is
    returned for every row; only odd rows complete a group."""
    partner_kv = state_kv[req]
    partner_score = state_score[req]
    keep = odd.unsqueeze(-1)
    state_kv[req] = torch.where(keep, partner_kv, kv)
    state_score[req] = torch.where(keep, partner_score, score)
    return partner_kv, partner_score


class DeepseekV41Compressor(nn.Module):
    """Pools compress_ratio consecutive tokens into one pre-RoPE KV latent.
    Ratio 1 is a plain bf16 projection; ratio 2 gates two tokens with a softmax
    over their fp32 scores."""

    def __init__(
        self, hidden_size: int, head_dim: int, compress_ratio: int, eps: float
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.norm = RMSNorm(head_dim, eps)
        proj_dtype = torch.float32 if compress_ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(hidden_size, head_dim, bias=False, dtype=proj_dtype)
        if compress_ratio > 1:
            self.wgate = nn.Linear(
                hidden_size, head_dim, bias=False, dtype=torch.float32
            )

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.compress_ratio == 1:
            return self.wkv(x), None
        x = x.float()
        return self.wkv(x), self.wgate(x)

    def finish(self, kv: torch.Tensor) -> torch.Tensor:
        return self.norm(kv.to(torch.bfloat16))

    @staticmethod
    def pool_pairs(kv2: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
        """kv2, score2 [n, 2, D] fp32 -> [n, D]"""
        return (kv2 * score2.softmax(dim=1)).sum(dim=1)
