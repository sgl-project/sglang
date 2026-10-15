"""Pure-torch top-k / top-p probability renormalization.

The reference for devices without a kernel: CPU, and NPU rows the vendor path
declines. Same threshold semantics as FlashInfer and the ROCm Triton fallback:
ties at the pivot are kept. Graph-safe: no host read of the per-row cutoffs.
"""

from __future__ import annotations

from typing import Union

import torch


def _per_row(value: Union[torch.Tensor, int, float], probs: torch.Tensor, dtype):
    rows = probs.shape[0]
    if not isinstance(value, torch.Tensor):
        return torch.full((rows,), value, device=probs.device, dtype=dtype)
    value = value.to(device=probs.device, dtype=dtype).reshape(-1)
    return value.expand(rows) if value.numel() == 1 else value


def _renorm_from_pivots(probs: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    kept = probs.masked_fill(probs < pivots, 0.0)
    return kept / kept.sum(dim=-1, keepdim=True)


def top_k_renorm_probs_torch(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Keep each row's ``top_k`` largest probabilities and renormalize."""
    probs = probs.float()
    vocab = probs.shape[-1]
    top_ks = _per_row(top_k, probs, torch.int64).clamp(min=1, max=vocab)
    sorted_probs = probs.sort(dim=-1, descending=True).values
    pivots = sorted_probs.gather(-1, (top_ks - 1).unsqueeze(-1))
    return _renorm_from_pivots(probs, pivots)


def top_p_renorm_probs_torch(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Keep each row's largest probabilities reaching mass ``top_p`` and renormalize."""
    probs = probs.float()
    top_ps = _per_row(top_p, probs, torch.float32)
    # Sort ascending and discard the prefix whose cumulative mass stays below 1 - p.
    sorted_probs = probs.sort(dim=-1).values
    cdf = sorted_probs.cumsum(dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(-1), right=False)
    pivots = sorted_probs.gather(-1, cutoff.clamp(max=probs.shape[-1] - 1))
    return _renorm_from_pivots(probs, pivots)


__all__ = ["top_k_renorm_probs_torch", "top_p_renorm_probs_torch"]
