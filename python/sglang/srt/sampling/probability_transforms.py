"""Probability cutoffs shared by speculative proposals and verification."""

from __future__ import annotations

import torch


def top_k_renorm_probs(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    top_ks = top_ks.reshape(-1).clamp(min=1, max=probs.shape[-1])
    if probs.is_cuda:
        if torch.version.hip is not None:
            from sglang.kernels.ops.sampling.renorm_triton import (
                top_k_renorm_probs_triton,
            )

            return top_k_renorm_probs_triton(probs, top_ks)
        from flashinfer.sampling import top_k_renorm_probs as renorm

        return renorm(probs, top_ks)
    if probs.device.type == "musa":
        from sgl_kernel import top_k_renorm_prob

        return top_k_renorm_prob(probs, top_ks)
    # Keep the fallback graph-safe: no host read of the per-request k values.
    sorted_probs, indices = probs.sort(dim=-1, descending=True)
    ranks = torch.arange(probs.shape[-1], device=probs.device)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_ks[:, None], 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, indices, sorted_probs)


def top_p_renorm_probs(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    if probs.is_cuda:
        if torch.version.hip is not None:
            from sglang.kernels.ops.sampling.renorm_triton import (
                top_p_renorm_probs_triton,
            )

            return top_p_renorm_probs_triton(probs, top_ps)
        from flashinfer.sampling import top_p_renorm_probs as renorm

        return renorm(probs, top_ps)
    if probs.device.type == "musa":
        from sgl_kernel import top_p_renorm_prob

        return top_p_renorm_prob(probs, top_ps)
    sorted_probs, indices = probs.sort(dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_probs = sorted_probs.masked_fill(
        cumulative_probs - sorted_probs > top_ps.reshape(-1, 1), 0.0
    )
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, indices, sorted_probs)
