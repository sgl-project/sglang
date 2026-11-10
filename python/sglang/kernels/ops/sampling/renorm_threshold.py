"""Threshold-based top-k / top-p probability renormalization.

The classic formulation masks in the sorted space and scatters the kept values
back to their original token positions
(``torch.zeros_like(probs).scatter_(dim=-1, index=sorted_idx, src=...)``).
On Ascend NPU that full-vocabulary random write dominates the op: roughly 85%
of its cost, ~3.1 ms for a 4 x 154880 row block. The threshold form below
replaces it with a compare and a multiply:

    keep = probs >= pivot,   pivot = smallest kept *unrenormalized* probability

``pivot`` is read from the sorted probabilities before renormalization, so it
lives on the same scale as ``probs``; the kept set is the one the scatter form
would write back, and the row is renormalized once at the end. Only aten ops are
used, so the module is backend-agnostic, and it is safe under graph capture: the
caller's input tensor is never mutated and the result is a fresh tensor.

Ties: when several tokens hold exactly the pivot probability, the threshold form
keeps all of them while the scatter form keeps an arbitrary subset of that tie
group. Both describe the same cut in value space; with the fp32 softmax outputs
SGLang produces, exact ties at a non-zero probability do not occur in practice.
"""

from typing import Union

import torch

__all__ = ["top_k_renorm_prob", "top_p_renorm_prob"]


def _renorm_from_sorted_probs(
    probs: torch.Tensor,
    sorted_probs: torch.Tensor,
) -> torch.Tensor:
    # ``sorted_probs`` already has the masked-out entries zeroed by the caller,
    # so the smallest non-zero entry is the last kept (unrenormalized) value.
    nonzero_counts = (sorted_probs > 0).sum(dim=-1)  # [B]
    cutoff_idx = (nonzero_counts - 1).clamp(min=0)  # [B]
    pivot = sorted_probs.gather(1, cutoff_idx.view(-1, 1)).squeeze(1)  # [B]
    result = probs.masked_fill(probs < pivot.view(-1, 1), 0.0)
    result.div_(result.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
    return result


def _as_batch_threshold(
    value: Union[torch.Tensor, int, float],
    probs: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.full((probs.shape[0],), value, device=probs.device, dtype=dtype)
    if value.ndim == 0:
        return value.to(device=probs.device, dtype=dtype).expand(probs.shape[0])
    if value.ndim != 1 or value.shape[0] != probs.shape[0]:
        raise ValueError(f"{name} must be a scalar or have shape [batch]")
    return value.to(device=probs.device, dtype=dtype)


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_ks: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Keep each row's top-k probabilities and renormalize the row."""
    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, vocab]")

    vocab_size = probs.shape[-1]
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    top_ks = _as_batch_threshold(top_ks, probs, torch.long, "top_ks").clamp(
        min=1, max=vocab_size
    )
    positions = torch.arange(vocab_size, device=probs.device).view(1, -1)
    sorted_probs.masked_fill_(positions >= top_ks.view(-1, 1), 0.0)
    return _renorm_from_sorted_probs(probs, sorted_probs)


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_ps: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep each row's nucleus probabilities and renormalize the row."""
    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, vocab]")

    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(
        min=0.0, max=1.0
    )
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_probs.masked_fill_(cumulative_probs - sorted_probs > top_ps.view(-1, 1), 0.0)
    return _renorm_from_sorted_probs(probs, sorted_probs)
