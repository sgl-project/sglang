"""Normalize compact top-k logits and scatter top-k-first probabilities."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_topk_probs(
    logits_ptr,
    indices_ptr,
    temperatures_ptr,
    top_ks_ptr,
    top_ps_ptr,
    output_ptr,
    K: tl.constexpr,
    VOCAB: tl.constexpr,
    LOGIT_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    APPLY_TOP_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rank = tl.arange(0, BLOCK)
    top_k = tl.load(top_ks_ptr + row)
    temperature = tl.load(temperatures_ptr + row)
    logits = tl.load(logits_ptr + row * LOGIT_STRIDE + rank, rank < K, other=0)
    scaled = tl.where((rank < K) & (rank < top_k), logits / temperature, -float("inf"))
    weights = tl.exp(scaled - tl.max(scaled, 0))
    probs = weights / tl.sum(weights, 0)
    if APPLY_TOP_P:
        top_p = tl.load(top_ps_ptr + row)
        mass_before = tl.cumsum(probs, 0) - probs
        # Match probability-threshold filtering, including ties at the cutoff.
        cutoff = tl.min(
            tl.where((rank < K) & (mass_before <= top_p), probs, float("inf")), 0
        )
        probs = tl.where(probs >= cutoff, probs, 0.0)
        probs /= tl.sum(probs, 0)
    indices = tl.load(indices_ptr + row * INDEX_STRIDE + rank, rank < K, other=0)
    tl.store(output_ptr + row * VOCAB + indices, probs, rank < K)


def scatter_top_k_top_p_probs(
    topk_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: Optional[torch.Tensor],
    vocab_size: int,
) -> torch.Tensor:
    """Scatter sorted compact logits into dense top-k-first probabilities.

    CUDA fp32 logits and their token indices have shape [rows, k], k <= 128,
    with contiguous columns. Temperatures are positive fp32 [rows, 1]; top-k
    limits and optional fp32 top-p thresholds are contiguous vectors of length
    rows. Indices are unique within each row and lie in [0, vocab_size).
    Each retained support must contain a finite logit; -inf masks are supported.
    """
    rows, k = topk_logits.shape
    assert topk_logits.is_cuda and topk_logits.dtype == torch.float32
    assert 1 <= k <= min(vocab_size, 128)
    assert topk_indices.shape == (rows, k)
    assert topk_indices.dtype in (torch.int32, torch.int64)
    assert topk_logits.stride(1) == topk_indices.stride(1) == 1
    assert temperatures.shape == (rows, 1) and temperatures.is_contiguous()
    assert temperatures.dtype == torch.float32
    assert top_ks.shape == (rows,) and top_ks.is_contiguous()
    assert top_ks.dtype in (torch.int32, torch.int64)
    for tensor in (topk_indices, temperatures, top_ks):
        assert tensor.device == topk_logits.device
    if top_ps is not None:
        assert top_ps.shape == (rows,) and top_ps.is_contiguous()
        assert top_ps.dtype == torch.float32 and top_ps.device == topk_logits.device
    output = torch.zeros(
        (rows, vocab_size), dtype=torch.float32, device=topk_logits.device
    )
    if rows:
        _scatter_topk_probs[(rows,)](
            topk_logits,
            topk_indices,
            temperatures,
            top_ks,
            top_ps,
            output,
            k,
            vocab_size,
            topk_logits.stride(0),
            topk_indices.stride(0),
            top_ps is not None,
            triton.next_power_of_2(k),
        )
    return output
