from __future__ import annotations

import torch


def torch_sparse_mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    query_chunk_size: int = 4,
    topk_chunk_size: int = 128,
) -> torch.Tensor:
    """BF16 sparse MLA correctness fallback with chunked online softmax.

    Query rows and selected KV positions are processed in independent chunks.
    The largest gathered KV temporary is therefore
    ``[query_chunk_size, topk_chunk_size, 576]`` rather than
    ``[T, H, K, 576]``. ``-1`` (and defensive out-of-range) indices are ignored.
    An all-invalid row produces zeros.
    """

    if q_nope.ndim != 3 or q_rope.ndim != 3:
        raise ValueError("torch sparse-MLA expects q_nope/q_rope shaped [T, H, D]")
    if q_nope.shape[:2] != q_rope.shape[:2]:
        raise ValueError(
            "torch sparse-MLA expects matching query token/head dimensions, "
            f"got {tuple(q_nope.shape)} and {tuple(q_rope.shape)}"
        )
    if q_nope.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
        raise ValueError(
            "torch sparse-MLA currently requires BF16 queries, "
            f"got {q_nope.dtype} and {q_rope.dtype}"
        )
    if kv_cache.ndim != 3 or kv_cache.shape[1] != 1:
        raise ValueError(
            "torch sparse-MLA expects kv_cache shaped [N, 1, D], "
            f"got {tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype != torch.bfloat16:
        raise ValueError(
            "torch sparse-MLA currently requires a BF16 KV cache, "
            f"got {kv_cache.dtype}"
        )
    if kv_cache.shape[0] == 0:
        raise ValueError("torch sparse-MLA expects a non-empty KV cache")
    expected_kv_dim = q_nope.shape[-1] + q_rope.shape[-1]
    if kv_cache.shape[-1] != expected_kv_dim:
        raise ValueError(
            "torch sparse-MLA query/KV dimensions do not match: expected "
            f"KV dim {expected_kv_dim}, got {kv_cache.shape[-1]}"
        )
    if indices.ndim != 2 or indices.shape[0] != q_nope.shape[0]:
        raise ValueError(
            "torch sparse-MLA expects indices shaped [T, K], "
            f"got {tuple(indices.shape)} for T={q_nope.shape[0]}"
        )
    if query_chunk_size <= 0 or topk_chunk_size <= 0:
        raise ValueError("query_chunk_size and topk_chunk_size must be positive")

    num_queries, num_heads, value_dim = q_nope.shape
    output = torch.zeros_like(q_nope)
    if num_queries == 0 or indices.shape[1] == 0:
        return output

    kv = kv_cache[:, 0]
    for query_start in range(0, num_queries, query_chunk_size):
        query_end = min(query_start + query_chunk_size, num_queries)
        qn = q_nope[query_start:query_end].float()
        qr = q_rope[query_start:query_end].float()
        row_indices = indices[query_start:query_end]

        running_max = torch.full(
            (query_end - query_start, num_heads),
            float("-inf"),
            dtype=torch.float32,
            device=q_nope.device,
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            (query_end - query_start, num_heads, value_dim),
            dtype=torch.float32,
            device=q_nope.device,
        )

        for topk_start in range(0, indices.shape[1], topk_chunk_size):
            topk_end = min(topk_start + topk_chunk_size, indices.shape[1])
            selected = row_indices[:, topk_start:topk_end].to(torch.long)
            valid = (selected >= 0) & (selected < kv.shape[0])
            selected = selected.clamp(0, kv.shape[0] - 1)
            gathered = kv[selected].float()
            kn = gathered[..., :value_dim]
            kr = gathered[..., value_dim:]

            scores = (
                torch.einsum("qhd,qkd->qhk", qn, kn)
                + torch.einsum("qhd,qkd->qhk", qr, kr)
            ) * sm_scale
            scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))

            chunk_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, chunk_max)
            safe_max = torch.where(
                torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
            )
            old_scale = torch.exp(running_max - safe_max)
            probabilities = torch.exp(scores - safe_max.unsqueeze(-1))

            accumulator = accumulator * old_scale.unsqueeze(-1)
            accumulator += torch.einsum("qhk,qkd->qhd", probabilities, kn)
            running_sum = running_sum * old_scale + probabilities.sum(dim=-1)
            running_max = new_max

        normalized = accumulator / running_sum.clamp_min(
            torch.finfo(torch.float32).tiny
        ).unsqueeze(-1)
        normalized = torch.where(
            (running_sum > 0).unsqueeze(-1), normalized, torch.zeros_like(normalized)
        )
        output[query_start:query_end] = normalized.to(output.dtype)

    return output
