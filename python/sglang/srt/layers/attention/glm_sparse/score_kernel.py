"""Triton scoring kernel for GLM sparse training-free indexer.

Replaces the Python loop in
`Glm4MoeAttention._compute_glm_sparse_training_free_flat_topk`. For each
``(batch, kv_head)`` pair it computes the per-query-head dot products
``<q[b, kv_head*group + g], K[b, t, kv_head]>`` for every query head ``g`` in
the GQA group and **max-pools over the group**:
``scores[b*H+h, t] = max_g <q[b, h*group+g], K[b, t, h]>``. Scoring uses
sglang's flat KV pool plus ``req_to_token`` for logical->physical address
translation.

Doing the GEMM per query head and max-pooling afterwards (rather than mean-
pooling the query first) keeps each head's full signal and selects the union of
tokens any head in the group considers important. The K block is loaded once
per ``(batch, kv_head)`` tile and reused across the group, so the extra cost is
only the group-many cheap dot products.

Inspired by FastDeploy's ``indexer_mha_page_logits`` Triton kernel; rewritten
to (a) use torch+sglang flat KV layout, (b) support GQA by max-pooling the
per-head scores down to ``[batch, kv_heads, ...]`` (no per-head ``weight``
learned head), (c) emit scores in the ``[batch*kv_heads, max_score_len]``
layout that ``sgl_kernel.fast_topk_v2`` consumes directly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _glm_sparse_score_kernel(
    q_ptr,
    k_cache_ptr,
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    scores_ptr,
    stride_q_b: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_r2t_req: tl.constexpr,
    stride_score_row: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    block_m: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    force_left: tl.constexpr,
    force_right: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    if seq_len <= 0:
        return

    n_start = pid_n * block_n
    if n_start >= seq_len:
        return

    offs_n = n_start + tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)
    offs_m = tl.arange(0, block_m)
    valid_n = offs_n < seq_len
    valid_m = offs_m < group_size

    # Load this (batch, kv_head) K tile once; reuse it across all query heads in the GQA group.
    # K is shared within a group, so we compute the whole group's scores with one tensor-core GEMM.
    req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
    phys_ptrs = req_to_token_ptr + req_idx * stride_r2t_req + offs_n
    phys_loc = tl.load(phys_ptrs, mask=valid_n, other=0).to(tl.int64)
    k_ptrs = (
        k_cache_ptr
        + phys_loc[:, None] * stride_k_t
        + pid_h * stride_k_h
        + offs_d[None, :]
    )
    # K tile [block_n, head_dim] in bf16 (kept low-precision for tensor-core dot).
    k_vec = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)

    # Load the whole GQA group's queries as a [block_m, head_dim] tile (block_m padded
    # up to a tensor-core-friendly size; rows >= group_size are masked to 0 and ignored
    # in the post-GEMM max-pool). q layout is [batch, num_heads, head_dim] with
    # num_heads = kv_heads*group; the group for kv-head pid_h is the contiguous block
    # [pid_h*group, pid_h*group + group).
    q_head_base = pid_h * group_size
    q_ptrs = (
        q_ptr
        + pid_b * stride_q_b
        + (q_head_base + offs_m[:, None]) * stride_q_h
        + offs_d[None, :]
    )
    q_tile = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    # One tensor-core GEMM: [block_m, head_dim] @ [head_dim, block_n] -> [block_m, block_n].
    # Cast both operands to bf16 so tl.dot uses tensor cores (accumulates in fp32).
    qk = tl.dot(q_tile.to(tl.bfloat16), tl.trans(k_vec).to(tl.bfloat16))  # [block_m, block_n] fp32

    # Mask padded query rows (>= group_size) to -inf so they don't win the max-pool,
    # then max-pool over the group dimension to get per-token scores.
    qk = tl.where(valid_m[:, None], qk, float("-inf"))
    scores_block = tl.max(qk, axis=0)  # [block_n]
    # KV positions beyond seq_len must stay -inf for the downstream topk padding.
    scores_block = tl.where(valid_n, scores_block, float("-inf"))

    # Force-select mask (StreamingLLM-style sink + local window): the first
    # ``force_left`` tokens and the last ``force_right`` history tokens are
    # forced into the TopK by setting their score to +inf. This only steers the
    # *selection*; FA3 recomputes the real Q*K^T weights on the gathered KV, so
    # +inf never reaches the attention softmax. ``forced & valid_n`` keeps the
    # forcing strictly inside ``[0, seq_len)``; out-of-range columns stay -inf.
    if force_left > 0 or force_right > 0:
        is_left = offs_n < force_left
        is_right = offs_n >= (seq_len - force_right)
        forced = (is_left | is_right) & valid_n
        scores_block = tl.where(forced, float("inf"), scores_block)

    out_row = pid_b * num_kv_heads + pid_h
    out_ptrs = scores_ptr + out_row * stride_score_row + offs_n
    tl.store(out_ptrs, scores_block, mask=valid_n)


def glm_sparse_compute_scores(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    max_score_len: int,
    block_n: int = 64,
    force_left: int = 0,
    force_right: int = 0,
) -> torch.Tensor:
    """Compute per-(batch, kv_head, token) max-pooled scores against a flat KV pool.

    For each ``(batch, kv_head, token)`` the kernel computes the dot product of
    every query head in the kv-head's GQA group against the shared K, then
    **max-pools over the group**:
    ``scores[b*H+h, t] = max_g <q[b, h*group+g], K[b, t, h]>``.

    Args:
        q: ``[batch, num_heads, head_dim]`` per-query-head query (NOT pooled).
            ``num_heads`` must be a multiple of ``k_cache``'s ``kv_heads``;
            ``group = num_heads // kv_heads`` query heads map to each kv head.
        k_cache: ``[pool_size, kv_heads, head_dim]`` flat KV pool slice.
        req_to_token: ``[num_req, max_context_len]`` logical->physical map.
        req_pool_indices: ``[batch]`` index into ``req_to_token``.
        seq_lens: ``[batch]`` int32 visible KV length per request.
        max_score_len: width of the output score matrix (must be >=
            ``seq_lens.max()``; the natural choice is ``req_to_token.shape[1]``).
        block_n: tile size along the KV-token axis.
        force_left: number of leading history tokens ``[0, force_left)`` to force
            into the TopK (attention sink). ``0`` disables.
        force_right: number of trailing history tokens
            ``[seq_len - force_right, seq_len)`` to force into the TopK (local
            window). ``0`` disables. Forced positions get score ``+inf``; this
            steers selection only and never reaches the FA3 softmax.

    Returns:
        ``scores [batch*kv_heads, max_score_len]`` fp32. Positions beyond each
        row's ``seq_len`` are filled with ``-inf`` so downstream topk treats
        them as padding.
    """
    assert q.dim() == 3, f"q must be [batch, num_heads, head_dim], got {q.shape}"
    batch, num_heads, head_dim = q.shape
    assert k_cache.dim() == 3 and k_cache.shape[2] == head_dim, (
        f"k_cache shape mismatch: expected [*, kv_heads, {head_dim}], got {tuple(k_cache.shape)}"
    )
    kv_heads = k_cache.shape[1]
    assert num_heads % kv_heads == 0, (
        f"num_heads={num_heads} must be divisible by kv_heads={kv_heads}"
    )
    group_size = num_heads // kv_heads
    assert req_pool_indices.shape[0] == batch
    assert seq_lens.shape[0] == batch
    assert max_score_len > 0
    assert force_left >= 0 and force_right >= 0
    assert force_left + force_right <= max_score_len, (
        f"force_left+force_right={force_left + force_right} exceeds max_score_len={max_score_len}"
    )

    scores = torch.full(
        (batch * kv_heads, max_score_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    q_for_kernel = q.contiguous()
    req_pool_i32 = req_pool_indices.to(torch.int32).contiguous()
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()

    # tl.dot requires the M (group) dim to be a tensor-core-friendly size (>= 16).
    # group_size for GLM is 12, so pad M up to 16; padded rows are masked out.
    block_m = max(16, triton.next_power_of_2(group_size))

    grid = (
        triton.cdiv(max_score_len, block_n),
        kv_heads,
        batch,
    )
    _glm_sparse_score_kernel[grid](
        q_for_kernel,
        k_cache,
        req_to_token,
        req_pool_i32,
        seq_lens_i32,
        scores,
        stride_q_b=q_for_kernel.stride(0),
        stride_q_h=q_for_kernel.stride(1),
        stride_k_t=k_cache.stride(0),
        stride_k_h=k_cache.stride(1),
        stride_r2t_req=req_to_token.stride(0),
        stride_score_row=scores.stride(0),
        num_kv_heads=kv_heads,
        group_size=group_size,
        block_m=block_m,
        head_dim=head_dim,
        block_n=block_n,
        force_left=force_left,
        force_right=force_right,
    )
    return scores
