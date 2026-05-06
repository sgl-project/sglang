"""Fused Triton kernel for Quest's per-page bounding-box scoring.

Replaces the PyTorch op chain in retrieve_topk (gather + where + multiply +
sum) with a single Triton kernel.  Each program block handles one request
× BLOCK_PAGES pages.  The kernel reads:

  Q[req] : [kv_heads, head_dim] (post-GQA mean)
  k_bounds[layer, req, page, kv_heads, head_dim, 2]

and writes ``scores[req, page] = Σ_d q[d] * (q[d] >= 0 ? k_max[d] : k_min[d])``.

Sentinel (+inf min, -inf max) for invalid pages naturally yields a score
of -inf, so retrieve_topk's downstream topk picks valid pages.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_score_kernel(
    q_ptr,                  # [bs, KV_HEADS_HEAD_DIM]    float32
    k_bounds_ptr,           # [NUM_LAYERS, MAX_REQS, MAX_PAGES, KV_HEADS, HEAD_DIM, 2] bfloat16
    req_pool_indices_ptr,   # [bs]                       int64
    scores_ptr,             # [bs, MAX_PAGES]            float32
    layer_offset: tl.constexpr,
    MAX_REQS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    KV_HEADS_HEAD_DIM: tl.constexpr,  # kv_heads * head_dim, must be power of 2
):
    """One block per (req, page).  Threads cooperatively dot Q with the
    chosen-bound K vector for that page; result is one scalar score.

    Each block reads:
      - Q[req]: KV_HEADS_HEAD_DIM float32 (~512 elements)
      - k_bounds[layer, req, page, :, :, 2]: KV_HEADS_HEAD_DIM*2 bf16
        (we read both [min, max] and pick per element via bound_sel)

    Writes one float32 score.
    """
    pid_req = tl.program_id(0)
    pid_page = tl.program_id(1)

    if pid_page >= MAX_PAGES:
        return

    # Load Q[req] (float32, post-GQA mean).
    d_off = tl.arange(0, KV_HEADS_HEAD_DIM)
    q_offsets = pid_req * KV_HEADS_HEAD_DIM + d_off
    q = tl.load(q_ptr + q_offsets)  # [KV_HEADS_HEAD_DIM] float32
    bound_sel = (q >= 0).to(tl.int32)  # 1 = pick max (idx 1), 0 = pick min (idx 0)

    req_pool_idx = tl.load(req_pool_indices_ptr + pid_req)

    # Offset into k_bounds for this (layer, req, page).
    # Layout: [NUM_LAYERS, MAX_REQS, MAX_PAGES, KV_HEADS_HEAD_DIM, 2].
    base = (
        tl.cast(layer_offset, tl.int64) * MAX_REQS * MAX_PAGES * KV_HEADS_HEAD_DIM * 2
        + req_pool_idx * (MAX_PAGES * KV_HEADS_HEAD_DIM * 2)
        + tl.cast(pid_page, tl.int64) * (KV_HEADS_HEAD_DIM * 2)
    )
    # Per-element offset: d * 2 + bound_sel[d]
    d_offsets = base + d_off * 2 + bound_sel  # [KV_HEADS_HEAD_DIM]

    k_chosen = tl.load(k_bounds_ptr + d_offsets).to(tl.float32)

    # Dot product → scalar score for this page.
    score = tl.sum(q * k_chosen)

    score_offset = pid_req * MAX_PAGES + pid_page
    tl.store(scores_ptr + score_offset, score)


def quest_score(
    q: torch.Tensor,
    page_k_bounds: torch.Tensor,
    req_pool_indices: torch.Tensor,
    layer_offset: int,
) -> torch.Tensor:
    """Compute per-page Quest scores via the fused Triton kernel.

    Args:
      q: ``[bs, kv_heads, head_dim]`` float32 (post-GQA mean).
      page_k_bounds: ``[num_layers, max_reqs, max_pages, kv_heads, head_dim, 2]``
        bfloat16.  Last axis: 0 = min, 1 = max.
      req_pool_indices: ``[bs]`` int64.
      layer_offset: int (layer_id - start_layer).

    Returns:
      scores: ``[bs, max_pages]`` float32.
    """
    bs = q.shape[0]
    num_layers, max_reqs, max_pages, kv_heads, head_dim, _ = page_k_bounds.shape
    kv_heads_head_dim = kv_heads * head_dim

    assert q.shape == (bs, kv_heads, head_dim), f"q shape {q.shape}"
    assert q.dtype == torch.float32, f"q must be float32, got {q.dtype}"
    assert q.is_contiguous(), "q must be contiguous"
    assert page_k_bounds.is_contiguous(), "page_k_bounds must be contiguous"

    scores = torch.empty(
        (bs, max_pages), dtype=torch.float32, device=q.device,
    )

    grid = (bs, max_pages)

    _quest_score_kernel[grid](
        q.view(bs, kv_heads_head_dim),
        page_k_bounds,
        req_pool_indices,
        scores,
        layer_offset=layer_offset,
        MAX_REQS=max_reqs,
        MAX_PAGES=max_pages,
        KV_HEADS_HEAD_DIM=kv_heads_head_dim,
        num_warps=4,
    )
    return scores
