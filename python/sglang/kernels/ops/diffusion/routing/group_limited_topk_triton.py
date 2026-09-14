"""Fused group-limited MoE top-k index selection for diffusion routers.

The reference LingBot Video router builds the group-limited top-k with a chain
of small kernels: per-group top-2 and sum, group top-k, a ``scatter_`` into a
zero mask, an ``expand``/``reshape`` broadcast, a ``masked_fill`` with
``-inf``, and the final expert top-k. On a launch-bound single GPU that chain
is pure overhead: every intermediate tensor is tiny and the whole computation
is bandwidth- and launch-bound. The later score gather remains in the caller.

This module fuses the entire selection into a single Triton kernel: one
program per token loads its score row once, reduces the per-group sums in
registers, masks non-selected groups with ``-inf``, and writes the top-k
expert ids. The selected expert-id set matches the reference CUDA
``torch.topk`` chain for the guarded layouts, including the production
128-expert / 4-group / 2-selected-group / top-8 configuration. The output
order is intentionally unspecified, matching the reference's ``sorted=False``
contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _group_limited_topk_kernel(
    scores_ptr,  # [T, E] f32, rows are scores_for_choice (scores + bias)
    out_idx_ptr,  # [T, TOP_K] i64
    stride_st,
    E: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOP_K: tl.constexpr,
    EPG: tl.constexpr,  # experts per group = E // N_GROUP
    BLOCK_E: tl.constexpr,  # padded E
    BLOCK_EPG: tl.constexpr,  # padded experts-per-group
    BLOCK_G: tl.constexpr,  # padded N_GROUP
):
    t = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    e_mask = offs_e < E
    scores = tl.load(
        scores_ptr + t * stride_st + offs_e, mask=e_mask, other=float("-inf")
    )

    # Per-group scores -> [BLOCK_G, BLOCK_EPG], pad with -inf so padded lanes
    # never win the per-group top-2 reduction.
    g = tl.reshape(scores, (BLOCK_G, BLOCK_EPG), can_reorder=False)
    epg_mask = tl.arange(0, BLOCK_EPG)[None, :] < EPG
    g = tl.where(epg_mask, g, float("-inf"))

    # group score = sum of top-2 experts within each group.
    group_e = tl.arange(0, BLOCK_EPG)[None, :]
    m1 = tl.max(g, axis=1)
    # Remove exactly one copy of the first maximum. Masking every value equal
    # to m1 would lose the second top-k entry when a group contains duplicate
    # maxima, which changes both the group score and the selected experts.
    m1_idx = tl.min(
        tl.where(g == m1[:, None], group_e, BLOCK_EPG),
        axis=1,
    )
    g2 = tl.where(group_e == m1_idx[:, None], float("-inf"), g)
    m2 = tl.max(g2, axis=1)
    group_scores = m1 + m2

    # Select TOPK_GROUP groups by descending group score with an explicit
    # left tie-break. Correctness tests compare the selected set because the
    # reference uses torch.topk(..., sorted=False).
    group_idx = tl.arange(0, BLOCK_G)
    gs_valid = tl.where(group_idx < N_GROUP, group_scores, float("-inf"))
    selected_group = tl.zeros((BLOCK_G,), dtype=tl.int1)
    for _ in tl.static_range(TOPK_GROUP):
        picked_idx = tl.argmax(
            gs_valid,
            axis=0,
            tie_break_left=True,
        )
        is_pick = group_idx == picked_idx
        selected_group = selected_group | is_pick
        gs_valid = tl.where(is_pick, float("-inf"), gs_valid)

    # Mask experts in non-selected groups, then flat top-k (same tie-break).
    masked = tl.where(selected_group[:, None], g, float("-inf"))
    flat = tl.reshape(masked, (BLOCK_E,), can_reorder=False)
    flat = tl.where(e_mask, flat, float("-inf"))
    for kk in tl.static_range(TOP_K):
        idx = tl.argmax(flat, axis=0, tie_break_left=True)
        tl.store(out_idx_ptr + t * TOP_K + kk, idx.to(tl.int64))
        flat = tl.where(offs_e == idx, float("-inf"), flat)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def can_use_group_limited_topk(
    scores_for_choice: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
) -> bool:
    """Return whether the fused CUDA path supports this routing problem."""
    if not scores_for_choice.is_cuda or torch.version.hip is not None:
        return False
    if scores_for_choice.ndim != 2 or scores_for_choice.dtype != torch.float32:
        return False
    if not scores_for_choice.is_contiguous() or scores_for_choice.shape[0] == 0:
        return False

    num_experts = scores_for_choice.shape[1]
    if n_group <= 1 or num_experts == 0 or num_experts % n_group != 0:
        return False
    experts_per_group = num_experts // n_group
    if experts_per_group < 2 or experts_per_group & (experts_per_group - 1):
        return False
    return 0 < topk_group <= n_group and 0 < top_k <= topk_group * experts_per_group


def _fake_group_limited_topk(
    scores_for_choice: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
) -> torch.Tensor:
    del n_group, topk_group
    return scores_for_choice.new_empty(
        (scores_for_choice.shape[0], top_k), dtype=torch.int64
    )


@register_custom_op(
    op_name="diffusion_group_limited_topk",
    mutates_args=[],
    fake_impl=_fake_group_limited_topk,
)
def _group_limited_topk_cuda(
    scores_for_choice: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
) -> torch.Tensor:
    t, e = scores_for_choice.shape
    epg = e // n_group
    out = torch.empty((t, top_k), dtype=torch.int64, device=scores_for_choice.device)
    _group_limited_topk_kernel[(t,)](
        scores_for_choice,
        out,
        scores_for_choice.stride(0),
        E=e,
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        TOP_K=top_k,
        EPG=epg,
        BLOCK_E=_next_pow2(e),
        BLOCK_EPG=_next_pow2(epg),
        BLOCK_G=_next_pow2(n_group),
        num_warps=4,
    )
    return out


def group_limited_topk(
    scores_for_choice: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
) -> torch.Tensor:
    """Fused group-limited top-k expert ids.

    ``scores_for_choice`` is the per-token expert score used for selection
    (already includes the correction bias), shape ``[T, E]`` float32. Returns
    the selected expert ids as ``[T, top_k]`` int64. The selected set matches
    the reference two-stage group-limited selection; output order is not part
    of the contract.
    """
    if not can_use_group_limited_topk(scores_for_choice, n_group, topk_group, top_k):
        raise ValueError(
            "group_limited_topk requires a nonempty contiguous CUDA float32 "
            "[tokens, experts] tensor, at least two power-of-two experts per "
            "group, 1 < n_group, 0 < topk_group <= n_group, and top_k no "
            "larger than the selected-group capacity"
        )
    return _group_limited_topk_cuda(scores_for_choice, n_group, topk_group, top_k)
