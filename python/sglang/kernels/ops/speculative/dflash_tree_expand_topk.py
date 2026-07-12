# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Triton kernels for DFlash tree verify with topk == 4.

- ``dflash_expand_topk4``: one expand + top-4 (scores ⊗ step probs).
- ``dflash_tree_verify_select_topk4_fused``: full ``num_steps`` recurrence in one kernel
  (the Python step loop runs inside Triton to cut launch / sync overhead).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# Host Python uses ints; Triton requires constexpr globals as tl.constexpr(...).
_DFLASH_EXPAND_TOPK = 4
_DFLASH_EXPAND_TOPK_SQ = 16
TOPK = tl.constexpr(4)
TOPK_SQ = tl.constexpr(16)


@triton.jit
def _dflash_expand_topk4_kernel(
    scores_ptr,
    topk_p_ptr,
    out_expand_ptr,
    out_topv_ptr,
    out_topi_ptr,
    stride_scores_b,
    stride_topk_p_r,
    stride_topk_p_k,
    stride_expand_b,
    stride_topv_b,
    stride_topi_b,
    bs,
):
    """Grid x: batch index. Computes [bs, 4] x [bs*4, 4] -> expand [bs, 16], top-4 by value."""
    b = tl.program_id(0)
    if b >= bs:
        return

    offs = tl.arange(0, TOPK_SQ)
    p = offs // TOPK
    c = offs % TOPK

    sc = tl.load(
        scores_ptr + b * stride_scores_b + p,
    ).to(tl.float32)
    tp = tl.load(
        topk_p_ptr
        + (b * TOPK + p) * stride_topk_p_r
        + c * stride_topk_p_k,
    ).to(tl.float32)
    expand_prod = sc * tp
    tl.store(out_expand_ptr + b * stride_expand_b + offs, expand_prod)

    vals = expand_prod
    ids = offs.to(tl.int64)
    remaining = offs == offs
    neg_inf = -3.4e38
    for r in tl.static_range(TOPK):
        masked = tl.where(remaining, vals, neg_inf)
        mx = tl.max(masked)
        is_elig = remaining & (masked == mx)
        pick = tl.min(tl.where(is_elig, offs, TOPK_SQ))
        remaining = remaining & (offs != pick)
        vk = tl.sum(tl.where(offs == pick, vals, 0.0))
        ik = tl.sum(tl.where(offs == pick, ids, 0))
        tl.store(out_topv_ptr + b * stride_topv_b + r, vk)
        tl.store(out_topi_ptr + b * stride_topi_b + r, ik)


def dflash_expand_topk4(
    scores: torch.Tensor,
    topk_p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused expand + top-4 for DFlash tree step (topk must be 4).

    Args:
        scores: [bs, 4] draft path scores so far.
        topk_p: [bs * 4, 4] step probabilities (repeat_interleave layout on batch).

    Returns:
        expand_scores: [bs, 4, 4] same as ``scores.unsqueeze(2) * topk_p.view(bs, 4, 4)``.
        topk_cs_p: [bs, 4] largest four products (descending).
        topk_cs_index: [bs, 4] int64 flat indices in [0, 15].
    """
    assert scores.dim() == 2 and scores.shape[1] == _DFLASH_EXPAND_TOPK
    assert topk_p.shape == (scores.shape[0] * _DFLASH_EXPAND_TOPK, _DFLASH_EXPAND_TOPK)
    bs = scores.shape[0]
    device = scores.device

    out_expand = torch.empty(
        (bs, _DFLASH_EXPAND_TOPK_SQ), device=device, dtype=scores.dtype
    )
    out_topv = torch.empty((bs, _DFLASH_EXPAND_TOPK), device=device, dtype=torch.float32)
    out_topi = torch.empty((bs, _DFLASH_EXPAND_TOPK), device=device, dtype=torch.int64)

    _dflash_expand_topk4_kernel[(bs,)](
        scores,
        topk_p,
        out_expand,
        out_topv,
        out_topi,
        scores.stride(0),
        topk_p.stride(0),
        topk_p.stride(1),
        out_expand.stride(0),
        out_topv.stride(0),
        out_topi.stride(0),
        bs,
        num_warps=1,
    )

    expand_scores = out_expand.view(bs, _DFLASH_EXPAND_TOPK, _DFLASH_EXPAND_TOPK)
    return expand_scores, out_topv, out_topi


@triton.jit
def _dflash_tree_verify_steps_kernel(
    probs_ptr,
    ids_ptr,
    out_scores_ptr,
    out_tokens_ptr,
    out_parents_ptr,
    stride_p_b,
    stride_p_s,
    stride_p_k,
    stride_i_b,
    stride_i_s,
    stride_i_k,
    out_s_sb,
    out_s_r,
    out_t_sb,
    out_p_sb,
    bs,
    TOPK: tl.constexpr,
    TOPK_SQ: tl.constexpr,
    NUM_STEPS: tl.constexpr,
):
    """One program per batch: full tree-verify score/token/parent generation for power-of-2 topk.

    Vectorized generalization of the old topk4-unrolled kernel. TOPK must be a power of 2
    (so ``tl.arange(0, TOPK)`` / ``tl.arange(0, TOPK_SQ)`` are legal) and TOPK_SQ == TOPK*TOPK.
    """
    b = tl.program_id(0)
    if b >= bs:
        return

    offs_k = tl.arange(0, TOPK)
    offs_sq = tl.arange(0, TOPK_SQ)

    # Step 0: scores = probs[:, 0]; tokens = ids[:, 0]; parents = arange(-1, topk).
    sc = tl.load(
        probs_ptr + b * stride_p_b + 0 * stride_p_s + offs_k * stride_p_k
    ).to(tl.float32)
    tl.store(out_scores_ptr + b * out_s_sb + 0 * out_s_r + offs_k, sc)

    tok0 = tl.load(ids_ptr + b * stride_i_b + 0 * stride_i_s + offs_k * stride_i_k)
    tl.store(out_tokens_ptr + b * out_t_sb + offs_k, tok0)

    # Match ``torch.arange(-1, topk)`` → width topk+1: [-1, 0, 1, ..., topk-1].
    if NUM_STEPS > 1:
        pb = out_parents_ptr + b * out_p_sb
        tl.store(pb + 0, -1)
        tl.store(pb + 1 + offs_k, offs_k.to(tl.int64))

    for i in tl.static_range(1, NUM_STEPS):
        t = tl.load(
            probs_ptr + b * stride_p_b + i * stride_p_s + offs_k * stride_p_k
        ).to(tl.float32)

        # expand_prod[p, c] = parent-path-score sc[p] * step child-prob t[c].
        expand_2d = sc[:, None] * t[None, :]

        # Store the expand-scores block: rows base_row + p, columns 0..TOPK-1.
        base_row = 1 + (i - 1) * TOPK
        tl.store(
            out_scores_ptr
            + b * out_s_sb
            + (base_row + offs_k)[:, None] * out_s_r
            + offs_k[None, :],
            expand_2d,
        )

        # Top-TOPK over the flat [TOPK_SQ] products; lowest-flat-index tie-break (lossless).
        vals = tl.reshape(expand_2d, (TOPK_SQ,))
        ids_sq = offs_sq.to(tl.int64)
        remaining = offs_sq == offs_sq
        neg_inf = -3.4e38
        topv = tl.zeros((TOPK,), tl.float32)
        topi = tl.zeros((TOPK,), tl.int64)
        for r in tl.static_range(TOPK):
            masked = tl.where(remaining, vals, neg_inf)
            mx = tl.max(masked)
            is_elig = remaining & (masked == mx)
            pick = tl.min(tl.where(is_elig, offs_sq, TOPK_SQ))
            remaining = remaining & (offs_sq != pick)
            vk = tl.sum(tl.where(offs_sq == pick, vals, 0.0))
            ik = tl.sum(tl.where(offs_sq == pick, ids_sq, 0))
            topv = tl.where(offs_k == r, vk, topv)
            topi = tl.where(offs_k == r, ik, topi)

        sc = topv

        # Tokens: lane [p, c] = step_ids[c] (independent of parent p).
        k = tl.load(ids_ptr + b * stride_i_b + i * stride_i_s + offs_k * stride_i_k)
        tok_2d = tl.broadcast_to(k[None, :], (TOPK, TOPK))
        tok_base = TOPK + (i - 1) * TOPK_SQ
        tl.store(
            out_tokens_ptr
            + b * out_t_sb
            + tok_base
            + offs_k[:, None] * TOPK
            + offs_k[None, :],
            tok_2d,
        )

        if i < NUM_STEPS - 1:
            par_shift = TOPK_SQ * (i - 1) + TOPK
            par_base = (TOPK + 1) + (i - 1) * TOPK
            tl.store(
                out_parents_ptr + b * out_p_sb + par_base + offs_k,
                topi + par_shift,
            )


def dflash_tree_verify_select_topk4_fused(
    topk_probs: torch.Tensor,
    topk_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full per-step expand + top-``topk`` recurrence in one Triton kernel.

    ``topk`` is inferred from ``topk_probs.shape[2]`` and must be a power of 2 in (4, 8, 16).
    Matches ``build_tree_verify_tokens`` tensor layout for ``score_list_cat``,
    ``ss_token_list``, and ``parent_list`` (parents exclude the last step). Name kept for callers.

    Args:
        topk_probs: [bs, num_steps, topk]
        topk_ids: [bs, num_steps, topk]

    Returns:
        score_list_cat: [bs, topk * (1 + topk * (num_steps - 1))] (``score_list`` flattened)
        tokens_flat: [bs, topk + topk**2 * (num_steps - 1)]
        parents_flat: [bs, 0] if ``num_steps == 1``, else
        ``[bs, 1 + topk * (num_steps - 1)]`` (matches ``torch.arange(-1, topk)`` at step 0).
    """
    assert topk_probs.dim() == 3
    assert topk_ids.shape == topk_probs.shape
    bs, num_steps, topk = topk_probs.shape
    # ponytail: power-of-2 keeps tl.arange(0, topk**2) legal; (4, 8, 16) is the used range.
    assert topk in (4, 8, 16) and (topk & (topk - 1)) == 0, f"unsupported topk={topk}"
    topk_sq = topk * topk
    device = topk_probs.device
    dtype = topk_probs.dtype

    score_rows = 1 + topk * (num_steps - 1)
    tok_width = topk + topk_sq * (num_steps - 1)
    if num_steps <= 1:
        par_width = 0
    else:
        par_width = 1 + topk * (num_steps - 1)

    out_scores = torch.empty((bs, score_rows, topk), device=device, dtype=dtype)
    out_tokens = torch.empty((bs, tok_width), device=device, dtype=topk_ids.dtype)
    out_parents = torch.empty((bs, par_width), device=device, dtype=torch.int64)

    _dflash_tree_verify_steps_kernel[(bs,)](
        topk_probs,
        topk_ids,
        out_scores,
        out_tokens,
        out_parents,
        topk_probs.stride(0),
        topk_probs.stride(1),
        topk_probs.stride(2),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_ids.stride(2),
        out_scores.stride(0),
        out_scores.stride(1),
        out_tokens.stride(0),
        out_parents.stride(0),
        bs,
        TOPK=topk,
        TOPK_SQ=topk_sq,
        NUM_STEPS=num_steps,
        num_warps=1,
    )

    score_list_cat = out_scores.flatten(1, 2)
    return score_list_cat, out_tokens, out_parents


def is_dflash_expand_topk4_available() -> bool:
    return torch.cuda.is_available()
