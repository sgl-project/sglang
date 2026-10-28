# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Adapted from vLLM models/step5/nvidia/ops/triton/router_bias.py.

Preserves sigmoid probabilities from the selected logits, ordered normalization,
and scale. This leaf uses one fixed launch configuration, not vLLM's optional
driver-launch cache or autotuner. Public metadata validation owns buffer safety.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step5_router_topk_kernel(
    logits,
    bias,
    weights_out,
    ids_out,
    logits_stride,
    weights_stride,
    ids_stride,
    num_token_non_padded,
    HAS_PADDING: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    expert = tl.arange(0, 512)
    valid = expert < 352
    gating = tl.load(logits + row * logits_stride + expert, mask=valid, other=0)
    probability = tl.sigmoid(gating.to(tl.float32))
    correction = tl.load(bias + expert, mask=valid, other=0).to(tl.float32)
    correction = tl.where(valid, correction, -float("inf"))
    scores = tl.where(valid, probability + correction, -float("inf"))
    has_bad = tl.max((gating != gating).to(tl.int32), axis=0) > 0

    weights = tl.zeros((8,), dtype=tl.float32)
    indices = tl.zeros((8,), dtype=tl.int32)
    total = 0.0
    slot = tl.arange(0, 8)
    for k in tl.static_range(8):
        score, index = tl.max(
            scores, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        index = index.to(tl.int32)
        # NaN rows are overwritten below, but still execute the reduction. A
        # masked padding winner must never read outside the logical logits row.
        selected_logit = tl.load(
            logits + row * logits_stride + index,
            mask=(index >= 0) & (index < 352),
            other=0,
        ).to(tl.float32)
        # Bias affects selection only. Subtracting it from the rounded score
        # can erase small probabilities and change normalized routing weights.
        selected_probability = tl.sigmoid(selected_logit)
        weights = tl.where(slot == k, selected_probability, weights)
        indices = tl.where(slot == k, index, indices)
        total += selected_probability
        scores = tl.where(expert == index, -float("inf"), scores)

    weights = weights / (total + 1e-20)
    weights = weights * 3.0
    weights = tl.where(has_bad, 0.0, weights)
    indices = tl.where(has_bad, 0, indices)
    tl.store(weights_out + row * weights_stride + slot, weights)
    if HAS_PADDING:
        indices = tl.where(row < tl.load(num_token_non_padded), indices, -1)
    tl.store(ids_out + row * ids_stride + slot, indices)


def _step5_router_topk(
    *,
    logits: torch.Tensor,
    bias: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    num_token_non_padded: torch.Tensor | None = None,
) -> None:
    _step5_router_topk_kernel[(logits.shape[0],)](
        logits,
        bias,
        weights,
        ids,
        logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        num_token_non_padded,
        HAS_PADDING=num_token_non_padded is not None,
        num_warps=4,
        num_stages=2,
    )
