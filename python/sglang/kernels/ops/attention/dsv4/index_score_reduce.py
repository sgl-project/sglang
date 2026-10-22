"""Fused BF16 score epilogue with the CUDA Torch 32-head accumulation order.

Context-dependent strides stay runtime scalars to reuse kernels across lengths.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _head_product(
    scores,
    weights,
    row,
    columns,
    head: tl.constexpr,
    width,
    score_row,
    score_head,
    score_col: tl.constexpr,
    weight_row: tl.constexpr,
    weight_head: tl.constexpr,
):
    value = tl.load(
        scores + row * score_row + head * score_head + columns * score_col,
        columns < width,
        0,
    ).to(tl.float32)
    weight = tl.load(weights + row * weight_row + head * weight_head).to(tl.float32)
    return (tl.maximum(value, 0.0) * weight).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _reduce_32_heads(
    scores,
    weights,
    output,
    width,
    score_row,
    score_head,
    score_col: tl.constexpr,
    weight_row: tl.constexpr,
    weight_head: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    acc0 = tl.full((block,), 0.0, tl.float32)
    acc1 = tl.full((block,), 0.0, tl.float32)
    acc2 = tl.full((block,), 0.0, tl.float32)
    acc3 = tl.full((block,), 0.0, tl.float32)
    # Torch Reduce.cuh uses vt0=4 independent accumulators for this strided sum.
    # A generic parallel head reduction changes rare BF16 halfway rounding cases.
    for group in tl.static_range(8):
        acc0 = acc0 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
        acc1 = acc1 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4 + 1,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
        acc2 = acc2 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4 + 2,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
        acc3 = acc3 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4 + 3,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
    result = ((acc0 + acc1) + acc2) + acc3
    result = result.to(tl.bfloat16).to(tl.float32)
    tl.store(output + row * width + columns, result, columns < width)


def reduce_head_scores(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    assert scores.ndim == 3 and weights.shape == scores.shape[:2]
    assert scores.dtype == weights.dtype == torch.bfloat16
    rows, heads, width = scores.shape
    if heads != 32 or width <= 1:
        return (scores.relu() * weights.unsqueeze(-1)).sum(dim=1).float()
    out = torch.empty((rows, width), device=scores.device, dtype=torch.float32)
    if rows:
        _reduce_32_heads[(rows, triton.cdiv(width, 256))](
            scores,
            weights,
            out,
            width,
            *scores.stride(),
            *weights.stride(),
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
