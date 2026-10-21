"""Reduce routed rows while skipping inactive permutation entries."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "weight_stride0", "weight_stride1"])
def _shuffle_weighted_sum_kernel(
    inputs,
    output,
    permutation,
    weights,
    width: tl.constexpr,
    rows,
    topk: tl.constexpr,
    input_stride: tl.constexpr,
    output_stride: tl.constexpr,
    weight_stride0,
    weight_stride1,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(topk):
        row = tl.load(permutation + token * topk + slot).to(tl.int64)
        active = (row >= 0) & (row < rows)
        values = tl.load(
            inputs + row * input_stride + columns,
            mask=active & (columns < width),
            other=0,
        ).to(tl.float32)
        factor = tl.load(
            weights + token * weight_stride0 + slot * weight_stride1,
            mask=active,
            other=0,
        ).to(tl.float32)
        total += values * factor
    tl.store(
        output + token.to(tl.int64) * output_stride + columns,
        total,
        mask=columns < width,
    )


def shuffle_weighted_sum(
    inputs: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Sum mapped expert outputs; inactive maps read neither data nor weights."""
    assert inputs.ndim == output.ndim == weights.ndim == 2
    assert inputs.shape[1] == output.shape[1]
    assert inputs.stride(1) == output.stride(1) == 1
    assert weights.shape[0] == output.shape[0]
    assert permutation.is_contiguous() and permutation.numel() == weights.numel()
    if output.numel() == 0:
        return
    _shuffle_weighted_sum_kernel[(output.shape[0], triton.cdiv(output.shape[1], 256))](
        inputs,
        output,
        permutation,
        weights,
        output.shape[1],
        inputs.shape[0],
        weights.shape[1],
        inputs.stride(0),
        output.stride(0),
        weights.stride(0),
        weights.stride(1),
        BLOCK=256,
    )
