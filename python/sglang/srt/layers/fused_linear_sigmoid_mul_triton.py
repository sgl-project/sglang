"""Triton fused op: sigmoid(x @ w^T) * m for w shape (1, H)"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_sigmoid_mul_triton_kernel(
    hidden_states_ptr,
    weight_ptr,
    shared_output_ptr,
    output_ptr,
    N,
    H,
    hidden_stride_row,
    shared_output_stride_row,
    output_stride_row,
    BLOCK_H: tl.constexpr,
):
    """Per row: dot(x[n,:], w[0,:]) -> sigmoid -> multiply shared_output[n,:]."""
    pid_n = tl.program_id(0)
    row_start_h = pid_n * hidden_stride_row
    row_start_m = pid_n * shared_output_stride_row
    row_start_out = pid_n * output_stride_row
    row_ok = pid_n < N

    sum_q = 0.0
    for i in range(0, H, BLOCK_H):
        col_offsets = i + tl.arange(0, BLOCK_H)
        col_mask = col_offsets < H
        mask_1d = row_ok & col_mask
        x = tl.load(
            hidden_states_ptr + row_start_h + col_offsets, mask=mask_1d, other=0.0
        )
        w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0)
        prod = x.to(tl.float32) * w.to(tl.float32)
        sum_q = sum_q + tl.sum(prod)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-sum_q))

    for i in range(0, H, BLOCK_H):
        col_offsets = i + tl.arange(0, BLOCK_H)
        col_mask = col_offsets < H
        mask_2d = row_ok & col_mask[None, :]
        y_ptrs = shared_output_ptr + row_start_m + col_offsets[None, :]
        y = tl.load(y_ptrs, mask=mask_2d, other=0.0)
        out = sigmoid_x * y.to(tl.float32)
        out_ptrs = output_ptr + row_start_out + col_offsets[None, :]
        tl.store(out_ptrs, out.to(output_ptr.dtype.element_ty), mask=mask_2d)


def fused_linear_sigmoid_mul(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    shared_output: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton: sigmoid(sum_h x[n,h]*W[0,h]) * shared_output[n,:]."""
    assert hidden_states.dim() == 2 and weight.dim() == 2 and shared_output.dim() == 2
    n, h = shared_output.shape
    assert hidden_states.shape == (n, h)
    assert weight.shape == (
        1,
        h,
    ), "shared expert gate weight must be (1, hidden_size), no bias"
    assert weight.device == hidden_states.device == shared_output.device

    if out is None:
        out = torch.empty_like(shared_output)
    else:
        assert out.shape == shared_output.shape and out.dtype == shared_output.dtype

    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert shared_output.is_contiguous(), "shared_output must be contiguous"
    assert out.is_contiguous(), "out must be contiguous"

    block_h = 2048
    grid = (triton.cdiv(n, 1),)
    fused_linear_sigmoid_mul_triton_kernel[grid](
        hidden_states,
        weight,
        shared_output,
        out,
        n,
        h,
        hidden_states.stride(0),
        shared_output.stride(0),
        out.stride(0),
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=2,
    )
    return out
