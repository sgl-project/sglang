"""Greedy token selection over softcapped logits of a vocab shard."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _softcap_row_argmax_kernel(
    logits_ptr,
    out_ptr,
    row_stride,
    valid_width,
    col_offset,
    softcapping_value,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best_val = -float("inf")
    best_idx = 0
    for start in range(0, valid_width, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < valid_width
        x = tl.load(logits_ptr + row * row_stride + cols, mask=mask, other=0.0)
        x = x / softcapping_value
        x = libdevice.tanh(x)
        x = x * softcapping_value
        x = tl.where(mask, x, -float("inf"))
        block_val, block_idx = tl.max(
            x, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        take = block_val > best_val
        best_idx = tl.where(take, start + block_idx, best_idx)
        best_val = tl.where(take, block_val, best_val)
    tl.store(out_ptr + row * 2, best_val)
    tl.store(out_ptr + row * 2 + 1, (best_idx + col_offset).to(tl.float32))


def softcap_row_argmax(logits, final_logit_softcapping, valid_width, col_offset):
    """Per row of a vocab shard: ``(max, first argmax + col_offset)`` of the
    softcapped logits over the first ``valid_width`` columns, as fp32 ``[rows, 2]``."""
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert col_offset + logits.shape[1] < 2**24
    out = torch.empty(logits.shape[0], 2, dtype=torch.float32, device=logits.device)
    _softcap_row_argmax_kernel[(logits.shape[0],)](
        logits,
        out,
        logits.stride(0),
        valid_width,
        col_offset,
        final_logit_softcapping,
        BLOCK_SIZE=4096,
        num_warps=8,
    )
    return out
