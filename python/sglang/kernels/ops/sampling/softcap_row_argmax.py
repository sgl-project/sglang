"""Greedy token selection over softcapped logits of a vocab shard."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _softcap_row_argmax_partial_kernel(
    logits_ptr,
    part_val_ptr,
    part_idx_ptr,
    row_stride,
    valid_width,
    softcapping_value,
    NSPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One vocab chunk of one row: (softcapped max, lowest column attaining it).
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    cols = split * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < valid_width
    x = tl.load(logits_ptr + row * row_stride + cols, mask=mask, other=0.0)
    x = x / softcapping_value
    x = libdevice.tanh(x)
    x = x * softcapping_value
    x = tl.where(mask, x, -float("inf"))
    top = tl.max(x, axis=0)
    idx = tl.min(tl.where(x == top, cols, 0x7FFFFFFF), axis=0)
    tl.store(part_val_ptr + row * NSPLIT + split, top)
    tl.store(part_idx_ptr + row * NSPLIT + split, idx)


@triton.jit
def _softcap_row_argmax_final_kernel(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    col_offset,
    NSPLIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, NSPLIT)
    val = tl.load(part_val_ptr + row * NSPLIT + s)
    idx = tl.load(part_idx_ptr + row * NSPLIT + s)
    top = tl.max(val, axis=0)
    best = tl.min(tl.where(val == top, idx, 0x7FFFFFFF), axis=0)
    tl.store(out_ptr + row * 2, top)
    tl.store(out_ptr + row * 2 + 1, (best + col_offset).to(tl.float32))


@triton.jit
def _merge_row_argmax_shards_kernel(
    gathered_ptr,
    out_ptr,
    num_rows,
    world,
    WORLD_P2: tl.constexpr,
):
    # Rank-major [world, rows, 2]; ranks own increasing vocab ranges, so the
    # lowest index among the tied maxima is the lowest rank's.
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, WORLD_P2)
    mask = r < world
    base = gathered_ptr + (r.to(tl.int64) * num_rows + row) * 2
    val = tl.load(base, mask=mask, other=-float("inf"))
    idx = tl.load(base + 1, mask=mask, other=float("inf"))
    top = tl.max(val, axis=0)
    best = tl.min(tl.where(val == top, idx, float("inf")), axis=0)
    tl.store(out_ptr + row, best.to(tl.int64))


def merge_row_argmax_shards(gathered):
    """Global argmax ``[rows]`` int64 from the rank-major ``[world, rows, 2]``
    all-gather of softcap_row_argmax. Equal maxima resolve to the lowest rank,
    hence the lowest vocab index, as torch.argmax over the gathered row does."""
    world, num_rows, _ = gathered.shape
    if not gathered.is_cuda:
        best_rank = gathered[..., 0].argmax(dim=0, keepdim=True)
        return gathered[..., 1].gather(0, best_rank).squeeze(0).long()
    assert gathered.is_contiguous()
    out = torch.empty(num_rows, dtype=torch.int64, device=gathered.device)
    _merge_row_argmax_shards_kernel[(num_rows,)](
        gathered,
        out,
        num_rows,
        world,
        WORLD_P2=triton.next_power_of_2(world),
        num_warps=1,
    )
    return out


def softcap_row_argmax(logits, final_logit_softcapping, valid_width, col_offset):
    """Per row of a vocab shard: ``(max, first argmax + col_offset)`` of the
    softcapped logits over the first ``valid_width`` columns, as fp32 ``[rows, 2]``."""
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert col_offset + logits.shape[1] < 2**24
    num_rows = logits.shape[0]
    # A verify has only bs x block rows; spreading each over chunks fills the GPU.
    block = 4096
    nsplit = triton.next_power_of_2(max(1, triton.cdiv(valid_width, block)))
    part_val = torch.empty(num_rows, nsplit, dtype=torch.float32, device=logits.device)
    part_idx = torch.empty(num_rows, nsplit, dtype=torch.int32, device=logits.device)
    _softcap_row_argmax_partial_kernel[(num_rows, nsplit)](
        logits,
        part_val,
        part_idx,
        logits.stride(0),
        valid_width,
        final_logit_softcapping,
        NSPLIT=nsplit,
        BLOCK_SIZE=block,
        num_warps=8,
    )
    out = torch.empty(num_rows, 2, dtype=torch.float32, device=logits.device)
    _softcap_row_argmax_final_kernel[(num_rows,)](
        part_val, part_idx, out, col_offset, NSPLIT=nsplit, num_warps=1
    )
    return out
