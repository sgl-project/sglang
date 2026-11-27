"""Fused device trace lookup and token replacement for normal decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _trace_replay_kernel(
    table,
    slots,
    lengths,
    steps,
    sampled,
    output,
    size,
    num_slots,
    width,
    row_stride,
    block_size: tl.constexpr,
    inplace: tl.constexpr,
):
    idx = tl.program_id(0) * block_size + tl.arange(0, block_size)
    in_batch = idx < size
    slot = tl.load(slots + idx, mask=in_batch, other=0)
    length = tl.load(lengths + idx, mask=in_batch, other=0)
    step = tl.load(steps + idx, mask=in_batch, other=0)
    valid = (
        in_batch
        & (slot > 0)
        & (slot < num_slots)
        & (step >= 0)
        & (step < length)
        & (step < width)
    )
    offset = slot.to(tl.int64) * row_stride + step.to(tl.int64)
    token = tl.load(table + offset, mask=valid, other=0)
    if inplace:
        tl.store(output + idx, token.to(output.dtype.element_ty), mask=valid)
    else:
        original = tl.load(sampled + idx, mask=in_batch, other=0)
        token = tl.where(valid, token.to(original.dtype), original)
        tl.store(output + idx, token, mask=in_batch)


def trace_replay_triton(table, slots, lengths, steps, sampled, *, inplace=False):
    size = steps.numel()
    output = sampled if inplace else torch.empty_like(sampled)
    if size:
        block_size = 64
        _trace_replay_kernel[(triton.cdiv(size, block_size),)](
            table,
            slots,
            lengths,
            steps,
            sampled,
            output,
            size,
            table.shape[0],
            table.shape[1],
            table.stride(0),
            block_size,
            inplace,
        )
    return output
