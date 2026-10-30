import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def softcap_inplace_logits_kernel(
    full_logits_ptr,
    softcapping_value,
    ncols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    output_ptr=None,
    output_row_stride=0,
    CAST_TO_FP32: tl.constexpr = False,
    APPLY_SOFTCAP: tl.constexpr = True,
):
    row = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ncols

    # Load values
    row_ptr = full_logits_ptr + row * row_stride
    x = tl.load(row_ptr + offsets, mask=mask)
    if CAST_TO_FP32:
        x = x.to(tl.float32)

    if APPLY_SOFTCAP:
        x = x / softcapping_value
        x = libdevice.tanh(x)
        x = x * softcapping_value

    # Store result
    if CAST_TO_FP32:
        tl.store(output_ptr + row * output_row_stride + offsets, x, mask=mask)
    else:
        tl.store(row_ptr + offsets, x, mask=mask)


def _softcap_logits(full_logits, final_logit_softcapping, output=None):
    if output is not None:
        assert output.shape == full_logits.shape and output.dtype == torch.float32
    if full_logits.is_contiguous() and (output is None or output.is_contiguous()):
        nrows, ncols = 1, full_logits.numel()
        row_stride = ncols
    else:
        assert full_logits.ndim == 2, "non-contiguous softcap requires 2D tensor"
        assert full_logits.stride(1) == 1, (
            "non-contiguous softcap requires contiguous columns"
        )
        nrows, ncols = full_logits.shape
        row_stride = full_logits.stride(0)
        if output is not None:
            assert output.stride(1) == 1 and (
                nrows <= 1 or output.stride(0) >= ncols
            ), "softcap output requires contiguous, non-overlapping rows"

    BLOCK_SIZE = 1024
    grid = ((ncols + BLOCK_SIZE - 1) // BLOCK_SIZE, nrows)

    softcap_inplace_logits_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        ncols=ncols,
        row_stride=row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
        output_ptr=output,
        output_row_stride=(ncols if nrows == 1 else output.stride(0))
        if output is not None
        else 0,
        CAST_TO_FP32=output is not None,
        APPLY_SOFTCAP=final_logit_softcapping is not None,
    )
    return full_logits if output is None else output


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    return _softcap_logits(full_logits, final_logit_softcapping)


def softcap_to_float32_logits(logits, final_logit_softcapping, output):
    """Write softcapped FP32 logits to an equally shaped, row-contiguous output."""
    return _softcap_logits(logits, final_logit_softcapping, output)


def cast_to_float32_logits(logits, output):
    """Write FP32 logits to an equally shaped, row-contiguous output."""
    return _softcap_logits(logits, None, output)


@triton.jit
def _gathered_softcap_logits_kernel(
    gathered_ptr,
    output_ptr,
    num_tokens,
    shard_width,
    vocab_size,
    output_row_stride,
    softcapping_value,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size
    shard = cols // shard_width
    x = tl.load(
        gathered_ptr
        + (shard * num_tokens + row) * shard_width
        + cols
        - shard * shard_width,
        mask=mask,
    )
    x = x / softcapping_value
    x = libdevice.tanh(x)
    x = x * softcapping_value
    tl.store(output_ptr + row * output_row_stride + cols, x, mask=mask)


def gathered_softcap_logits(gathered, final_logit_softcapping, output):
    """Softcap rank-major all-gathered logits ``[world, tokens, shard]`` into
    vocab-major ``output[tokens, :vocab]`` in one pass."""
    world, num_tokens, shard_width = gathered.shape
    vocab_size = output.shape[1]
    assert gathered.is_contiguous() and output.stride(1) == 1
    assert output.shape[0] == num_tokens and vocab_size <= world * shard_width
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(vocab_size, BLOCK_SIZE), num_tokens)
    _gathered_softcap_logits_kernel[grid](
        gathered,
        output,
        num_tokens,
        shard_width,
        vocab_size,
        output.stride(0),
        final_logit_softcapping,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


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
