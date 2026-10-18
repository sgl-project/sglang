import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

softcap_out_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 32768}, num_warps=32),
    ],
    key=["n_ele"],
)


@triton.jit
def softcap_out_kernel(
    output_ptr,
    input_ptr,
    n_ele,
    softcap_const: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele
    x = tl.load(input_ptr + offsets, mask=mask)
    fx = x.to(tl.float32)
    fxs = fx / softcap_const
    exped = tl.exp(2 * fxs)
    top = exped - 1
    bottom = exped + 1
    output = top / bottom * softcap_const
    tl.store(output_ptr + offsets, output, mask=mask)


softcap_out_kernel_autotuned = softcap_out_autotune(softcap_out_kernel)


def softcap_out(x, softcap_const, autotune=False):
    output = torch.empty_like(x, dtype=torch.float32)
    n_elements = output.numel()
    if autotune:

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        softcap_out_kernel_autotuned[grid](output, x, n_elements, softcap_const)
    else:
        softcap_out_kernel[(triton.cdiv(n_elements, 128),)](
            output, x, n_elements, softcap_const, BLOCK_SIZE=128, num_warps=8
        )
    return output


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
    )
    return full_logits if output is None else output


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    return _softcap_logits(full_logits, final_logit_softcapping)


def softcap_to_float32_logits(logits, final_logit_softcapping, output):
    """Write softcapped FP32 logits to an equally shaped, row-contiguous output."""
    return _softcap_logits(logits, final_logit_softcapping, output)


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
