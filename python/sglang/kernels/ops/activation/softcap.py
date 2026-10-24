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
