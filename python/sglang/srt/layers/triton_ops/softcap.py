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
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
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
):
    row = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ncols

    # Load values
    row_ptr = full_logits_ptr + row * row_stride
    x = tl.load(row_ptr + offsets, mask=mask, other=0.0)

    # Perform operations in-place
    x = x / softcapping_value
    x = libdevice.tanh(x)
    x = x * softcapping_value

    # Store result
    tl.store(row_ptr + offsets, x, mask=mask)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    if full_logits.is_contiguous():
        nrows, ncols = 1, full_logits.numel()
        row_stride = ncols
    else:
        assert full_logits.ndim == 2, "non-contiguous softcap requires 2D tensor"
        assert (
            full_logits.stride(1) == 1
        ), "non-contiguous softcap requires contiguous columns"
        nrows, ncols = full_logits.shape
        row_stride = full_logits.stride(0)

    BLOCK_SIZE = 1024
    grid = ((ncols + BLOCK_SIZE - 1) // BLOCK_SIZE, nrows)

    softcap_inplace_logits_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        ncols=ncols,
        row_stride=row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits
