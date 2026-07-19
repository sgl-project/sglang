import triton
import triton.language as tl


@triton.jit
def _zero_triton_kernel(output, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(output + offsets, 0.0, mask=offsets < n_elements)


def zero_triton(output):
    """Zero a contiguous device tensor with an explicit non-empty launch grid."""
    assert output.is_contiguous()
    n_elements = output.numel()
    if n_elements == 0:
        return output

    block_size = 256
    _zero_triton_kernel[(triton.cdiv(n_elements, block_size),)](
        output, n_elements, BLOCK_SIZE=block_size
    )
    return output
