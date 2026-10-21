"""Zero the rows a padded CUDA-graph replay reads past the batch.

The batch length lives on the device (``num_logical_tokens``);
the grid covers a fixed number of rows after it, and programs past the tensor return.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_rows_past_batch_kernel(
    x,
    num_logical_tokens,
    num_tokens,
    stride_row,
    stride_col,
    cols,
    BLOCK: tl.constexpr,
):
    row = tl.load(num_logical_tokens) + tl.program_id(0)
    if row >= num_tokens:
        return
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        x + row.to(tl.int64) * stride_row + offs * stride_col,
        tl.zeros([BLOCK], dtype=x.dtype.element_ty),
        mask=offs < cols,
    )


def zero_rows_past_batch(
    x: torch.Tensor, *, num_logical_tokens: torch.Tensor, rows: int
) -> None:
    """Zero ``rows`` rows of ``x`` ([num_tokens, ...]) from ``num_logical_tokens``."""
    x2 = x.view(x.shape[0], -1)
    block = 1024
    _zero_rows_past_batch_kernel[(rows, triton.cdiv(x2.shape[1], block))](
        x2,
        num_logical_tokens,
        x2.shape[0],
        x2.stride(0),
        x2.stride(1),
        x2.shape[1],
        BLOCK=block,
    )
