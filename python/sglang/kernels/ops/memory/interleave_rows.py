"""Coalesced row copies between interleaved CP rank and sequence order."""

import math

import torch
import triton
import triton.language as tl


# Row counts stay runtime arguments: they change with every batch, and specializing on
# them would compile and load another kernel for each distinct token count.
@triton.jit(do_not_specialize=["input_rows", "rank_rows"])
def _interleave_rows_kernel(
    X,
    Y,
    input_rows,
    rank_rows,
    WIDTH: tl.constexpr,
    CP_SIZE: tl.constexpr,
    TO_RANK_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 64-bit offsets: rows x width overflows int32 at long contexts.
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if TO_RANK_MAJOR:
        source = row % rank_rows * CP_SIZE + row // rank_rows
    else:
        source = row % CP_SIZE * rank_rows + row // CP_SIZE
    value = tl.load(
        X + source * WIDTH + col, (col < WIDTH) & (source < input_rows), other=0
    )
    tl.store(Y + row * WIDTH + col, value, col < WIDTH)


def interleave_rows(
    x: torch.Tensor,
    cp_size: int,
    output_rows: int,
    *,
    to_rank_major: bool = False,
) -> torch.Tensor:
    """Reorder contiguous rows; zero-pad sequence-to-rank copies when needed.

    Rank-major input/output consists of equally sized, contiguous rank blocks.
    ``output_rows`` trims rank padding when restoring sequence order. This is a
    local copy and performs no communication.
    """
    assert x.device.type != "cpu" and x.is_contiguous() and x.ndim >= 2
    assert cp_size > 0 and output_rows >= 0
    rank_total = output_rows if to_rank_major else x.shape[0]
    sequence_total = x.shape[0] if to_rank_major else output_rows
    assert rank_total % cp_size == 0 and sequence_total <= rank_total
    out = x.new_empty((output_rows, *x.shape[1:]))
    width = math.prod(x.shape[1:])
    if output_rows and width:
        block = min(triton.next_power_of_2(width), 8192)
        _interleave_rows_kernel[(output_rows, triton.cdiv(width, block))](
            x,
            out,
            x.shape[0],
            rank_total // cp_size,
            width,
            cp_size,
            to_rank_major,
            block,
            num_warps=8 if block >= 2048 else 4,
        )
    return out
