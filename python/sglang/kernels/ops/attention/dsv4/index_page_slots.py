import torch
import triton
import triton.language as tl


@triton.jit
def _index_page_slots_kernel(
    raw_ptr,
    starts_ptr,
    slots_ptr,
    out_ptr,
    width,
    raw_stride,
    out_stride,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts_ptr + row).to(tl.int64)
    for off in range(0, width, BLOCK_W):
        cols = off + tl.arange(0, BLOCK_W)
        mask = cols < width
        raw = tl.load(raw_ptr + row * raw_stride + cols, mask=mask, other=-1)
        chosen = raw >= 0
        slot = tl.load(
            slots_ptr + tl.where(chosen, raw.to(tl.int64) + start, 0),
            mask=mask & chosen,
            other=-1,
        )
        tl.store(
            out_ptr + row * out_stride + cols,
            tl.where(chosen, slot, -1).to(tl.int32),
            mask=mask,
        )


def index_page_slots(
    raw: torch.Tensor,
    request_starts: torch.Tensor,
    k_slots: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out[r, j] = k_slots[raw[r, j] + request_starts[r]]`` where ``raw >= 0``,
    ``-1`` elsewhere, over the full width of ``raw``."""
    rows, width = raw.shape
    if rows == 0:
        return
    assert out.shape[0] >= rows and out.shape[1] >= width
    block = min(1024, triton.next_power_of_2(width))
    _index_page_slots_kernel[(rows,)](
        raw,
        request_starts,
        k_slots,
        out,
        width,
        raw.stride(0),
        out.stride(0),
        BLOCK_W=block,
        num_warps=4,
    )
