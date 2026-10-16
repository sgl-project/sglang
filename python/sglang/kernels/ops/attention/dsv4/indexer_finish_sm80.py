"""SM80 selected-index finishing, without modifying index scores or top-k ties.

The caller provides torch.topk-style nonnegative, int32-representable IDs and
0 <= lens <= lmax < INT_MAX. Sort order is ascending logical index, duplicates
are retained, and unreachable/output-padding positions are -1. This operates
on already selected IDs only; it is not a replacement top-k algorithm.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _finish_kernel(
    idx,
    slots,
    lens,
    out,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    LS: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    LMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    ids = tl.load(idx + row * IS0 + j * IS1, j < K, other=2147483647).to(
        tl.int32
    )
    ids = tl.sort(ids, descending=False)
    length = tl.load(lens + row * LS)
    reach = (j < K) & (ids < length)
    value = tl.load(
        slots + row * SS0 + tl.minimum(ids, LMAX - 1) * SS1,
        reach,
        other=0,
    ).to(tl.int32)
    tl.store(out + row * OS0 + j * OS1, tl.where(reach, value, -1), j < WIDTH)


def finish_indexer_selected_sm80(
    idx: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    out: torch.Tensor,
    *,
    warps: int = 4,
) -> torch.Tensor:
    """Sort/gather/mask/store using shapes and strides, with no live host reads.

    Output is preallocated; the full output width is written, including the
    -1 tail outside selected k. Runtime opt-in/architecture gates are owned by
    the V4.1 decode writer; additional batch shapes are diagnostic only.
    """
    assert idx.ndim == slots.ndim == out.ndim == 2 and lens.ndim == 1
    bs, k = idx.shape
    assert idx.is_cuda and slots.is_cuda and lens.is_cuda and out.is_cuda
    assert idx.dtype in (torch.int32, torch.int64)
    assert slots.dtype in (torch.int32, torch.int64)
    assert lens.dtype in (torch.int32, torch.int64)
    assert out.dtype == torch.int32
    assert idx.device == slots.device == lens.device == out.device
    assert slots.shape[0] == bs and lens.shape == (bs,)
    assert out.shape[0] == bs and 0 < k <= out.shape[1] <= 512
    assert 0 < slots.shape[1] < 2147483647 and warps in (4, 8)
    assert all(s > 0 for t in (idx, slots, lens, out) for s in t.stride())
    _finish_kernel[(bs,)](
        idx,
        slots,
        lens,
        out,
        *idx.stride(),
        *slots.stride(),
        lens.stride(0),
        *out.stride(),
        k,
        out.shape[1],
        slots.shape[1],
        triton.next_power_of_2(out.shape[1]),
        num_warps=warps,
    )
    return out
