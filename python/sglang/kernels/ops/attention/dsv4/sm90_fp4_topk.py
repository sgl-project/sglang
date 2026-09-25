"""Sort selected logical positions and map slots without dense intermediates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sort_map_topk(
    selected,
    lens,
    req,
    mapping,
    pages,
    raw,
    selected_row_stride: tl.constexpr,
    selected_col_stride: tl.constexpr,
    lens_stride: tl.constexpr,
    req_stride: tl.constexpr,
    mapping_row_stride: tl.constexpr,
    pages_row_stride: tl.constexpr,
    raw_row_stride: tl.constexpr,
    K: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    logical = tl.load(
        selected + row * selected_row_stride + col * selected_col_stride,
        col < K,
        other=0x7FFFFFFFFFFFFFFF,
    )
    logical = tl.sort(logical, descending=False)
    visible = tl.load(lens + row * lens_stride)
    request = tl.load(req + row * req_stride).to(tl.int64)
    valid = (col < K) & (logical < visible)
    # Invalid candidate positions can equal the scan bound; never read their
    # mappings. The backend has already filled the unused output tail with -1.
    position = tl.where(valid, logical, 0)
    slot = tl.load(
        mapping + request * mapping_row_stride + position * RATIO,
        valid,
        other=0,
    )
    tl.store(
        pages + row * pages_row_stride + col,
        tl.where(valid, slot // RATIO, -1),
        col < K,
    )
    if HAS_RAW:
        tl.store(
            raw + row * raw_row_stride + col, tl.where(valid, logical, -1), col < K
        )


def sort_map_topk(selected, lens, req, mapping, pages, raw, ratio):
    """Publish sorted TopK values; selection and tie-breaking stay in PyTorch.

    The caller supplies int64 selected positions, with invalid candidates
    replaced by the scan bound, and pre-fills both output tensors with -1.
    Only the opt-in backend uses this small-K epilogue.
    """
    rows, k = selected.shape
    assert selected.dtype == torch.int64 and 0 < k <= 1024
    _sort_map_topk[(rows,)](
        selected,
        lens,
        req,
        mapping,
        pages,
        raw if raw is not None else pages,
        selected.stride(0),
        selected.stride(1),
        lens.stride(0),
        req.stride(0),
        mapping.stride(0),
        pages.stride(0),
        raw.stride(0) if raw is not None else 0,
        K=k,
        RATIO=ratio,
        HAS_RAW=raw is not None,
        BLOCK=triton.next_power_of_2(k),
        num_warps=4,
    )
