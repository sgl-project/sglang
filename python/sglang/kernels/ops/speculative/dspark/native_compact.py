"""Fixed-capacity ragged verify kernels. No device-to-host shape discovery."""

import triton
import triton.language as tl


def _search_steps(bs: int) -> int:
    # Bucket to multiples of 8 so every bs <= 255 shares one compiled kernel.
    return max(8, ((bs.bit_length() + 7) // 8) * 8)


@triton.jit
def _row_request(rows, indptr, bs, steps: tl.constexpr):
    lo = tl.zeros(rows.shape, dtype=tl.int32)
    hi = lo + bs
    for _ in range(steps):
        mid = (lo + hi) // 2
        end = tl.load(indptr + tl.minimum(mid + 1, bs))
        right = end <= rows
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    return lo


@triton.jit
def _pack(
    indptr,
    reqs,
    prefixes,
    req_to_token,
    anchors,
    drafts,
    ids,
    positions,
    locations,
    window_index,
    bs,
    n,
    width,
    table_stride,
    anchor_stride,
    draft_stride,
    draft_col_stride,
    search_steps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = _row_request(row, indptr, bs, search_steps)
    valid = (row < n) & (req < bs)
    start = tl.load(indptr + req, mask=valid, other=0)
    col = row - start
    slot = tl.load(reqs + req, mask=valid, other=0).to(tl.int64)
    prefix = tl.load(prefixes + req, mask=valid, other=0).to(tl.int64)
    pos = prefix + col
    anchor = tl.load(anchors + req * anchor_stride, mask=valid & (col == 0), other=0)
    draft = tl.load(
        drafts + req * draft_stride + (col - 1) * draft_col_stride,
        mask=valid & (col > 0),
        other=0,
    )
    loc = tl.load(req_to_token + slot * table_stride + pos, mask=valid, other=0)
    tl.store(ids + row, tl.where(col == 0, anchor, draft), mask=row < n)
    tl.store(positions + row, tl.where(valid, pos, 0), mask=row < n)
    tl.store(locations + row, loc, mask=row < n)
    tl.store(window_index + row, tl.where(valid, req * width + col, -1), mask=row < n)


def pack_verify_inputs(
    *,
    indptr,
    reqs,
    prefixes,
    req_to_token,
    anchors,
    drafts,
    width,
    ids,
    positions,
    locations,
    window_index,
):
    n = ids.numel()
    assert positions.numel() == locations.numel() == window_index.numel() == n
    _pack[(triton.cdiv(n, 128),)](
        indptr,
        reqs,
        prefixes,
        req_to_token,
        anchors,
        drafts,
        ids,
        positions,
        locations,
        window_index,
        reqs.numel(),
        n,
        width,
        req_to_token.stride(0),
        anchors.stride(0),
        drafts.stride(0),
        drafts.stride(1),
        _search_steps(reqs.numel()),
        BLOCK=128,
    )
