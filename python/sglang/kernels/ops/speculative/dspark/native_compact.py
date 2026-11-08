"""Fixed-capacity ragged verify kernels. No device-to-host shape discovery."""

import triton
import triton.language as tl


@triton.jit
def _row_request(rows, indptr, bs: tl.constexpr, steps: tl.constexpr):
    lo = tl.full(rows.shape, 0, tl.int32)
    hi = tl.full(rows.shape, bs, tl.int32)
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
    bs: tl.constexpr,
    n: tl.constexpr,
    width: tl.constexpr,
    table_stride: tl.constexpr,
    anchor_stride: tl.constexpr,
    draft_stride: tl.constexpr,
    draft_col_stride: tl.constexpr,
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
        reqs.numel().bit_length(),
        BLOCK=128,
    )


@triton.jit
def _accept(
    ids,
    prediction,
    indptr,
    lens,
    correct,
    bonus,
    trim,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(lens + req)
    start = tl.load(indptr + req)
    # Match the legacy zero-filled rows, including cap_trim's token-zero case.
    pred = tl.load(prediction + start + col, mask=col < length, other=0)
    candidate = tl.load(ids + start + col + 1, mask=col + 1 < length, other=0)
    first_bad = tl.min(
        tl.where((col < width - 1) & (pred != candidate), col, width - 1), 0
    )
    accepted = tl.minimum(first_bad, length - 1)
    final_token = tl.load(prediction + start + accepted, mask=length > 0, other=0)
    tl.store(correct + req, accepted)
    tl.store(bonus + req, final_token)
    tl.store(trim + req, first_bad - accepted)


def accept_ragged(*, ids, prediction, indptr, lens, width, correct, bonus, trim):
    _accept[(lens.numel(),)](
        ids,
        prediction,
        indptr,
        lens,
        correct,
        bonus,
        trim,
        width,
        BLOCK=triton.next_power_of_2(width),
    )


@triton.jit
def _commit_layout(
    indptr,
    lens,
    reqs,
    prefixes,
    commits,
    gate,
    req_to_token,
    full_to_swa,
    positions,
    locations,
    n: tl.constexpr,
    bs: tl.constexpr,
    table_stride: tl.constexpr,
    ring: tl.constexpr,
    search_steps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = _row_request(row, indptr, bs, search_steps)
    valid = (row < n) & (req < bs)
    start = tl.load(indptr + req, mask=valid, other=0)
    length = tl.load(lens + req, mask=valid, other=0)
    count = tl.load(commits + req, mask=valid, other=0)
    col = row - start
    committed = valid & (col < length) & (col < count) & (tl.load(gate) != 0)
    slot = tl.load(reqs + req, mask=valid, other=0).to(tl.int64)
    prefix = tl.load(prefixes + req, mask=valid, other=0).to(tl.int64)
    pos = tl.where(valid & (col < length), prefix + col, 0)
    if ring > 0:
        loc = slot * ring + pos % ring
    else:
        full = tl.load(
            req_to_token + slot * table_stride + pos, mask=committed, other=0
        ).to(tl.int64)
        loc = tl.load(full_to_swa + full, mask=committed, other=-1)
    tl.store(positions + row, pos, mask=row < n)
    tl.store(locations + row, tl.where(committed, loc, -1), mask=row < n)


def compact_commit_layout(
    *,
    indptr,
    lens,
    reqs,
    prefixes,
    commits,
    gate,
    req_to_token,
    full_to_swa,
    ring,
    positions,
    locations,
):
    n = positions.numel()
    mapping = full_to_swa if full_to_swa is not None else reqs
    _commit_layout[(triton.cdiv(n, 128),)](
        indptr,
        lens,
        reqs,
        prefixes,
        commits,
        gate,
        req_to_token,
        mapping,
        positions,
        locations,
        n,
        lens.numel(),
        req_to_token.stride(0),
        ring or 0,
        lens.numel().bit_length(),
        BLOCK=128,
    )


@triton.jit
def _small_candidates(ids, indptr, lens, out, width: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(lens + req)
    start = tl.load(indptr + req)
    value = tl.load(ids + start + col, mask=col < length, other=0)
    tl.store(out + req * width + col, value, mask=col < width)


def ragged_candidates(*, ids, indptr, lens, out):
    _small_candidates[(lens.numel(),)](
        ids, indptr, lens, out, out.shape[1], BLOCK=triton.next_power_of_2(out.shape[1])
    )
