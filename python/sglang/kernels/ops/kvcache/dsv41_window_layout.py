"""Build query and history metadata after request-window group scans."""

import torch
import triton
import triton.language as tl


@triton.jit
def _window_layout_queries(
    Req,
    Pos,
    Floor,
    Group,
    First,
    Last,
    Starts,
    OutReq,
    OutPos,
    Write,
    Indices,
    Lengths,
    Commit,
    N,
    PAD,
    HISTORY,
    GROUPS,
    WINDOW: tl.constexpr,
    CAPACITY: tl.constexpr,
    RS: tl.constexpr,
    PS: tl.constexpr,
    FS: tl.constexpr,
    HAS_FLOOR: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    live = row < N
    req = tl.load(Req + row * RS, live, 0).to(tl.int64)
    pos = tl.load(Pos + row * PS, live, 0).to(tl.int64)
    group = tl.load(Group + row, live, 0)
    first = tl.load(First + row, live, 0)
    last = tl.load(Last + row, live, 0)
    # Starts holds num_groups slots; window_layout's invariant reports a run past them.
    tl.store(Starts + group, row, live & (row == first) & (group < GROUPS))
    first_pos = pos - (row - first)
    lookback = tl.arange(0, W)
    seen = pos[:, None] - lookback[None, :]
    valid = live[:, None] & (lookback[None, :] < WINDOW) & (seen >= 0)
    if HAS_FLOOR:
        floor = tl.load(Floor + row * FS, live, 0).to(tl.int64)
        valid &= seen >= floor[:, None]
    old_loc = group[:, None] * WINDOW + seen - (first_pos[:, None] - WINDOW)
    new_loc = HISTORY + first[:, None] + seen - first_pos[:, None]
    loc = tl.where(seen < first_pos[:, None], old_loc, new_loc)
    tl.store(
        Indices + row[:, None] * WINDOW + lookback[None, :],
        tl.where(valid, loc, -1),
        (row[:, None] < PAD) & (lookback[None, :] < WINDOW),
    )
    tl.store(OutReq + row, req, row < PAD)
    tl.store(OutPos + row, pos, row < PAD)
    tl.store(Write + row, HISTORY + row, row < PAD)
    tl.store(Lengths + row, tl.sum(valid.to(tl.int32), 1), row < PAD)
    tl.store(Commit + row, live & ((last - row) < CAPACITY), row < PAD)


@triton.jit
def _window_layout_history(
    Req,
    Pos,
    Floor,
    Last,
    Starts,
    HistReq,
    HistPos,
    HistLoc,
    HistValid,
    HistSrc,
    ROWS,
    WINDOW: tl.constexpr,
    CAPACITY: tl.constexpr,
    ZERO,
    RS: tl.constexpr,
    PS: tl.constexpr,
    FS: tl.constexpr,
    HAS_FLOOR: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    mask = row < ROWS
    start = tl.load(Starts + row // WINDOW, mask, -1)
    live = mask & (start >= 0)
    end = tl.load(Last + start, live, 0)
    req = tl.load(Req + end * RS, live, 0).to(tl.int64)
    # Every row of a contiguous request run carries the same first position and
    # floor, so its last row stands for the run.
    first_pos = (
        tl.load(Pos + end * PS, live, 0).to(tl.int64) - end + tl.maximum(start, 0)
    )
    pos = first_pos - WINDOW + row % WINDOW
    valid = live & (pos >= 0)
    if HAS_FLOOR:
        floor = tl.load(Floor + end * FS, live, 0).to(tl.int64)
        valid &= pos >= floor
    src = tl.where(valid, req * CAPACITY + pos % CAPACITY, ZERO)
    tl.store(HistReq + row, req, mask)
    tl.store(HistPos + row, pos, mask)
    tl.store(HistLoc + row, row, mask)
    tl.store(HistValid + row, valid, mask)
    tl.store(HistSrc + row, src, mask)


def build_window_layout(
    req,
    pos,
    group,
    first,
    last,
    *,
    zero_row,
    window,
    capacity,
    floor,
    num_groups,
    padded_rows,
):
    n = pos.numel()
    padded = max(n, padded_rows or n)
    groups = n if num_groups is None else int(num_groups)
    history_rows = groups * window
    device = pos.device

    def empty(shape, dtype=torch.int64):
        return torch.empty(shape, dtype=dtype, device=device)

    out_req, out_pos = empty(padded), empty(padded)
    write = empty(padded, torch.int32)
    indices = empty((padded, window), torch.int32)
    lengths = empty(padded, torch.int32)
    commit = empty(padded, torch.bool)
    starts = torch.full((groups,), -1, dtype=torch.int64, device=device)
    hist_req, hist_pos, hist_loc = (empty(history_rows) for _ in range(3))
    hist_valid = empty(history_rows, torch.bool)
    hist_src = empty(history_rows)
    floor_arg = floor if floor is not None else pos
    strides = (req.stride(0), pos.stride(0), floor_arg.stride(0))
    _window_layout_queries[(triton.cdiv(padded, 4),)](
        req,
        pos,
        floor_arg,
        group,
        first,
        last,
        starts,
        out_req,
        out_pos,
        write,
        indices,
        lengths,
        commit,
        n,
        padded,
        history_rows,
        groups,
        window,
        capacity,
        *strides,
        floor is not None,
        4,
        triton.next_power_of_2(window),
        num_warps=4,
    )
    _window_layout_history[(triton.cdiv(history_rows, 256),)](
        req,
        pos,
        floor_arg,
        last,
        starts,
        hist_req,
        hist_pos,
        hist_loc,
        hist_valid,
        hist_src,
        history_rows,
        window,
        capacity,
        zero_row,
        *strides,
        floor is not None,
        256,
        num_warps=4,
    )
    # Keyed by WindowLayout's fields.
    return dict(
        req=out_req,
        pos=out_pos,
        write_loc=write,
        indices=indices,
        lengths=lengths,
        history_req=hist_req,
        history_pos=hist_pos,
        history_loc=hist_loc,
        history_valid=hist_valid,
        history_src=hist_src,
        commit_mask=commit,
        size=history_rows + padded,
    )
