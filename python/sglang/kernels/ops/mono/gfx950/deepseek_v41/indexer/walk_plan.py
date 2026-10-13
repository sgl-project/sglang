# SPDX-License-Identifier: Apache-2.0
"""The FP4 score stage's FULL-layer walk plan on the device: ATOM's host
``index_plan.fill_walk_plan`` as one single-CTA Triton launch, so a replayed graph
lays it out from the step's device bounds (no host round trip; the no-sync step
has no CPU sequence lengths in time). ``[ctas, 4]`` int32 (lead, span, ch0, ch1)
a CTA, zeros past the last task, equal to the host plan (``test_dsv41_k2a_indexer``).

``owners[t]``: the first row of row t's request -- the rows of one request share
a page table, so a piece (up to WALK_ROWS adjacent rows of one owner) scores
every row off one key load."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.mono.gfx950.common.plan.execution import BLOCKS

from . import index_plan as ip

MAX_ROWS = 64  # the kernel's row tile; a mono step has at most 48 rows


@triton.jit
def _imax(a, b):
    return tl.maximum(a, b)


@triton.jit
def _walk_plan_kernel(
    bounds_ptr,
    owners_ptr,
    plan_ptr,
    n,
    CTAS: tl.constexpr,
    ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
    TOPK: tl.constexpr,
    WALK_ROWS: tl.constexpr,
):
    t = tl.arange(0, ROWS)
    row = t < n
    bound = tl.load(bounds_ptr + t, mask=row, other=0)
    owner = tl.load(owners_ptr + t, mask=row, other=-1)
    prev = tl.load(owners_ptr + t - 1, mask=row & (t > 0), other=-1)
    run_start = row & ((t == 0) | (owner != prev))
    run_first = tl.associative_scan(tl.where(run_start, t, 0), 0, _imax)
    lead = row & (run_start | ((t - run_first) % WALK_ROWS == 0))
    piece = tl.cumsum(lead.to(tl.int32), 0) - 1
    chunks = tl.where(row & (bound > TOPK), (bound + CHUNK - 1) // CHUNK, 0)

    # per piece p (pieces in lead order; a dead one, need 0, gets no task)
    p = tl.arange(0, ROWS)
    member = (piece[None, :] == p[:, None]) & row[None, :]
    need = tl.max(tl.where(member, chunks[None, :], 0), 1)
    first = tl.min(tl.where(member, t[None, :], ROWS), 1)
    span = tl.sum(member.to(tl.int32), 1)
    pieces = tl.sum((need > 0).to(tl.int32), 0)
    total = tl.sum(need, 0)

    # the smallest width whose tasks fit in CTAS (feasibility is monotone)
    lo = tl.maximum((total + CTAS - 1) // CTAS, 1)
    room = tl.maximum(CTAS - pieces, 1)
    hi = tl.maximum((total + room - 1) // room, lo)
    width = lo
    found = 0
    for w in range(lo, hi + 1):
        fits = tl.sum((need + w - 1) // w, 0) <= CTAS
        width = tl.where((found == 0) & fits, w, width)
        found = tl.where(fits, 1, found)
    k = (need + width - 1) // width
    end = tl.cumsum(k, 0)

    # CTA j's piece: the first whose tasks end past j (a dead piece ends where it starts)
    j = tl.arange(0, CTAS)
    c = tl.minimum(tl.sum((end[None, :] <= j[:, None]).to(tl.int32), 1), ROWS - 1)
    has = j < tl.sum(k, 0)
    j_k = tl.maximum(tl.gather(k, c, 0), 1)
    j_need = tl.gather(need, c, 0)
    i = j - (tl.gather(end, c, 0) - tl.gather(k, c, 0))
    out = plan_ptr + j * 4
    tl.store(out + 0, tl.where(has, tl.gather(first, c, 0), 0))
    tl.store(out + 1, tl.where(has, tl.gather(span, c, 0), 0))
    tl.store(out + 2, tl.where(has, i * j_need // j_k, 0))
    tl.store(out + 3, tl.where(has, (i + 1) * j_need // j_k, 0))


def fill_walk_plan_device(
    bounds: torch.Tensor, owners: torch.Tensor, plan: torch.Tensor, rows: int
) -> None:
    """``plan`` [ctas, 4] int32 <- the walk plan of ``rows`` rows (``bounds``, ``owners`` int32 on the device);
    graph safe: one launch, no host reads."""
    assert 0 < rows <= MAX_ROWS, rows
    assert bounds.dtype == owners.dtype == plan.dtype == torch.int32
    assert plan.is_contiguous() and plan.shape[1] == 4 and plan.shape[0] % 64 == 0
    _walk_plan_kernel[(1,)](
        bounds,
        owners,
        plan,
        rows,
        CTAS=plan.shape[0],
        ROWS=MAX_ROWS,
        CHUNK=ip.CHUNK,
        TOPK=ip.TOPK,
        WALK_ROWS=ip.WALK_ROWS,
        num_warps=8,
    )


def new_plan(device) -> torch.Tensor:
    return torch.zeros(BLOCKS, 4, dtype=torch.int32, device=device)
