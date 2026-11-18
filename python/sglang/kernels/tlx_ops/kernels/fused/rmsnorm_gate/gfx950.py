"""Per-head weightless RMSNorm times a sigmoid gate, on gfx950.

For 64-element heads, ``y = bf16(x * rstd * sigmoid(g))`` with ``rstd = 1 / sqrt(sum(x * x) / 64 + eps)`` per head,
all in FP32. One program owns 8 heads of one token.

The sum of squares follows one fixed order: adjacent element pairs first, then the 32 pair sums folded in halves (pair
``i`` plus pair ``i + 16``, then ``+ 8``, ``+ 4``, ``+ 2``, ``+ 1``). That is the order of a plain Triton kernel that
gives each head 32 lanes of two elements and reduces across the lanes from the farthest lane down. Two layouts keep
it, so their results are bitwise equal and a row's result does not depend on the batch:

- ``ELEMS=2`` (latency bound, up to three programs per CU): 32 lanes of two elements per head, four waves per
  program.
  Measured on MI350X, short calls cost less when their programs stay on few XCDs (work groups go to the 8 XCDs in
  turn): each XCD's caches take part in the end-of-kernel writeback. Only programs ``pid % 8 < xcds`` work, with
  ``xcds`` the fewest XCDs that give every program its own CU.
- ``ELEMS=8`` (bandwidth bound): 8 lanes of 8 elements (16-byte loads and stores) per head, one wave per program.
  A lane holds pairs ``4 * lane + j``: the folds at pair distances 16, 8 and 4 cross lanes (lane distances 4, 2, 1),
  the last two stay in the lane.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 64
HEADS_PER_PROGRAM = 8
# gfx950 (MI350X, MI355X) has 8 XCDs.
NUM_XCDS = 8
# Element offsets at or past this need 64-bit token offsets.
INT32_OFFSETS = 1 << 31


def pick_layout(programs: int, num_cus: int) -> tuple[int, int]:
    """(elements per lane, XCDs used) for a call of ``programs`` programs."""
    # Measured on MI350X (256 CUs, 16 heads): the 16-byte layout is faster from 448 tokens (3.5 programs per CU),
    # the two-element layout up to 384 tokens.
    if programs > 3 * num_cus:
        return 8, NUM_XCDS
    xcds = 1
    while xcds < NUM_XCDS and programs > xcds * (num_cus // NUM_XCDS):
        xcds *= 2
    return 2, xcds


@triton.jit
def _kernel(X, G, Y, eps, T, stride_x, stride_g, stride_y, H: tl.constexpr, ELEMS: tl.constexpr, XCDS: tl.constexpr,
            ROW64: tl.constexpr):
    pid = tl.program_id(0)
    if XCDS == 8:
        _heads(X, G, Y, eps, T, stride_x, stride_g, stride_y, pid, H, ELEMS, ROW64)
    elif pid % 8 < XCDS:
        _heads(X, G, Y, eps, T, stride_x, stride_g, stride_y, (pid // 8) * XCDS + pid % 8, H, ELEMS, ROW64)


@triton.jit
def _heads(X, G, Y, eps, T, stride_x, stride_g, stride_y, work, H: tl.constexpr, ELEMS: tl.constexpr,
           ROW64: tl.constexpr):
    HEADS: tl.constexpr = 8
    LANES: tl.constexpr = 64 // ELEMS
    PAIRS: tl.constexpr = ELEMS // 2
    tok = work // (H // HEADS)
    mask = tok < T
    tok = tok.to(tl.int64) if ROW64 else tok
    # [head, lane, element]: a lane holds ELEMS contiguous elements of one head.
    col = (((work % (H // HEADS)) * HEADS + tl.arange(0, HEADS))[:, None, None] * 64 +
           tl.arange(0, LANES)[None, :, None] * ELEMS + tl.arange(0, ELEMS)[None, None, :])
    x = tl.load(X + tok * stride_x + col, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + tok * stride_g + col, mask=mask, other=0.0).to(tl.float32)

    # Pair PAIRS * lane + j. The cross-lane sum folds from the farthest lane down (pair distances 16 .. PAIRS), the
    # last folds stay in the lane; the tests compare the bits with the two-element-lane kernel.
    pairs = tl.sum(tl.reshape(x * x, (HEADS, LANES, PAIRS, 2)), axis=3)
    folded = tl.sum(pairs, axis=1)
    if PAIRS == 4:
        folded = tl.sum(tl.reshape(folded, (HEADS, 2, 2)), axis=1)
    if PAIRS >= 2:
        sum_sq = tl.sum(folded, axis=1)
    else:
        sum_sq = tl.reshape(folded, (HEADS, ))
    rstd = 1.0 / tl.sqrt(sum_sq / 64 + eps)
    y = x * rstd[:, None, None]
    y = y * tl.sigmoid(g)
    tl.store(Y + tok * stride_y + col, y.to(tl.bfloat16), mask=mask)


def rmsnorm_gate(x, gate, eps, out):
    T, N = x.shape
    if out is None:
        out = torch.empty(T, N, device=x.device, dtype=torch.bfloat16)
    if T == 0:
        return out
    H = N // HEAD_DIM
    programs = T * (H // HEADS_PER_PROGRAM)
    num_cus = torch.cuda.get_device_properties(x.device).multi_processor_count
    elems, xcds = pick_layout(programs, num_cus)
    row64 = (T - 1) * max(x.stride(0), gate.stride(0), out.stride(0)) + N >= INT32_OFFSETS
    grid = (triton.cdiv(programs, xcds) * NUM_XCDS, )
    _kernel[grid](x, gate, out, float(eps), T, x.stride(0), gate.stride(0), out.stride(0), H=H, ELEMS=elems, XCDS=xcds,
                  ROW64=row64, num_warps=4 if elems == 2 else 1, num_stages=1)
    return out
