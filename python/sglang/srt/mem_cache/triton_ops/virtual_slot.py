"""Shared-KV-pool virtual<->physical slot Triton kernels.

Houses the unified memory pool's slot-management kernel used by
``MultiEndedAllocator``: the fused take-physical-pages + bind alloc fast path.
Kept in ``triton_ops/`` (next to the upstream allocator kernels) per the
upstream convention that Triton kernels live here, not in ``utils.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fuse `take_physical_pages` + `bind_pages` into one Triton kernel for the
# alloc fast path.
#
# Replaces the three GPU ops the current `take_physical_pages` +
# `bind_pages` pair emits (one `torch.arange` + two `index_put_` scatters
# → ~30-60 µs total dispatch+launch overhead per alloc) with a single
# fused Triton kernel (~15-25 µs). Saves ~20-40 µs per alloc on the fast
# path (no holes need draining), which is the dominant case for both
# eager mode (no holes ever accumulate) and lazy mode (>95% of allocs
# in the standard matrix).
#
# Design notes:
#   - The kernel is invoked ONLY when `_hole_count == 0` (i.e., no holes
#     need draining). Preserves Invariant B (greedy hole reuse): when
#     holes exist, control flows to the existing slow path which drains
#     them FIRST via `_pop_hole_directional`.
#   - Caller pre-checks index-space + peer byte-frontier overflow and
#     advances `watermark_physical` BEFORE launching the kernel; passes
#     the PRE-extension watermark as the kernel arg.
#   - Capturable under cuda-graph: no `.item()`, no Python branching on
#     tensor values, all strides known at launch time. Allocator runs on
#     the scheduler thread (not inside the captured decode graph), so the
#     `watermark` runtime scalar is the live CPU integer at each launch.
# ---------------------------------------------------------------------------


@triton.jit
def alloc_bind_inplace_kernel(
    v_pages_ptr,  # in: [N] int64 — virtual page ids
    v2p_ptr,  # in/out: int64 — virtual_to_physical table
    p2v_ptr,  # in/out: int64 — physical_to_virtual table
    out_phys_ptr,  # out: [N] int64 — physical page ids
    N,  # runtime: number of pages to allocate
    start_phys,  # runtime: lowest physical page id in the
    # newly-allocated range (caller computes
    # based on grow direction)
    BLOCK: tl.constexpr,
):
    """Token-parallel fused write: ascending arange + v2p scatter + p2v scatter.

    For each lane i in [0, N):
        phys[i]       = start_phys + i
        out_phys[i]   = phys[i]
        v2p[v[i]]     = phys[i]
        p2v[phys[i]]  = v[i]

    Direction-agnostic — the caller chooses `start_phys`:
        grow-up:   start_phys = start_wm
        grow-down: start_phys = start_wm - N + 1

    In both cases the kernel emits the page range in **ascending order**,
    matching the existing `_take_physical_eager` (`torch.arange`)
    semantics, so the v→p mapping is byte-identical to the slow path for
    every direction.

    Cuda-graph safe: no `.item()`, no Python branching on tensor values.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Compute physical page id for each lane. Ascending in both
    # directions because the caller has pre-adjusted `start_phys`.
    phys = (start_phys + offs).to(tl.int64)

    # Load virtual page id.
    v = tl.load(v_pages_ptr + offs, mask=mask, other=0).to(tl.int64)

    # Three writes fused into one kernel. Masked stores on out-of-range
    # lanes are skipped per Triton's `tl.store(..., mask=)` semantics, so
    # we never write to v2p[0] or p2v[0] (the padding-sink slot) due to
    # the `other=0` fallback on the v load.
    tl.store(out_phys_ptr + offs, phys, mask=mask)
    tl.store(v2p_ptr + v, phys, mask=mask)
    tl.store(p2v_ptr + phys, v, mask=mask)


ALLOC_BIND_BLOCK = 128


def alloc_bind_inplace(
    v_pages: torch.Tensor,
    v2p: torch.Tensor,
    p2v: torch.Tensor,
    start_phys: int,
) -> torch.Tensor:
    """Allocate N physical pages starting at `start_phys` (ascending) and
    bind to `v_pages` in one Triton kernel.

    Args:
        v_pages: [N] int64 — virtual page ids on `v2p`'s device.
        v2p: virtual_to_physical table, int64. Scattered into.
        p2v: physical_to_virtual table, int64. Scattered into.
        start_phys: Python int — the lowest physical page id in the
            new range. Caller computes based on grow direction:
                grow-up:   start_phys = start_wm
                grow-down: start_phys = start_wm - N + 1
            so the range `[start_phys, start_phys + N)` is always
            ascending — byte-identical to `torch.arange(start_phys,
            start_phys + N)` in both directions.

    Returns:
        [N] int64 tensor of allocated physical page ids on `v_pages`'s
        device, in ascending order (matches `_take_physical_eager`'s
        `torch.arange` semantics).

    NOTE: This launcher does NOT advance the allocator's watermark.
    The caller is responsible for advancing `watermark_physical` by N
    (grow-up: `+= N`, grow-down: `-= N`) AND for verifying index-space
    / peer byte-frontier overflow BEFORE invoking the launcher.
    """
    N = int(v_pages.numel())
    if N == 0:
        return torch.empty(0, dtype=torch.int64, device=v_pages.device)
    if not v_pages.is_cuda:
        # Pure-torch CPU reference for the CUDA-only Triton kernel (CPU unit
        # tests / CPU-only environments). Matches the kernel exactly: an
        # ascending physical range scattered into v2p[v]=phys and p2v[phys]=v.
        phys_pages = torch.arange(
            start_phys, start_phys + N, dtype=torch.int64, device=v_pages.device
        )
        v = v_pages.to(torch.int64)
        v2p[v] = phys_pages
        p2v[phys_pages] = v
        return phys_pages
    phys_pages = torch.empty(N, dtype=torch.int64, device=v_pages.device)
    grid = (triton.cdiv(N, ALLOC_BIND_BLOCK),)
    alloc_bind_inplace_kernel[grid](
        v_pages,
        v2p,
        p2v,
        phys_pages,
        N,
        start_phys,
        BLOCK=ALLOC_BIND_BLOCK,
    )
    return phys_pages
