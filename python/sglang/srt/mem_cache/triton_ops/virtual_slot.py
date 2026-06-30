"""Shared-KV-pool virtual<->physical slot Triton kernels.

Houses the unified memory pool's slot-management kernels used by
``MultiEndedAllocator``: the GPU-bounded in-place virtual->physical index
translate, and the fused take-physical-pages + bind alloc fast path. Kept in
``triton_ops/`` (next to the upstream allocator kernels) per the upstream
convention that Triton kernels live here, not in ``utils.py``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# translate_kv_indices_inplace — fused, GPU-bounded, in-place virtual->physical
# translate of a shared-KV-pool attention index buffer. Used to CAPTURE
# the read-path translate into the decode cuda-graph: the build
# (`create_flashinfer_kv_indices_triton`) writes VIRTUAL
# ids into `cuda_graph_kv_indices` / `cuda_graph_window_kv_indices` eagerly in
# replay-prep; this kernel — recorded as a graph node at the front of the
# captured decode graph — rewrites that buffer in place to PHYSICAL ids,
# reading the live v2p table at replay.
#
# Why a dedicated kernel instead of reusing `MultiEndedAllocator.translate_kv_loc`
# over a fixed slice:
#   - No `.item()`: the valid extent is read on-device from `kv_indptr[BS]`,
#     so there is no D2H sync — the per-step `cudaStreamSynchronize` the eager
#     `.item()` slice incurred is removed, and the op is cuda-graph-capturable.
#   - No over-translation: a grid-stride loop bounded by `kv_indptr[BS]`
#     touches ONLY the valid prefix `[0, total)`; the multi-MB stale tail is
#     never loaded/stored.
#   - No transient / no cuda-graph-pool growth: in-place, register-only — vs
#     `index_select`/page-math which allocate an N-sized transient the graph
#     pool would hold per captured batch size.
#   - Single launch; tombstone-safe via `clamp_min(0)`.
# ---------------------------------------------------------------------------


@triton.jit
def translate_kv_indices_inplace_kernel(
    dst_ptr,  # out: PHYSICAL token ids
    src_ptr,  # in: VIRTUAL token ids (may ALIAS dst_ptr for the in-place case)
    v2p_ptr,  # virtual_to_physical table (int64); full OR swa sub-pool's table
    bound_ptr,  # the kv_indptr buffer; bound = bound_ptr[BS] = sum(seq_lens)
    BS,  # batch size (runtime int): index of the valid-extent entry in kv_indptr
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Translate ``dst[0:total] = v2p_resolve(src[0:total])`` where ``total =
    bound_ptr[BS]`` is read on-device (no ``.item()``). When ``src_ptr`` aliases
    ``dst_ptr`` this is the legacy in-place form; when they are distinct buffers
    it is out-of-place (read VIRTUAL src, write PHYSICAL dst) — idempotent under
    cuda-graph replay because the src is never overwritten by this kernel.

    Fixed grid (``tl.num_programs(0)``) + GPU-side grid-stride loop bounded by
    ``total`` so the launch is cuda-graph-stable while the work tracks the
    actual valid extent. Tombstoned entries (``v2p == -1``, or ``[-ps,-1]`` at
    ``page_size>1``) are clamped to physical slot 0 (the reserved padding sink).
    """
    # Everything int64: `total`/offsets index buffers that can exceed int32
    # (max_num_tokens * max_context_len), and keeping a single dtype avoids
    # Triton loop type-inference issues regardless of kv_indptr's dtype.
    total = tl.load(bound_ptr + BS).to(tl.int64)
    num_active_blocks = tl.cdiv(total, BLOCK)  # GPU-side loop bound
    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    block_arange = tl.arange(0, BLOCK).to(tl.int64)
    for blk in range(pid, num_active_blocks, nprog):
        offs = blk * BLOCK + block_arange
        mask = offs < total
        virt = tl.load(src_ptr + offs, mask=mask, other=0).to(tl.int64)
        if PAGE_SIZE == 1:
            phys = tl.load(v2p_ptr + virt, mask=mask, other=0)
        else:
            page = virt // PAGE_SIZE
            off = virt % PAGE_SIZE
            phys = tl.load(v2p_ptr + page, mask=mask, other=0) * PAGE_SIZE + off
        phys = tl.maximum(phys, 0)  # clamp_min(0): tombstone -> padding sink
        tl.store(dst_ptr + offs, phys, mask=mask)


def translate_kv_indices_inplace(
    kv_indices: torch.Tensor,
    v2p: torch.Tensor,
    kv_indptr: torch.Tensor,
    bs: int,
    page_size: int,
    *,
    src: Optional[torch.Tensor] = None,
    block: int = 512,
    num_programs: int = 1024,
) -> None:
    """Launch :func:`translate_kv_indices_inplace_kernel`.

    Writes ``kv_indices[0:kv_indptr[bs]]`` = physical(``src[0:kv_indptr[bs]]``)
    using ``v2p`` (the relevant sub-pool's ``virtual_to_physical`` table). The
    valid extent ``kv_indptr[bs]`` is read on-device, so no ``.item()`` / D2H
    sync occurs — this is what makes the op capturable into the decode
    cuda-graph.

    ``src`` (default ``None`` -> ``kv_indices``) is the VIRTUAL source:
        - ``src is None`` (in-place): ``kv_indices`` holds VIRTUAL on entry and
          PHYSICAL on return (read==write buffer). This is the path the shared
          pool uses.
        - ``src is not None`` (out-of-place): read VIRTUAL ids from the dedicated
          ``src`` buffer, write PHYSICAL ids into ``kv_indices``; ``src`` is never
          written, so the op is idempotent under replay.

    Contract:
        - ``kv_indices``: 1-D int64, the metadata buffer the attention reads
          (e.g. ``cuda_graph_kv_indices`` / ``cuda_graph_window_kv_indices``).
        - ``src``: 1-D int64 VIRTUAL buffer (same shape) or None (== kv_indices).
        - ``v2p``: 1-D int64 ``virtual_to_physical`` (page-granular for ps>1,
          sized ``num_pages + 1`` with a trailing ``-1`` sentinel).
        - ``kv_indptr``: 1-D int tensor whose element ``[bs]`` is the valid
          extent ``sum(seq_lens)``. The buffer (not a slice) is passed so the
          kernel can index ``[bs]`` on-device.
        - ``page_size``: a Python int (constexpr in the kernel).
        - ``bs``: batch size, index of the valid-extent entry in ``kv_indptr``.

    The grid is fixed (``num_programs``) so the launch is cuda-graph-stable;
    the kernel grid-strides over the active blocks (bounded GPU-side by
    ``kv_indptr[bs]``).
    """
    if src is None:
        src = kv_indices  # legacy in-place (read==write buffer)
    if not kv_indices.is_cuda:
        # Pure-torch CPU reference for the CUDA-only Triton kernel (CPU unit
        # tests / CPU-only environments). Translate the valid prefix
        # [0, kv_indptr[bs]) from VIRTUAL src to PHYSICAL (tombstones clamped
        # to the slot-0 padding sink); the stale tail is left untouched.
        if kv_indices.numel() == 0:
            return
        total = int(kv_indptr[bs])
        if total == 0:
            return
        virt = src[:total].to(torch.int64)
        if page_size == 1:
            phys = v2p[virt]
        else:
            phys = v2p[virt // page_size] * page_size + (virt % page_size)
        kv_indices[:total] = torch.clamp_min(phys, 0).to(kv_indices.dtype)
        return
    assert kv_indices.is_cuda and v2p.is_cuda and kv_indptr.is_cuda and src.is_cuda
    assert kv_indices.ndim == 1 and src.ndim == 1, (
        f"translate_kv_indices_inplace: kv_indices/src must be 1-D, got "
        f"{tuple(kv_indices.shape)}/{tuple(src.shape)}"
    )
    assert (
        v2p.dtype == torch.int64
    ), f"translate_kv_indices_inplace: v2p must be int64, got {v2p.dtype}"
    if kv_indices.numel() == 0:
        return
    grid = (num_programs,)
    translate_kv_indices_inplace_kernel[grid](
        kv_indices,  # dst (physical)
        src,  # src (virtual; aliases dst when in-place)
        v2p,
        kv_indptr,
        bs,
        PAGE_SIZE=page_size,
        BLOCK=block,
    )


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
