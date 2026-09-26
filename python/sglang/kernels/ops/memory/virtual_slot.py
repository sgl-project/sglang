# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Virtual<->physical slot Triton kernels for the unified memory pool."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Fused take-physical-pages + bind for the alloc fast path. Invoked ONLY when
# `_hole_count == 0`; otherwise the slow path drains holes first (Invariant B,
# greedy hole reuse). Caller advances `watermark_physical` and checks overflow
# BEFORE launch, passing the PRE-extension watermark. Cuda-graph safe (no
# `.item()`, no tensor branching); runs on the scheduler thread.


@triton.jit
def alloc_bind_inplace_kernel(
    v_pages_ptr,  # in: [N] int64 — virtual page ids
    v2p_ptr,  # in/out: int64 — virtual_to_physical table
    p2v_ptr,  # in/out: int64 — physical_to_virtual table
    out_phys_ptr,  # out: [N] int64 — physical page ids
    N,  # runtime: number of pages to allocate
    start_phys,  # runtime: lowest physical page id in the new range
    BLOCK: tl.constexpr,
):
    """Fused: ascending arange + out_phys/v2p/p2v scatter.

    Caller pre-adjusts `start_phys` per direction so the range is always
    ascending (grow-up: start_wm; grow-down: start_wm - N + 1), making the
    v->p mapping byte-identical to the `torch.arange` slow path.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    phys = (start_phys + offs).to(tl.int64)
    v = tl.load(v_pages_ptr + offs, mask=mask, other=0).to(tl.int64)

    # Masked stores skip out-of-range lanes, and `other=0` keeps us off the
    # v2p[0]/p2v[0] padding-sink slot.
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
    """Allocate N ascending physical pages from `start_phys` and bind to `v_pages`.

    Caller must advance `watermark_physical` by N and verify overflow BEFORE
    calling; this launcher does neither.
    """
    N = int(v_pages.numel())
    if N == 0:
        return torch.empty(0, dtype=torch.int64, device=v_pages.device)
    if not v_pages.is_cuda:
        # Pure-torch CPU reference for the CUDA-only kernel.
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


@triton.jit
def free_unbind_inplace_kernel(
    v_pages_ptr,  # in: [N] int64 — virtual page ids being freed
    v2p_ptr,  # in/out: int64 — virtual_to_physical table
    p2v_ptr,  # in/out: int64 — physical_to_virtual table
    out_phys_ptr,  # out: [N] int64 — physical page ids released
    N,  # runtime: number of pages to free
    BLOCK: tl.constexpr,
):
    """Fused inverse of `alloc_bind_inplace_kernel`: v2p read + both tombstones.

    Each lane owns one virtual page, so the read-then-tombstone of `v2p[v]` has
    no cross-lane dependency. That holds only because the caller's ids are
    unique (`_free_lazy` dedups at ps>1 and takes uniqueness from its contract
    at ps==1); duplicates would race on `p2v[p]`.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    v = tl.load(v_pages_ptr + offs, mask=mask, other=0).to(tl.int64)
    p = tl.load(v2p_ptr + v, mask=mask, other=0).to(tl.int64)

    tl.store(out_phys_ptr + offs, p, mask=mask)
    tl.store(v2p_ptr + v, -1, mask=mask)
    tl.store(p2v_ptr + p, -1, mask=mask)


@triton.jit
def bind_inplace_kernel(
    v_pages_ptr,  # in: [N] int64 — virtual page ids
    p_pages_ptr,  # in: [N] int64 — physical page ids to bind them to
    v2p_ptr,  # in/out: int64
    p2v_ptr,  # in/out: int64
    N,  # runtime: number of pages
    BLOCK: tl.constexpr,
):
    """`alloc_bind_inplace_kernel` for a caller-supplied physical range.

    The fast path generates an ascending range in-kernel; the hole-draining
    slow path already holds the physical ids, so it passes them instead.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    v = tl.load(v_pages_ptr + offs, mask=mask, other=0).to(tl.int64)
    p = tl.load(p_pages_ptr + offs, mask=mask, other=0).to(tl.int64)

    tl.store(v2p_ptr + v, p, mask=mask)
    tl.store(p2v_ptr + p, v, mask=mask)


def free_unbind_inplace(
    v_pages: torch.Tensor,
    v2p: torch.Tensor,
    p2v: torch.Tensor,
) -> torch.Tensor:
    """Tombstone `v_pages` in both tables and return the physical pages freed."""
    N = int(v_pages.numel())
    if N == 0:
        return torch.empty(0, dtype=torch.int64, device=v_pages.device)
    v = v_pages.to(torch.int64)
    if not v_pages.is_cuda:
        # Pure-torch CPU reference for the CUDA-only kernel.
        phys_pages = v2p[v].clone()
        v2p.index_fill_(0, v, -1)
        p2v.index_fill_(0, phys_pages, -1)
        return phys_pages
    phys_pages = torch.empty(N, dtype=torch.int64, device=v_pages.device)
    grid = (triton.cdiv(N, ALLOC_BIND_BLOCK),)
    free_unbind_inplace_kernel[grid](v, v2p, p2v, phys_pages, N, BLOCK=ALLOC_BIND_BLOCK)
    return phys_pages


def bind_inplace(
    v_pages: torch.Tensor,
    p_pages: torch.Tensor,
    v2p: torch.Tensor,
    p2v: torch.Tensor,
) -> None:
    """Bind `v_pages` to `p_pages` in both tables."""
    N = int(v_pages.numel())
    if N == 0:
        return
    v = v_pages.to(torch.int64)
    p = p_pages.to(torch.int64)
    if not v_pages.is_cuda:
        v2p[v] = p
        p2v[p] = v
        return
    grid = (triton.cdiv(N, ALLOC_BIND_BLOCK),)
    bind_inplace_kernel[grid](v, p, v2p, p2v, N, BLOCK=ALLOC_BIND_BLOCK)


WRITE_LOC_BLOCK = 512


@triton.jit
def write_loc_to_kernel_id_kernel(
    loc_ptr,  # in:  [N] int64 — WIDENED virtual token ids
    v2p_ptr,  # in:  [num_pages + 1] int64 — virtual->physical page table
    out_ptr,  # out: [N] int64 — kernel-facing ids
    N,  # runtime: live element count
    W,  # runtime: lanes to write; [N, W) get 0
    stride,  # runtime: pool_page_size (a physical id IS the kernel id)
    num_v_pages,  # runtime: rows of v2p; a page at or past it is unmapped
    cols,  # runtime: row width of the (rows, cols) view; STRIDED only
    loc_row_stride,  # runtime: element strides of `loc`; STRIDED only
    loc_col_stride,
    out_row_stride,  # runtime: element strides of `out`; STRIDED only
    out_col_stride,
    PAGE_SIZE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    STRIDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``kernel_id(t) = v2p[t // ps] * ps + t % ps``, clamped at 0.

    Under DCP the incoming id is WIDENED: ``loc % dcp_size`` names its owner
    and ``loc // dcp_size`` is the row. Ids this rank does not own resolve to
    kernel id 0, the padding sink every write kernel skips.

    Triton truncates ``//`` toward zero where torch floors it, so a negative
    loc is tested explicitly rather than left to the division; it resolves to
    0, as the torch path does.

    Writing ``W > N`` lanes fills ``[N, W)`` with 0, the padding sink, so a
    caller may hand in a capture-stable buffer wider than this batch and have
    the stale tail cleared in the same launch.

    ``STRIDED`` walks `loc` and `out` as a ``(rows, cols)`` view, so a column
    slice of a wider page table addresses correctly; a contiguous pair takes
    the flat path and the stride arguments go unread. Note `stride` above is
    the id-space multiplier, not a memory stride.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < W
    mask = offs < N
    if STRIDED:
        wide = offs.to(tl.int64)
        row = wide // cols
        col = wide - row * cols
        loc_off = row * loc_row_stride + col * loc_col_stride
        out_off = row * out_row_stride + col * out_col_stride
    else:
        loc_off = offs
        out_off = offs
    loc = tl.load(loc_ptr + loc_off, mask=mask, other=0).to(tl.int64)

    keep = mask & (loc >= 0)
    if DCP_SIZE > 1:
        keep = keep & ((loc % DCP_SIZE) == DCP_RANK)
        loc = loc // DCP_SIZE

    page = loc // PAGE_SIZE if PAGE_SIZE > 1 else loc
    offset = loc % PAGE_SIZE if PAGE_SIZE > 1 else 0
    # Torch's gather bounds-checks and `tl.load` does not, so an id past the
    # table is excluded here rather than read out of bounds.
    keep = keep & (page < num_v_pages)
    phys = tl.load(v2p_ptr + tl.where(keep, page, 0), mask=mask, other=0).to(tl.int64)
    ids = tl.maximum(phys * stride + offset, 0)
    tl.store(out_ptr + out_off, tl.where(keep, ids, 0), mask=in_range)


def _row_col_strides(t: torch.Tensor) -> Tuple[int, int]:
    """Element strides of `t` seen as ``(rows, cols)``; 0-d and 1-D are one row."""
    if t.dim() == 2:
        return t.stride(0), t.stride(1)
    return 0, (t.stride(0) if t.dim() == 1 else 1)


def write_loc_to_kernel_ids(
    *,
    loc: torch.Tensor,
    v2p: torch.Tensor,
    page_size: int,
    stride: int,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    out: Optional[torch.Tensor] = None,
    out_width: Optional[int] = None,
) -> torch.Tensor:
    """One launch for the whole write-loc conversion; see the kernel.

    ``out`` is written in place when given (a captured graph records the
    gather against a fixed ``data_ptr``), else a fresh int64 tensor is
    returned. Cuda-graph safe: no ``.item()``, no host sync, no allocation on
    the ``out=`` path.

    ``out_width`` writes that many lanes rather than ``loc.numel()``, zeroing
    the ones past the batch; pass the captured tier's width to clear a stale
    tail here. That mode addresses a packed lane range, so it takes a
    contiguous ``loc``; everywhere else a 1-D or 2-D strided view is legal.
    """
    N = int(loc.numel())
    if out is None:
        out = torch.empty(loc.shape, dtype=torch.int64, device=loc.device)
    width = N if out_width is None else int(out_width)
    assert out.dtype == torch.int64, (
        f"write_loc_to_kernel_ids: out dtype must be int64 (matches v2p), "
        f"got {out.dtype}"
    )
    if out_width is None:
        # `out` mirrors `loc` whatever its shape; a 2-D page table is legal.
        assert out.shape == loc.shape, (
            f"write_loc_to_kernel_ids: out shape {tuple(out.shape)} must match "
            f"loc shape {tuple(loc.shape)}"
        )
    else:
        assert loc.is_contiguous(), (
            f"write_loc_to_kernel_ids: out_width writes a packed lane range, "
            f"so loc must be packed; got stride {tuple(loc.stride())}"
        )
        assert out.dim() == 1 and out.is_contiguous() and out.numel() >= width, (
            f"write_loc_to_kernel_ids: out_width needs a packed 1-D out of at "
            f"least {width}, got {tuple(out.shape)}"
        )
        assert width >= N, (
            f"write_loc_to_kernel_ids: out_width {width} is under the batch's "
            f"{N} locs, which would drop live rows"
        )
    if width == 0:
        return out
    if not loc.is_cuda:
        # Pure-torch reference; taken by the CPU unit tests and by every
        # non-CUDA accelerator.
        big = loc.to(torch.int64)
        keep = big >= 0
        if dcp_size > 1:
            keep = keep & (big % dcp_size == dcp_rank)
            big = torch.div(big, dcp_size, rounding_mode="floor")
        page = torch.div(big, page_size, rounding_mode="floor")
        keep = keep & (page < v2p.numel())
        page = torch.where(keep, page, 0)
        offset = big % page_size if page_size > 1 else 0
        ids = (v2p[page] * stride + offset).clamp_(min=0)
        ids = torch.where(keep, ids, torch.zeros_like(ids))
        if out_width is None:
            out.copy_(ids)
        else:
            out[:N].copy_(ids)
            if width > N:
                out[N:width].zero_()
        return out
    flat = loc.is_contiguous() and out.is_contiguous()
    if flat:
        cols = loc_row_stride = loc_col_stride = 1
        out_row_stride = out_col_stride = 1
    else:
        assert loc.dim() <= 2 and out.dim() <= 2, (
            f"write_loc_to_kernel_ids: a strided loc/out is walked as 2-D, got "
            f"{loc.dim()}-D loc and {out.dim()}-D out; pass a contiguous view"
        )
        cols = loc.shape[-1] if loc.dim() == 2 else N
        loc_row_stride, loc_col_stride = _row_col_strides(loc)
        out_row_stride, out_col_stride = _row_col_strides(out)
    grid = (triton.cdiv(width, WRITE_LOC_BLOCK),)
    write_loc_to_kernel_id_kernel[grid](
        loc,
        v2p,
        out,
        N,
        width,
        stride,
        int(v2p.numel()),
        cols,
        loc_row_stride,
        loc_col_stride,
        out_row_stride,
        out_col_stride,
        PAGE_SIZE=page_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        STRIDED=not flat,
        BLOCK=WRITE_LOC_BLOCK,
    )
    return out
