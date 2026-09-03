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

from typing import Optional

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


# ---------------------------------------------------------------------------
# Fused virtual WRITE loc -> kernel-facing id.
#
# The eager form of this is a chain of ~6 torch ops (floor_divide, remainder,
# take, mul, add, clamp), and ~12 once the DCP owner rule is resolved on top
# (remainder, eq, floor_divide, zeros_like, where, copy). It runs per forward
# on the write loc, OUTSIDE any cuda graph, so every op is a real launch on the
# critical path. That is invisible next to a Hopper decode step and is not
# next to a Blackwell one, which is why this is one kernel.
# ---------------------------------------------------------------------------

WRITE_LOC_BLOCK = 512


@triton.jit
def write_loc_to_kernel_id_kernel(
    loc_ptr,  # in:  [N] int64 — WIDENED virtual token ids
    v2p_ptr,  # in:  [num_pages + 1] int64 — virtual->physical page table
    out_ptr,  # out: [N] int64 — kernel-facing ids
    N,  # runtime: element count
    stride,  # runtime: pool_page_size * kernel_page_multiplier
    PAGE_SIZE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``kernel_id(t) = v2p[t // ps] * ps * mult + t % ps``, clamped at 0.

    Under DCP the incoming id is WIDENED: ``loc % dcp_size`` names its owner
    and ``loc // dcp_size`` is the row. Ids this rank does not own resolve to
    kernel id 0, the padding sink every write kernel skips.

    A negative loc resolves to 0, matching the torch path: it floor-divides to
    page -1, gathers the v2p sentinel row (-1), and clamps. Triton truncates
    toward zero instead, so the sign is tested explicitly rather than relying
    on the division.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    loc = tl.load(loc_ptr + offs, mask=mask, other=0).to(tl.int64)

    keep = loc >= 0
    if DCP_SIZE > 1:
        keep = keep & ((loc % DCP_SIZE) == DCP_RANK)
        loc = loc // DCP_SIZE

    page = loc // PAGE_SIZE if PAGE_SIZE > 1 else loc
    offset = loc % PAGE_SIZE if PAGE_SIZE > 1 else 0
    # `keep` already excludes negatives, so the gather index is in range.
    phys = tl.load(v2p_ptr + tl.where(keep, page, 0), mask=mask, other=0).to(tl.int64)
    ids = tl.maximum(phys * stride + offset, 0)
    tl.store(out_ptr + offs, tl.where(keep, ids, 0), mask=mask)


def write_loc_to_kernel_ids(
    *,
    loc: torch.Tensor,
    v2p: torch.Tensor,
    page_size: int,
    stride: int,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One launch for the whole write-loc conversion; see the kernel.

    ``out`` is written in place when given (a captured graph records the
    gather against a fixed ``data_ptr``), else a fresh int64 tensor is
    returned. Cuda-graph safe: no ``.item()``, no host sync, no allocation on
    the ``out=`` path.
    """
    N = int(loc.numel())
    if out is None:
        out = torch.empty_like(loc, dtype=torch.int64)
    assert out.dtype == torch.int64, (
        f"write_loc_to_kernel_ids: out dtype must be int64 (matches v2p), "
        f"got {out.dtype}"
    )
    assert out.shape == loc.shape, (
        f"write_loc_to_kernel_ids: out shape {tuple(out.shape)} must match "
        f"loc shape {tuple(loc.shape)}"
    )
    if N == 0:
        return out
    if not loc.is_cuda:
        # Pure-torch reference; the allocator's unit tests run on CPU.
        big = loc.to(torch.int64)
        keep = big >= 0
        if dcp_size > 1:
            keep = keep & (big % dcp_size == dcp_rank)
            big = torch.div(big, dcp_size, rounding_mode="floor")
        page = torch.where(keep, torch.div(big, page_size, rounding_mode="floor"), 0)
        offset = big % page_size if page_size > 1 else 0
        ids = (v2p[page] * stride + offset).clamp_(min=0)
        out.copy_(torch.where(keep, ids, torch.zeros_like(ids)))
        return out
    grid = (triton.cdiv(N, WRITE_LOC_BLOCK),)
    write_loc_to_kernel_id_kernel[grid](
        loc,
        v2p,
        out,
        N,
        stride,
        PAGE_SIZE=page_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        BLOCK=WRITE_LOC_BLOCK,
    )
    return out
