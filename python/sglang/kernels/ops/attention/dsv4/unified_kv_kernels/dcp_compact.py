"""Host-side owner filtering + compaction of DCP decode streams.

PR #29185 masks non-owned entries *inside* the decode kernel
(``_dcp_row_owner``): the K loop still walks every entry of every row and only
the loads are predicated off. That cuts HBM traffic to 1/dcp but leaves the
kernel's iteration count -- and therefore its MFMA work and its runtime --
unchanged.

Compacting on the metadata side instead makes each row physically 1/dcp long,
so the decode kernel's work drops too. As a bonus the compaction also applies
the PHYSICAL row remap (``SWA_PAGES + (slot-SWA_PAGES)//dcp``), which means the
decode kernel can then run with ``DCP_SIZE=1, PHYSICAL=False``: after this pass
the stream is a plain local stream and the kernel needs no DCP awareness at all.

Ownership rules mirror ``_dcp_row_owner`` exactly:
  * read-only DCP  : owner = (position within the row) % dcp   -- stride-1
  * PHYSICAL       : SWA slot      -> owner = slot % dcp,  row = slot
                     compressed    -> owner = (slot-SWA)%dcp,
                                      row = SWA + (slot-SWA)//dcp

Empty local rows are legal and are handled by the decode kernel (it emits
out=0 / lse=-inf, which contributes nothing to the cross-rank merge).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

_buf_cache: Dict[tuple, torch.Tensor] = {}


def _cached(key: tuple, factory):
    t = _buf_cache.get(key)
    if t is None:
        t = factory()
        _buf_cache[key] = t
    return t


@triton.jit
def _owner_of(
    slot,
    pos,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    PHYSICAL: tl.constexpr,
    SWA_PAGES: tl.constexpr,
):
    """(is_owned, local_row) for one entry -- mirrors _dcp_row_owner."""
    if PHYSICAL:
        is_swa = slot < SWA_PAGES
        page = slot - SWA_PAGES
        owner = tl.where(is_swa, slot % DCP_SIZE, page % DCP_SIZE)
        row = tl.where(is_swa, slot, SWA_PAGES + page // DCP_SIZE)
    else:
        owner = pos % DCP_SIZE
        row = slot
    return owner == DCP_RANK, row


@triton.jit
def _count_kernel(
    indices_ptr,
    indptr_ptr,
    out_len_ptr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    PHYSICAL: tl.constexpr,
    SWA_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    start = tl.load(indptr_ptr + t)
    end = tl.load(indptr_ptr + t + 1)
    n = end - start

    if PHYSICAL:
        # Data-dependent: ownership depends on the slot value.
        total = 0
        for k in range(0, tl.cdiv(n, BLOCK)):
            pos = k * BLOCK + tl.arange(0, BLOCK)
            m = pos < n
            slot = tl.load(indices_ptr + start + pos, mask=m, other=0)
            keep, _ = _owner_of(slot, pos, DCP_SIZE, DCP_RANK, PHYSICAL, SWA_PAGES)
            total += tl.sum((keep & m).to(tl.int32), axis=0)
        tl.store(out_len_ptr + t, total)
    else:
        # Analytic: owner is the position modulo dcp, so the count is exact
        # without touching the data -- ceil((n - rank) / dcp), clamped at 0.
        c = (n - DCP_RANK + DCP_SIZE - 1) // DCP_SIZE
        tl.store(out_len_ptr + t, tl.maximum(c, 0))


@triton.jit
def _compact_kernel(
    indices_ptr,
    indptr_ptr,
    out_indices_ptr,
    out_indptr_ptr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    PHYSICAL: tl.constexpr,
    SWA_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    start = tl.load(indptr_ptr + t)
    end = tl.load(indptr_ptr + t + 1)
    n = end - start
    out_start = tl.load(out_indptr_ptr + t)

    acc = 0
    for k in range(0, tl.cdiv(n, BLOCK)):
        pos = k * BLOCK + tl.arange(0, BLOCK)
        m = pos < n
        slot = tl.load(indices_ptr + start + pos, mask=m, other=0)
        keep, row = _owner_of(slot, pos, DCP_SIZE, DCP_RANK, PHYSICAL, SWA_PAGES)
        keep = keep & m
        # exclusive scan within the tile, offset by what earlier tiles wrote
        rank_in_tile = tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(out_indices_ptr + out_start + acc + rank_in_tile, row, mask=keep)
        acc += tl.sum(keep.to(tl.int32), axis=0)


def compact_dcp_streams(
    indices: torch.Tensor,  # [total] int32, flat concatenated rows
    indptr: torch.Tensor,  # [T+1] int32
    *,
    dcp_size: int,
    dcp_rank: int,
    physical: bool = False,
    swa_pages: int = 0,
    block: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter each row down to this rank's entries and compact them.

    Returns ``(local_indices, local_indptr)`` where the indices are already
    remapped to local rows, so the caller runs the decode kernel with
    ``dcp_size=1`` (no in-kernel owner masking).

    Buffers are cached per (shape, device) and reused in place, keeping the pass
    CUDA-graph capturable. The output index buffer is sized like the input,
    which is always a safe upper bound.
    """
    T = indptr.numel() - 1
    if T <= 0:
        return indices, indptr
    device = indices.device

    out_len = _cached(
        ("len", device, T, indptr.dtype),
        lambda: torch.empty((T,), dtype=indptr.dtype, device=device),
    )
    out_indptr = _cached(
        ("indptr", device, T + 1, indptr.dtype),
        lambda: torch.empty((T + 1,), dtype=indptr.dtype, device=device),
    )
    out_indices = _cached(
        ("idx", device, indices.numel(), indices.dtype),
        lambda: torch.empty_like(indices),
    )

    grid = (T,)
    _count_kernel[grid](
        indices,
        indptr,
        out_len,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        PHYSICAL=physical,
        SWA_PAGES=swa_pages,
        BLOCK=block,
        num_warps=4,
    )
    # Same idiom as runtime._lengths_to_indptr, but written into a persistent
    # buffer instead of allocating (F.pad would allocate on every step).
    out_indptr[0] = 0
    # NB: do not pass dtype= together with out= -- some torch builds reject the
    # combination. out_len and out_indptr already share indptr's dtype.
    torch.cumsum(out_len, dim=0, out=out_indptr[1:])
    _compact_kernel[grid](
        indices,
        indptr,
        out_indices,
        out_indptr,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        PHYSICAL=physical,
        SWA_PAGES=swa_pages,
        BLOCK=block,
        num_warps=4,
    )
    return out_indices, out_indptr
