"""ROCm Triton kernel populating sliding-window KV indices from a per-request ring."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _dspark_ring_window_kv_indices_kernel(
    req_pool_indices_ptr,
    starts_ptr,
    kv_lens_ptr,
    kv_indptr_ptr,
    out_indices_ptr,
    ROWS_PER_REQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    kv_len = tl.load(kv_lens_ptr + req_idx)
    offs = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < kv_len
    slot = tl.load(req_pool_indices_ptr + req_idx).to(tl.int64)
    start = tl.load(starts_ptr + req_idx).to(tl.int64)
    out_base = tl.load(kv_indptr_ptr + req_idx).to(tl.int64)
    pos = start + offs.to(tl.int64)
    ring_loc = 1 + slot * ROWS_PER_REQ + (pos % ROWS_PER_REQ)
    tl.store(out_indices_ptr + out_base + offs, ring_loc, mask=mask)


def fill_dspark_ring_window_buffer_hip(
    *,
    min_pos: torch.Tensor,
    rows_per_req: int,
    dcp_size: int,
    window_kv_indptr: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    bs: int,
    sliding_window_size: int,
    window_kv_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill sliding-window rail indices from a per-request draft ring.

    Coverage: ROCm (Triton), ``dcp_size == 1``, positive window and ring rows.
    Request ``b`` reads keys ``[start_b, seq_lens[b])`` with ``start_b =
    max(min_pos[slot_b], seq_lens[b] - sliding_window_size, 0)``, from ring row
    ``1 + slot_b * rows_per_req + pos % rows_per_req``. Returns ``(kv_indptr``
    int32 ``[bs + 1]``, ``kv_indices`` int64, ``kv_lens``, ``starts)``. A given
    ``window_kv_indices`` (the CUDA graph buffer) is filled in place without a
    host sync; otherwise it is sized with one ``.item()``."""
    if dcp_size != 1:
        raise ValueError(
            f"fill_dspark_ring_window_buffer_hip requires dcp_size == 1, got {dcp_size}"
        )
    if sliding_window_size <= 0 or rows_per_req <= 0:
        raise ValueError(
            "fill_dspark_ring_window_buffer_hip needs a positive window and ring "
            f"rows, got {sliding_window_size} and {rows_per_req}"
        )
    req_slots = req_pool_indices[:bs]
    lens = seq_lens[:bs].to(torch.int32)
    floors = min_pos.index_select(0, req_slots.to(torch.int64))
    starts = torch.maximum(floors, lens - int(sliding_window_size)).clamp_min(0)
    kv_lens = (lens - starts).clamp_min(0)
    window_kv_indptr[0] = 0
    window_kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
    indptr = window_kv_indptr[: bs + 1]
    if window_kv_indices is None:
        total = int(indptr[-1].item()) if bs > 0 else 0
        window_kv_indices = torch.empty(
            total, dtype=torch.int64, device=seq_lens.device
        )
    if bs > 0 and window_kv_indices.numel() > 0:
        block_size = 256
        num_tiles = triton.cdiv(int(sliding_window_size), block_size)
        _dspark_ring_window_kv_indices_kernel[(bs, num_tiles)](
            req_slots,
            starts,
            kv_lens,
            indptr,
            window_kv_indices,
            ROWS_PER_REQ=rows_per_req,
            BLOCK_SIZE=block_size,
        )
    return indptr, window_kv_indices, kv_lens, starts
