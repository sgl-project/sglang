"""Per-query sliding-window KV indices for MLA under decode context parallelism.

A sliding-window MLA layer (the SWA DFlash draft) attends, for a query at
position ``p``, the keys at positions ``[p - window_left, p]``. Under DCP the
cyclic layout puts position ``pos`` on rank ``pos % dcp_size`` at rank-local
slot ``req_to_token[pos] // dcp_size`` (the layout of
``create_mla_kv_page_table_for_dcp``). This kernel lists, per query row, the
flat paged-cache rows of the window keys this rank owns, front-packed with -1
after, plus their count: the block table and ``seq_lens`` of a trtllm-gen
sparse MLA call. ``dcp_size == 1`` gives the plain (non-DCP) window.

With a draft KV ring (``ring_rows_per_req``), a request's keys live in its own
ring of rows instead of ``req_to_token`` slots: position ``pos`` sits at local
row ``1 + req_pool_index * ring_rows_per_req + (pos // dcp_size) % ring_rows_per_req``
(row 0 is the write sink). ``min_pos``, indexed by req pool index, raises each
request's window start past the positions its ring does not hold.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def mla_window_top_k(window_left: int, dcp_size: int) -> int:
    """Index-list width: the most window keys one rank owns, padded to 4 for
    trtllm-gen sparse MLA."""
    owned = -(-(window_left + 1) // dcp_size)
    return -(-owned // 4) * 4


@triton.jit
def _mla_window_sparse_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    positions_ptr,
    v2p_ptr,
    min_pos_ptr,
    indices_ptr,
    counts_ptr,
    req_to_token_stride,
    ROWS_PER_REQ: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_V2P: tl.constexpr,
    RING_ROWS_PER_REQ: tl.constexpr,
    HAS_MIN_POS: tl.constexpr,
):
    row = tl.program_id(0)
    req_pool_index = tl.load(req_pool_indices_ptr + row // ROWS_PER_REQ).to(tl.int64)
    pos = tl.load(positions_ptr + row).to(tl.int64)
    lo = tl.maximum(pos - WINDOW_LEFT, 0)
    if HAS_MIN_POS:
        lo = tl.maximum(lo, tl.load(min_pos_ptr + req_pool_index).to(tl.int64))
    # First window position this rank owns (lo >= 0, so % is non-negative).
    first = lo + (DCP_RANK + DCP_SIZE - lo % DCP_SIZE) % DCP_SIZE
    offs = tl.arange(0, BLOCK)
    key_pos = first + offs.to(tl.int64) * DCP_SIZE
    valid = (offs < TOP_K) & (key_pos <= pos)
    if RING_ROWS_PER_REQ > 0:
        flat = (
            1
            + req_pool_index * RING_ROWS_PER_REQ
            + (key_pos // DCP_SIZE) % RING_ROWS_PER_REQ
        )
    else:
        loc = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_stride + key_pos,
            mask=valid,
            other=0,
        ).to(tl.int64)
        local_slot = loc // DCP_SIZE
        page = local_slot // PAGE_SIZE
        if HAS_V2P:
            # Same clamp as create_mla_kv_page_table_for_dcp: -1 -> padding page 0.
            page = tl.where(loc < 0, 0, page)
            page = tl.maximum(tl.load(v2p_ptr + page, mask=valid, other=0), 0)
        flat = page * PAGE_SIZE + local_slot % PAGE_SIZE
    tl.store(
        indices_ptr + row * TOP_K + offs,
        tl.where(valid, flat, -1).to(tl.int32),
        mask=offs < TOP_K,
    )
    tl.store(counts_ptr + row, tl.sum(valid.to(tl.int32), axis=0))


def mla_window_sparse_indices(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    window_left: int,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
    v2p: Optional[torch.Tensor] = None,
    ring_rows_per_req: int = 0,
    min_pos: Optional[torch.Tensor] = None,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rank-local window indices ``[rows, top_k]`` (int32, -1 padded) and
    counts ``[rows]`` for query ``positions`` (``rows`` = requests x a uniform
    number of query tokens per request, request-major). With
    ``ring_rows_per_req`` > 0 the keys come from the draft KV ring rather than
    ``req_to_token``; ``min_pos`` floors each request's window start."""
    assert not (ring_rows_per_req and v2p is not None), "a ring has no page table"
    rows = positions.shape[0]
    num_reqs = req_pool_indices.shape[0]
    assert num_reqs > 0 and rows % num_reqs == 0, (rows, num_reqs)
    top_k = mla_window_top_k(window_left, dcp_size)
    if out is None:
        indices = torch.empty((rows, top_k), dtype=torch.int32, device=positions.device)
        counts = torch.empty(rows, dtype=torch.int32, device=positions.device)
    else:
        indices, counts = out
        assert indices.shape == (rows, top_k) and counts.shape == (rows,)
    if rows == 0:
        return indices, counts
    _mla_window_sparse_indices_kernel[(rows,)](
        req_to_token,
        req_pool_indices,
        positions,
        v2p,
        min_pos,
        indices,
        counts,
        req_to_token.stride(0),
        ROWS_PER_REQ=rows // num_reqs,
        WINDOW_LEFT=window_left,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        PAGE_SIZE=page_size,
        TOP_K=top_k,
        BLOCK=triton.next_power_of_2(top_k),
        HAS_V2P=v2p is not None,
        RING_ROWS_PER_REQ=ring_rows_per_req,
        HAS_MIN_POS=min_pos is not None,
        num_warps=4,
    )
    return indices, counts
