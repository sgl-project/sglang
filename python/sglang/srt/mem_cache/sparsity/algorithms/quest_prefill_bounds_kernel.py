"""Fused Triton kernel for Quest prefill bounds computation.

Replaces the 48-iteration Python loop in
``HiSparseCoordinator._update_quest_prefill_representations`` (which calls
``QuestAlgorithm.update_prefill_representations`` once per layer) with a
single kernel launch that processes ALL Quest layers and ALL pages of one
request.

For each (layer, page) pair the kernel:
  1. Gathers ``page_size`` token rows from that layer's K buffer using
     ``prefill_indices``.
  2. Reduces along the page dimension to produce per-(kv_head, head_dim)
     min and max.
  3. Writes the results to either:
       - ``page_k_bounds[layer, req, page, :, :, 0/1]`` for full pages, or
       - ``running_k_min/max[layer, req, :, :]`` for the partial last page.

This dramatically reduces the per-admit cost on the staging stream — the
bottleneck behind the 31.6 s p50 TTFT regression in quest_hisparse mode.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_prefill_bounds_kernel(
    k_data_ptrs,            # [num_total_layers] uint64 (pointer per layer)
    prefill_indices_ptr,    # [prefill_len] int64
    page_k_bounds_ptr,      # bf16 contiguous; layout [L, R, P, KV, D, 2]
    running_k_min_ptr,      # bf16 contiguous; layout [L, R, KV, D]
    running_k_max_ptr,      # bf16 contiguous; layout [L, R, KV, D]
    layer_offset_start,     # int (scalar), where Quest's layers start in k_data_ptrs
    req_pool_idx,           # int (scalar), target req row
    num_full_pages,         # int (scalar), runtime
    partial_count,          # int (scalar), runtime; 0 if no partial page
    MAX_REQS: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KVHD: tl.constexpr,        # kv_heads * head_dim, total
    BLOCK_KVHD: tl.constexpr,  # tile size along KVHD per block
):
    """One block per (layer, page, kvhd_tile)."""
    pid_layer = tl.program_id(0)
    pid_page = tl.program_id(1)
    pid_kvhd = tl.program_id(2)

    # Block-uniform branch: the partial-page slot is at index num_full_pages.
    is_partial = pid_page == num_full_pages

    # Skip blocks past the actual page count (caller already trims grid_y to
    # num_full_pages + (partial_count > 0), but keep this guard so callers
    # with conservative grids stay correct).
    if not is_partial and pid_page >= num_full_pages:
        return
    if is_partial and partial_count == 0:
        return

    # Layer's K buffer pointer.  K layout: [pool_size + page_size, KV, D] bf16
    # contiguous, so stride to next token = KVHD elements.
    k_ptr_int = tl.load(k_data_ptrs + layer_offset_start + pid_layer)
    k_buf_ptr = k_ptr_int.to(tl.pointer_type(tl.bfloat16))

    # Token range for this page.
    page_start = pid_page * PAGE_SIZE
    t_off = tl.arange(0, PAGE_SIZE)
    if is_partial:
        t_mask = t_off < partial_count
    else:
        t_mask = t_off < PAGE_SIZE  # always true; kept uniform for clarity

    token_indices = tl.load(
        prefill_indices_ptr + page_start + t_off,
        mask=t_mask,
        other=0,
    ).to(tl.int64)

    # KVHD tile.
    d_off_local = tl.arange(0, BLOCK_KVHD)
    d_off = pid_kvhd * BLOCK_KVHD + d_off_local

    # Gather K values for this page × kvhd tile: [PAGE_SIZE, BLOCK_KVHD] bf16.
    k_offsets = token_indices[:, None] * KVHD + d_off[None, :]
    k_vals = tl.load(
        k_buf_ptr + k_offsets,
        mask=t_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # For masked entries, substitute neutral elements so they don't perturb
    # min/max reductions.  (other=0.0 above can pollute on real-data pages.)
    pos_inf = float("inf")
    neg_inf = float("-inf")
    k_for_min = tl.where(t_mask[:, None], k_vals, pos_inf)
    k_for_max = tl.where(t_mask[:, None], k_vals, neg_inf)

    k_min = tl.min(k_for_min, axis=0).to(tl.bfloat16)  # [BLOCK_KVHD]
    k_max = tl.max(k_for_max, axis=0).to(tl.bfloat16)

    if is_partial:
        # running_k_min/max[layer, req, :, :] → flat layout [L, R, KVHD]
        out_base = pid_layer * MAX_REQS * KVHD + req_pool_idx * KVHD + d_off
        tl.store(running_k_min_ptr + out_base, k_min)
        tl.store(running_k_max_ptr + out_base, k_max)
    else:
        # page_k_bounds[layer, req, page, :, :, 0/1] interleaved on last axis.
        # Flat offset for the [min, max] pair at kvhd index d:
        #   layer * MAX_REQS * MAX_PAGES * KVHD * 2
        # + req   * MAX_PAGES * KVHD * 2
        # + page  * KVHD * 2
        # + d * 2  (+0 = min, +1 = max)
        out_base_min = (
            pid_layer * MAX_REQS * MAX_PAGES * KVHD * 2
            + req_pool_idx * MAX_PAGES * KVHD * 2
            + pid_page * KVHD * 2
            + d_off * 2
        )
        tl.store(page_k_bounds_ptr + out_base_min, k_min)
        tl.store(page_k_bounds_ptr + out_base_min + 1, k_max)


def quest_prefill_bounds(
    k_data_ptrs: torch.Tensor,
    prefill_indices: torch.Tensor,
    page_k_bounds: torch.Tensor,
    running_k_min: torch.Tensor,
    running_k_max: torch.Tensor,
    layer_offset_start: int,
    req_pool_idx: int,
    page_size: int,
) -> None:
    """Compute Quest prefill bounds for one request, all layers, in one kernel.

    Args:
      k_data_ptrs: ``[num_total_layers]`` uint64 — pointer to each layer's K
        buffer (as exposed by MHATokenToKVPool.k_data_ptrs).
      prefill_indices: ``[prefill_len]`` int.  Token positions in the K
        buffer for this request, in order.
      page_k_bounds: ``[num_quest_layers, max_reqs, max_pages, kv_heads,
        head_dim, 2]`` bf16 — Quest's combined min/max bounds tensor.
      running_k_min: ``[num_quest_layers, max_reqs, kv_heads, head_dim]`` bf16.
      running_k_max: same shape as running_k_min.
      layer_offset_start: int — index into ``k_data_ptrs`` of the first
        Quest layer (i.e., quest.start_layer).
      req_pool_idx: int — request slot to write into.
      page_size: int — Quest page size (bounds granularity).

    Behaviour matches the per-layer Python loop in
    ``QuestAlgorithm.update_prefill_representations`` exactly:
      * full pages [0, num_full_pages) → page_k_min/max
      * partial last page (if any) → running_k_min/max
      * other slots untouched (callers must invalidate if reusing a slot).
    """
    if prefill_indices.numel() == 0:
        return

    assert page_k_bounds.is_contiguous(), "page_k_bounds must be contiguous"
    assert running_k_min.is_contiguous(), "running_k_min must be contiguous"
    assert running_k_max.is_contiguous(), "running_k_max must be contiguous"
    assert page_k_bounds.dtype == torch.bfloat16
    assert running_k_min.dtype == torch.bfloat16
    assert running_k_max.dtype == torch.bfloat16

    num_quest_layers, max_reqs, max_pages, kv_heads, head_dim, two = (
        page_k_bounds.shape
    )
    assert two == 2
    kvhd = kv_heads * head_dim

    prefill_len = int(prefill_indices.shape[0])
    num_full_pages = prefill_len // page_size
    partial_count = prefill_len - num_full_pages * page_size

    grid_y = num_full_pages + (1 if partial_count > 0 else 0)
    if grid_y == 0:
        return

    # Pick a BLOCK_KVHD that divides KVHD and keeps the working tile small
    # enough.  Working tile per block: PAGE_SIZE * BLOCK_KVHD * 4 bytes (fp32
    # intermediate).  Aim for ~32 KB so multiple blocks coexist on an SM.
    if kvhd >= 256 and kvhd % 128 == 0:
        block_kvhd = 128
    elif kvhd >= 128 and kvhd % 64 == 0:
        block_kvhd = 64
    else:
        block_kvhd = kvhd  # fall back to whole-vector
    kvhd_tiles = kvhd // block_kvhd

    grid = (num_quest_layers, grid_y, kvhd_tiles)

    # Kernel expects int64 prefill_indices.
    if prefill_indices.dtype != torch.int64:
        prefill_indices = prefill_indices.to(torch.int64)

    _quest_prefill_bounds_kernel[grid](
        k_data_ptrs,
        prefill_indices,
        page_k_bounds,
        running_k_min,
        running_k_max,
        layer_offset_start,
        req_pool_idx,
        num_full_pages,
        partial_count,
        MAX_REQS=max_reqs,
        MAX_PAGES=max_pages,
        PAGE_SIZE=page_size,
        KVHD=kvhd,
        BLOCK_KVHD=block_kvhd,
        num_warps=4,
    )
