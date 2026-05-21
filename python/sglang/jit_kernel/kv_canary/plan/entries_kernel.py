from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.plan.utils import (
    _compute_window_start,
    _swa_translate_tile,
)

# Inner-tile width for _plan_entries_kernel. Each (req, j-tile) program owns this many entries along the
# j-axis of the (bs, max_verify_per_req) logical grid.
_PLAN_VERIFY_INNER_BLOCK: int = 64


def launch_plan_entries_kernel(
    *,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    lut_tensor: torch.Tensor,
    verify_offsets_scratch: torch.Tensor,
    verify_slot_indices: torch.Tensor,
    verify_positions: torch.Tensor,
    verify_prev_slot_indices: torch.Tensor,
    bs: int,
    req_to_token_stride0: int,
    lut_len: int,
    verify_capacity: int,
    swa_window_size: int,
    has_swa_lut: bool,
) -> None:
    if bs > 0 and verify_capacity > 0:
        max_j_tiles = (
            verify_capacity + _PLAN_VERIFY_INNER_BLOCK - 1
        ) // _PLAN_VERIFY_INNER_BLOCK
        grid_entries = (bs, max_j_tiles)
        _plan_entries_kernel[grid_entries](
            fb_req_pool_indices,
            fb_prefix_lens,
            req_to_token,
            lut_tensor,
            verify_offsets_scratch,
            verify_slot_indices,
            verify_positions,
            verify_prev_slot_indices,
            req_to_token_stride0,
            lut_len,
            verify_capacity,
            INNER_BLOCK=_PLAN_VERIFY_INNER_BLOCK,
            SWA_WINDOW=int(swa_window_size),
            HAS_SWA_LUT=has_swa_lut,
        )


@triton.jit
def _plan_entries_kernel(
    # Input pointers.
    req_pool_indices_ptr,
    prefix_lens_ptr,
    req_to_token_ptr,
    full_to_swa_lut_ptr,
    verify_offsets_ptr,
    # Output pointers.
    out_verify_slot_indices_ptr,
    out_verify_positions_ptr,
    out_verify_prev_slot_indices_ptr,
    # Runtime sizes.
    req_to_token_stride0,
    swa_lut_len,
    verify_capacity,
    # Compile-time constants.
    INNER_BLOCK: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SWA_LUT: tl.constexpr,
):
    """Entries kernel: materialize per-req verify entries. Grid = (bs, j_tiles).

    Each program owns one (req, j-tile) cell. Verify capacity is the upper bound on entries-per-req used to
    pick the grid; per-req actual count comes from ``verify_offsets[r+1] - verify_offsets[r]``.
    """
    r = tl.program_id(0)  # scalar
    tile_idx = tl.program_id(1)  # scalar

    rpi = tl.load(req_pool_indices_ptr + r)  # scalar
    prefix_lens = tl.load(prefix_lens_ptr + r)  # scalar

    # Skip padding rows entirely.
    if rpi == 0:
        return

    window_start = _compute_window_start(prefix_lens, SWA_WINDOW)  # scalar

    verify_start = tl.load(verify_offsets_ptr + r)  # scalar
    verify_end = tl.load(verify_offsets_ptr + r + 1)  # scalar
    my_verify_len = verify_end - verify_start  # scalar

    if my_verify_len <= 0:
        return

    j_offs = tile_idx * INNER_BLOCK + tl.arange(0, INNER_BLOCK)  # [INNER_BLOCK]
    j_mask = j_offs < my_verify_len  # [INNER_BLOCK] bool

    positions = window_start + j_offs  # [INNER_BLOCK]
    rpi_i64 = rpi.to(tl.int64)  # scalar
    stride_i64 = req_to_token_stride0  # scalar
    positions_i64 = positions.to(tl.int64)  # [INNER_BLOCK]

    slot_full = tl.load(  # [INNER_BLOCK]
        req_to_token_ptr + rpi_i64 * stride_i64 + positions_i64,
        mask=j_mask,
        other=0,
    )

    prev_pos_valid = (positions > 0) & j_mask  # [INNER_BLOCK] bool
    prev_positions_i64 = (positions - 1).to(tl.int64)  # [INNER_BLOCK]
    safe_prev_positions_i64 = tl.where(
        prev_pos_valid, prev_positions_i64, 0
    )  # [INNER_BLOCK]
    prev_slot_full = tl.load(  # [INNER_BLOCK]
        req_to_token_ptr + rpi_i64 * stride_i64 + safe_prev_positions_i64,
        mask=prev_pos_valid,
        other=0,
    )

    if HAS_SWA_LUT:
        slot = _swa_translate_tile(
            slot_full, j_mask, full_to_swa_lut_ptr, swa_lut_len
        )  # [INNER_BLOCK]
        prev_translated = _swa_translate_tile(  # [INNER_BLOCK]
            prev_slot_full,
            prev_pos_valid,
            full_to_swa_lut_ptr,
            swa_lut_len,
        )
    else:
        slot = slot_full
        prev_translated = prev_slot_full

    chain_head_tile = tl.full((INNER_BLOCK,), -1, dtype=slot.dtype)  # [INNER_BLOCK]
    prev_slot = tl.where(
        prev_pos_valid, prev_translated, chain_head_tile
    )  # [INNER_BLOCK]

    out_offs = verify_start + j_offs  # [INNER_BLOCK]
    cap_mask = out_offs < verify_capacity  # [INNER_BLOCK] bool
    write_mask = j_mask & cap_mask  # [INNER_BLOCK] bool

    tl.store(
        out_verify_slot_indices_ptr + out_offs,
        slot.to(tl.int64),
        mask=write_mask,
    )
    tl.store(
        out_verify_positions_ptr + out_offs,
        positions.to(tl.int64),
        mask=write_mask,
    )
    tl.store(
        out_verify_prev_slot_indices_ptr + out_offs,
        prev_slot.to(tl.int64),
        mask=write_mask,
    )
