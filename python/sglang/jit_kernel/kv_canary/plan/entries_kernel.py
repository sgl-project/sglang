from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.plan.utils import (
    _compute_window_start,
    _require_2d,
    _require_dtype,
    _require_len,
    _require_min_len,
    _require_same_device,
    _swa_translate_tile,
)
from sglang.jit_kernel.kv_canary.verify import _assert_contiguous

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
    _validate_entries_kernel_inputs(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        req_to_token=req_to_token,
        lut_tensor=lut_tensor,
        verify_offsets_scratch=verify_offsets_scratch,
        verify_slot_indices=verify_slot_indices,
        verify_positions=verify_positions,
        verify_prev_slot_indices=verify_prev_slot_indices,
        bs=bs,
        req_to_token_stride0=req_to_token_stride0,
        lut_len=lut_len,
        verify_capacity=verify_capacity,
        has_swa_lut=has_swa_lut,
    )

    if bs == 0 or verify_capacity == 0:
        return

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


def _validate_entries_kernel_inputs(
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
    has_swa_lut: bool,
) -> None:
    _assert_contiguous(fb_req_pool_indices, "fb_req_pool_indices")
    _assert_contiguous(fb_prefix_lens, "fb_prefix_lens")
    _assert_contiguous(req_to_token, "req_to_token")
    _assert_contiguous(lut_tensor, "lut_tensor")
    _assert_contiguous(verify_offsets_scratch, "verify_offsets_scratch")
    _assert_contiguous(verify_slot_indices, "verify_slot_indices")
    _assert_contiguous(verify_positions, "verify_positions")
    _assert_contiguous(verify_prev_slot_indices, "verify_prev_slot_indices")

    _require_dtype(fb_req_pool_indices, "fb_req_pool_indices", torch.int64)
    _require_dtype(fb_prefix_lens, "fb_prefix_lens", torch.int64)
    _require_dtype(req_to_token, "req_to_token", torch.int32)
    _require_dtype(lut_tensor, "lut_tensor", torch.int64)
    _require_dtype(verify_offsets_scratch, "verify_offsets_scratch", torch.int64)
    _require_dtype(verify_slot_indices, "verify_slot_indices", torch.int64)
    _require_dtype(verify_positions, "verify_positions", torch.int64)
    _require_dtype(verify_prev_slot_indices, "verify_prev_slot_indices", torch.int64)

    if bs < 0:
        raise ValueError(f"kv-canary: entries kernel bs must be non-negative, got {bs}")
    if verify_capacity < 0:
        raise ValueError(
            f"kv-canary: verify_capacity must be non-negative, got {verify_capacity}"
        )
    if req_to_token_stride0 <= 0:
        raise ValueError(
            f"kv-canary: req_to_token_stride0 must be positive, got {req_to_token_stride0}"
        )
    if lut_len < 0:
        raise ValueError(f"kv-canary: lut_len must be non-negative, got {lut_len}")
    if not isinstance(has_swa_lut, bool):
        raise ValueError(
            f"kv-canary: has_swa_lut must be bool, got {type(has_swa_lut).__name__}"
        )
    if has_swa_lut and lut_len <= 0:
        raise ValueError("kv-canary: lut_len must be positive when has_swa_lut is True")
    if not has_swa_lut and lut_len != 0:
        raise ValueError("kv-canary: lut_len must be 0 when has_swa_lut is False")

    _require_len(fb_req_pool_indices, "fb_req_pool_indices", bs)
    _require_len(fb_prefix_lens, "fb_prefix_lens", bs)
    _require_2d(req_to_token, "req_to_token")
    _require_min_len(lut_tensor, "lut_tensor", max(lut_len, 1))
    _require_min_len(verify_offsets_scratch, "verify_offsets_scratch", bs + 1)
    _require_len(verify_slot_indices, "verify_slot_indices", verify_capacity)
    _require_len(verify_positions, "verify_positions", verify_capacity)
    _require_len(
        verify_prev_slot_indices, "verify_prev_slot_indices", verify_capacity
    )

    if req_to_token_stride0 != int(req_to_token.stride(0)):
        raise ValueError(
            f"kv-canary: req_to_token_stride0={req_to_token_stride0} does not match "
            f"req_to_token.stride(0)={int(req_to_token.stride(0))}"
        )

    _require_same_device(
        verify_offsets_scratch,
        "verify_offsets_scratch",
        (
            (fb_req_pool_indices, "fb_req_pool_indices"),
            (fb_prefix_lens, "fb_prefix_lens"),
            (req_to_token, "req_to_token"),
            (lut_tensor, "lut_tensor"),
            (verify_slot_indices, "verify_slot_indices"),
            (verify_positions, "verify_positions"),
            (verify_prev_slot_indices, "verify_prev_slot_indices"),
        ),
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
