from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.consts import (
    REQ_POOL_IDX_PADDING,
    TOKEN_TO_KV_SLOT_PADDING,
)
from sglang.jit_kernel.kv_canary.plan.utils import (
    _compute_window_start,
    _require_2d,
    _require_dtype,
    _require_len,
    _require_min_len,
    _require_same_device,
    _resolve_swa_lut,
    _swa_translate_tile,
)
from sglang.jit_kernel.kv_canary.verify import _assert_contiguous

# Inner-tile width for _plan_entries_kernel. Each (req, verify-entry tile) program owns this many entries along
# the per-req verify-entry axis of the (bs, max_verify_per_req) logical grid.
_PLAN_VERIFY_INNER_BLOCK: int = 64


def launch_plan_entries_kernel(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    verify_offsets_scratch: torch.Tensor,
    out_verify_slot_indices: torch.Tensor,
    out_verify_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    swa_window_size: int,
) -> None:
    bs = int(req_pool_indices.shape[0])
    verify_capacity = int(out_verify_slot_indices.shape[0])
    lut_tensor, lut_len, has_swa_lut = _resolve_swa_lut(
        full_to_swa_index_mapping, verify_offsets_scratch.device
    )
    req_to_token_stride0 = int(req_to_token.stride(0))

    _validate_entries_kernel_inputs(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        req_to_token=req_to_token,
        lut_tensor=lut_tensor,
        verify_offsets_scratch=verify_offsets_scratch,
        out_verify_slot_indices=out_verify_slot_indices,
        out_verify_positions=out_verify_positions,
        out_verify_prev_slot_indices=out_verify_prev_slot_indices,
        bs=bs,
        req_to_token_stride0=req_to_token_stride0,
        lut_len=lut_len,
        verify_capacity=verify_capacity,
        has_swa_lut=has_swa_lut,
    )

    if bs == 0 or verify_capacity == 0:
        return

    max_verify_entry_tiles = (
        verify_capacity + _PLAN_VERIFY_INNER_BLOCK - 1
    ) // _PLAN_VERIFY_INNER_BLOCK
    grid_entries = (bs, max_verify_entry_tiles)
    _plan_entries_kernel[grid_entries](
        req_pool_indices,
        prefix_lens,
        req_to_token,
        lut_tensor,
        verify_offsets_scratch,
        out_verify_slot_indices,
        out_verify_positions,
        out_verify_prev_slot_indices,
        req_to_token_stride0,
        lut_len,
        verify_capacity,
        INNER_BLOCK=_PLAN_VERIFY_INNER_BLOCK,
        SWA_WINDOW=int(swa_window_size),
        HAS_SWA_LUT=has_swa_lut,
        REQ_POOL_IDX_PADDING=REQ_POOL_IDX_PADDING,
        TOKEN_TO_KV_SLOT_PADDING=TOKEN_TO_KV_SLOT_PADDING,
    )


def _validate_entries_kernel_inputs(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    lut_tensor: torch.Tensor,
    verify_offsets_scratch: torch.Tensor,
    out_verify_slot_indices: torch.Tensor,
    out_verify_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    bs: int,
    req_to_token_stride0: int,
    lut_len: int,
    verify_capacity: int,
    has_swa_lut: bool,
) -> None:
    _assert_contiguous(req_pool_indices, "req_pool_indices")
    _assert_contiguous(prefix_lens, "prefix_lens")
    _assert_contiguous(req_to_token, "req_to_token")
    _assert_contiguous(lut_tensor, "lut_tensor")
    _assert_contiguous(verify_offsets_scratch, "verify_offsets_scratch")
    _assert_contiguous(out_verify_slot_indices, "out_verify_slot_indices")
    _assert_contiguous(out_verify_positions, "out_verify_positions")
    _assert_contiguous(out_verify_prev_slot_indices, "out_verify_prev_slot_indices")

    _require_dtype(req_pool_indices, "req_pool_indices", torch.int64)
    _require_dtype(prefix_lens, "prefix_lens", torch.int64)
    _require_dtype(req_to_token, "req_to_token", torch.int32)
    _require_dtype(lut_tensor, "lut_tensor", torch.int64)
    _require_dtype(verify_offsets_scratch, "verify_offsets_scratch", torch.int64)
    _require_dtype(out_verify_slot_indices, "out_verify_slot_indices", torch.int64)
    _require_dtype(out_verify_positions, "out_verify_positions", torch.int64)
    _require_dtype(
        out_verify_prev_slot_indices, "out_verify_prev_slot_indices", torch.int64
    )

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

    _require_len(req_pool_indices, "req_pool_indices", bs)
    _require_len(prefix_lens, "prefix_lens", bs)
    _require_2d(req_to_token, "req_to_token")
    _require_min_len(lut_tensor, "lut_tensor", max(lut_len, 1))
    _require_min_len(verify_offsets_scratch, "verify_offsets_scratch", bs + 1)
    _require_len(out_verify_slot_indices, "out_verify_slot_indices", verify_capacity)
    _require_len(out_verify_positions, "out_verify_positions", verify_capacity)
    _require_len(
        out_verify_prev_slot_indices, "out_verify_prev_slot_indices", verify_capacity
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
            (req_pool_indices, "req_pool_indices"),
            (prefix_lens, "prefix_lens"),
            (req_to_token, "req_to_token"),
            (lut_tensor, "lut_tensor"),
            (out_verify_slot_indices, "out_verify_slot_indices"),
            (out_verify_positions, "out_verify_positions"),
            (out_verify_prev_slot_indices, "out_verify_prev_slot_indices"),
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
    REQ_POOL_IDX_PADDING: tl.constexpr,
    TOKEN_TO_KV_SLOT_PADDING: tl.constexpr,
):
    """Entries kernel: materialize per-req verify entries. Grid = (bs, verify_entry_tiles).

    Each program owns one (request, verify-entry tile) cell. Verify capacity is the upper bound on entries per
    request used to pick the grid; the actual per-request count comes from adjacent entries in
    ``verify_offsets``.
    """
    request_offset = tl.program_id(0)  # scalar
    verify_tile_index = tl.program_id(1)  # scalar

    request_pool_index = tl.load(req_pool_indices_ptr + request_offset)  # scalar
    prefix_lens = tl.load(prefix_lens_ptr + request_offset)  # scalar

    if request_pool_index == REQ_POOL_IDX_PADDING:
        return

    window_start = _compute_window_start(prefix_lens, SWA_WINDOW)  # scalar

    verify_start = tl.load(verify_offsets_ptr + request_offset)  # scalar
    verify_end = tl.load(verify_offsets_ptr + request_offset + 1)  # scalar
    request_verify_len = verify_end - verify_start  # scalar

    if request_verify_len <= 0:
        return

    entry_offsets = verify_tile_index * INNER_BLOCK + tl.arange(0, INNER_BLOCK)
    entry_mask = entry_offsets < request_verify_len  # [INNER_BLOCK] bool

    positions = window_start + entry_offsets  # [INNER_BLOCK]
    slot, previous_slot = _load_verify_entry_slots(
        req_to_token_ptr,
        full_to_swa_lut_ptr,
        request_pool_index,
        positions,
        entry_mask,
        req_to_token_stride0,
        swa_lut_len,
        INNER_BLOCK,
        HAS_SWA_LUT,
        TOKEN_TO_KV_SLOT_PADDING,
    )

    _store_verify_entries(
        out_verify_slot_indices_ptr,
        out_verify_positions_ptr,
        out_verify_prev_slot_indices_ptr,
        slot,
        positions,
        previous_slot,
        verify_start,
        entry_offsets,
        entry_mask,
        verify_capacity,
    )


@triton.jit
def _load_verify_entry_slots(
    req_to_token_ptr,
    full_to_swa_lut_ptr,
    request_pool_index,
    positions,
    entry_mask,
    req_to_token_stride0,
    swa_lut_len,
    INNER_BLOCK: tl.constexpr,
    HAS_SWA_LUT: tl.constexpr,
    TOKEN_TO_KV_SLOT_PADDING: tl.constexpr,
):
    request_pool_index_i64 = request_pool_index.to(tl.int64)  # scalar
    stride_i64 = req_to_token_stride0  # scalar
    positions_i64 = positions.to(tl.int64)  # [INNER_BLOCK]

    slot_full = tl.load(  # [INNER_BLOCK]
        req_to_token_ptr + request_pool_index_i64 * stride_i64 + positions_i64,
        mask=entry_mask,
        other=TOKEN_TO_KV_SLOT_PADDING,
    )

    previous_position_valid = (positions > 0) & entry_mask  # [INNER_BLOCK] bool
    previous_positions_i64 = (positions - 1).to(tl.int64)  # [INNER_BLOCK]
    safe_previous_positions_i64 = tl.where(
        previous_position_valid, previous_positions_i64, 0
    )  # [INNER_BLOCK]
    previous_slot_full = tl.load(  # [INNER_BLOCK]
        req_to_token_ptr
        + request_pool_index_i64 * stride_i64
        + safe_previous_positions_i64,
        mask=previous_position_valid,
        other=TOKEN_TO_KV_SLOT_PADDING,
    )

    if HAS_SWA_LUT:
        slot = _swa_translate_tile(
            slot_full, entry_mask, full_to_swa_lut_ptr, swa_lut_len
        )  # [INNER_BLOCK]
        previous_translated = _swa_translate_tile(  # [INNER_BLOCK]
            previous_slot_full,
            previous_position_valid,
            full_to_swa_lut_ptr,
            swa_lut_len,
        )
    else:
        slot = slot_full
        previous_translated = previous_slot_full

    chain_head_tile = tl.full((INNER_BLOCK,), -1, dtype=slot.dtype)  # [INNER_BLOCK]
    previous_slot = tl.where(
        previous_position_valid, previous_translated, chain_head_tile
    )  # [INNER_BLOCK]
    return slot, previous_slot


@triton.jit
def _store_verify_entries(
    out_verify_slot_indices_ptr,
    out_verify_positions_ptr,
    out_verify_prev_slot_indices_ptr,
    slot,
    positions,
    previous_slot,
    verify_start,
    entry_offsets,
    entry_mask,
    verify_capacity,
):
    out_offsets = verify_start + entry_offsets  # [INNER_BLOCK]
    capacity_mask = out_offsets < verify_capacity  # [INNER_BLOCK] bool
    write_mask = entry_mask & capacity_mask  # [INNER_BLOCK] bool

    tl.store(
        out_verify_slot_indices_ptr + out_offsets,
        slot.to(tl.int64),
        mask=write_mask,
    )
    tl.store(
        out_verify_positions_ptr + out_offsets,
        positions.to(tl.int64),
        mask=write_mask,
    )
    tl.store(
        out_verify_prev_slot_indices_ptr + out_offsets,
        previous_slot.to(tl.int64),
        mask=write_mask,
    )
