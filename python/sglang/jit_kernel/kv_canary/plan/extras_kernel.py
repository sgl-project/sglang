from __future__ import annotations

import torch
import triton
import triton.language as tl

# Inner-tile width for _plan_extras_kernel. Each program copies this many extras into the verify tail.
_PLAN_EXTRAS_INNER_BLOCK: int = 64


def launch_plan_extras_kernel(
    *,
    extra_verify_slot_indices: torch.Tensor,
    extra_verify_positions: torch.Tensor,
    extra_verify_prev_slot_indices: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    verify_offsets_scratch: torch.Tensor,
    verify_slot_indices: torch.Tensor,
    verify_positions: torch.Tensor,
    verify_prev_slot_indices: torch.Tensor,
    bs: int,
    verify_capacity: int,
    extras_capacity: int,
) -> None:
    if extras_capacity > 0:
        max_k_tiles = (
            extras_capacity + _PLAN_EXTRAS_INNER_BLOCK - 1
        ) // _PLAN_EXTRAS_INNER_BLOCK
        grid_extras = (max_k_tiles,)
        _plan_extras_kernel[grid_extras](
            extra_verify_slot_indices,
            extra_verify_positions,
            extra_verify_prev_slot_indices,
            extra_verify_num_valid,
            verify_offsets_scratch,
            verify_slot_indices,
            verify_positions,
            verify_prev_slot_indices,
            bs,
            verify_capacity,
            INNER_BLOCK=_PLAN_EXTRAS_INNER_BLOCK,
        )


@triton.jit
def _plan_extras_kernel(
    # Input pointers.
    extra_slot_ptr,
    extra_positions_ptr,
    extra_prev_slot_ptr,
    extra_num_valid_ptr,
    verify_offsets_ptr,
    # Output pointers.
    out_verify_slot_indices_ptr,
    out_verify_positions_ptr,
    out_verify_prev_slot_indices_ptr,
    # Runtime sizes.
    bs,
    verify_capacity,
    # Compile-time constants.
    INNER_BLOCK: tl.constexpr,
):
    """Extras kernel: append extras into the verify tail at base = verify_offsets[bs]. Grid = (k_tiles,).

    Extras are caller-pre-translated; this kernel only copies (no LUT pass).
    """
    tile_idx = tl.program_id(0)  # scalar
    k_offs = tile_idx * INNER_BLOCK + tl.arange(0, INNER_BLOCK)  # [INNER_BLOCK]

    extras_count = tl.load(extra_num_valid_ptr)  # scalar
    extras_count = tl.where(extras_count > 0, extras_count, 0)
    in_range_mask = k_offs < extras_count  # [INNER_BLOCK] bool

    base_idx = tl.load(verify_offsets_ptr + bs)  # scalar

    slots = tl.load(
        extra_slot_ptr + k_offs, mask=in_range_mask, other=0
    )  # [INNER_BLOCK]
    positions = tl.load(
        extra_positions_ptr + k_offs, mask=in_range_mask, other=0
    )  # [INNER_BLOCK]
    prevs = tl.load(
        extra_prev_slot_ptr + k_offs, mask=in_range_mask, other=0
    )  # [INNER_BLOCK]

    out_offs = base_idx + k_offs  # [INNER_BLOCK]
    cap_mask = out_offs < verify_capacity  # [INNER_BLOCK] bool
    write_mask = in_range_mask & cap_mask  # [INNER_BLOCK] bool

    tl.store(
        out_verify_slot_indices_ptr + out_offs, slots.to(tl.int64), mask=write_mask
    )
    tl.store(
        out_verify_positions_ptr + out_offs, positions.to(tl.int64), mask=write_mask
    )
    tl.store(
        out_verify_prev_slot_indices_ptr + out_offs, prevs.to(tl.int64), mask=write_mask
    )
