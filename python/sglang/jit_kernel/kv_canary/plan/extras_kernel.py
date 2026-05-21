from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.plan.utils import (
    _require_dtype,
    _require_len,
    _require_min_len,
    _require_same_device,
)
from sglang.jit_kernel.kv_canary.verify import _assert_contiguous

# Inner-tile width for _plan_extras_kernel. Each program copies this many extras into the verify tail.
_PLAN_EXTRAS_INNER_BLOCK: int = 64


def launch_plan_extras_kernel(
    *,
    extra_verify_slot_indices: torch.Tensor,
    extra_verify_positions: torch.Tensor,
    extra_verify_prev_slot_indices: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    verify_offsets_scratch: torch.Tensor,
    out_verify_slot_indices: torch.Tensor,
    out_verify_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    bs: int,
) -> None:
    verify_capacity = int(out_verify_slot_indices.shape[0])
    extras_capacity = int(extra_verify_slot_indices.shape[0])

    _validate_extras_kernel_inputs(
        extra_verify_slot_indices=extra_verify_slot_indices,
        extra_verify_positions=extra_verify_positions,
        extra_verify_prev_slot_indices=extra_verify_prev_slot_indices,
        extra_verify_num_valid=extra_verify_num_valid,
        verify_offsets_scratch=verify_offsets_scratch,
        out_verify_slot_indices=out_verify_slot_indices,
        out_verify_positions=out_verify_positions,
        out_verify_prev_slot_indices=out_verify_prev_slot_indices,
        bs=bs,
        verify_capacity=verify_capacity,
        extras_capacity=extras_capacity,
    )

    if extras_capacity == 0:
        return

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
        out_verify_slot_indices,
        out_verify_positions,
        out_verify_prev_slot_indices,
        bs,
        verify_capacity,
        INNER_BLOCK=_PLAN_EXTRAS_INNER_BLOCK,
    )


def _validate_extras_kernel_inputs(
    *,
    extra_verify_slot_indices: torch.Tensor,
    extra_verify_positions: torch.Tensor,
    extra_verify_prev_slot_indices: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    verify_offsets_scratch: torch.Tensor,
    out_verify_slot_indices: torch.Tensor,
    out_verify_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    bs: int,
    verify_capacity: int,
    extras_capacity: int,
) -> None:
    _assert_contiguous(extra_verify_slot_indices, "extra_verify_slot_indices")
    _assert_contiguous(extra_verify_positions, "extra_verify_positions")
    _assert_contiguous(extra_verify_prev_slot_indices, "extra_verify_prev_slot_indices")
    _assert_contiguous(extra_verify_num_valid, "extra_verify_num_valid")
    _assert_contiguous(verify_offsets_scratch, "verify_offsets_scratch")
    _assert_contiguous(out_verify_slot_indices, "out_verify_slot_indices")
    _assert_contiguous(out_verify_positions, "out_verify_positions")
    _assert_contiguous(out_verify_prev_slot_indices, "out_verify_prev_slot_indices")

    _require_dtype(extra_verify_slot_indices, "extra_verify_slot_indices", torch.int64)
    _require_dtype(extra_verify_positions, "extra_verify_positions", torch.int64)
    _require_dtype(
        extra_verify_prev_slot_indices, "extra_verify_prev_slot_indices", torch.int64
    )
    _require_dtype(extra_verify_num_valid, "extra_verify_num_valid", torch.int32)
    _require_dtype(verify_offsets_scratch, "verify_offsets_scratch", torch.int64)
    _require_dtype(out_verify_slot_indices, "out_verify_slot_indices", torch.int64)
    _require_dtype(out_verify_positions, "out_verify_positions", torch.int64)
    _require_dtype(
        out_verify_prev_slot_indices, "out_verify_prev_slot_indices", torch.int64
    )

    if bs < 0:
        raise ValueError(f"kv-canary: extras kernel bs must be non-negative, got {bs}")
    if verify_capacity < 0:
        raise ValueError(
            f"kv-canary: verify_capacity must be non-negative, got {verify_capacity}"
        )
    if extras_capacity < 0:
        raise ValueError(
            f"kv-canary: extras_capacity must be non-negative, got {extras_capacity}"
        )

    _require_len(
        extra_verify_slot_indices, "extra_verify_slot_indices", extras_capacity
    )
    _require_len(extra_verify_positions, "extra_verify_positions", extras_capacity)
    _require_len(
        extra_verify_prev_slot_indices,
        "extra_verify_prev_slot_indices",
        extras_capacity,
    )
    _require_min_len(extra_verify_num_valid, "extra_verify_num_valid", 1)
    _require_min_len(verify_offsets_scratch, "verify_offsets_scratch", bs + 1)
    _require_len(out_verify_slot_indices, "out_verify_slot_indices", verify_capacity)
    _require_len(out_verify_positions, "out_verify_positions", verify_capacity)
    _require_len(
        out_verify_prev_slot_indices, "out_verify_prev_slot_indices", verify_capacity
    )

    _require_same_device(
        verify_offsets_scratch,
        "verify_offsets_scratch",
        (
            (extra_verify_slot_indices, "extra_verify_slot_indices"),
            (extra_verify_positions, "extra_verify_positions"),
            (extra_verify_prev_slot_indices, "extra_verify_prev_slot_indices"),
            (extra_verify_num_valid, "extra_verify_num_valid"),
            (out_verify_slot_indices, "out_verify_slot_indices"),
            (out_verify_positions, "out_verify_positions"),
            (out_verify_prev_slot_indices, "out_verify_prev_slot_indices"),
        ),
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
