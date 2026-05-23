from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.plan.entries_kernel_cuda import (
    launch_plan_entries_cuda,
)
from sglang.jit_kernel.kv_canary.plan.utils import (
    _require_2d,
    _require_dtype,
    _require_len,
    _require_min_len,
    _require_same_device,
    _resolve_swa_lut,
)
from sglang.jit_kernel.kv_canary.verify import _assert_contiguous


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

    launch_plan_entries_cuda(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_offsets_scratch=verify_offsets_scratch,
        out_verify_slot_indices=out_verify_slot_indices,
        out_verify_positions=out_verify_positions,
        out_verify_prev_slot_indices=out_verify_prev_slot_indices,
        swa_window_size=int(swa_window_size),
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
