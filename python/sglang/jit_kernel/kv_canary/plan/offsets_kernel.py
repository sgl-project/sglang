from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kv_canary.plan.utils import (
    _compute_window_start,
    _require_1d,
    _require_2d,
    _require_dtype,
    _require_len,
    _require_min_len,
    _require_same_device,
    _resolve_swa_lut,
    _swa_translate_tile,
)
from sglang.jit_kernel.kv_canary.verify import _assert_contiguous

# Upper bound on bs for _plan_offsets_kernel's block-level cumsum. Reqs larger than this exceed Triton's
# single-program tl.cumsum reach. Increase if real workloads ever push past it; the cap is intentionally
# generous so the wrapper never silently truncates.
_PLAN_BS_BLOCK_SIZE: int = 4096


def launch_plan_offsets_kernel(
    *,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    extra_verify_num_valid: torch.Tensor,
    verify_offsets_scratch: torch.Tensor,
    write_offsets: torch.Tensor,
    write_seed_slot_indices: torch.Tensor,
    verify_num_valid: torch.Tensor,
    verify_enable: torch.Tensor,
    write_num_valid_reqs: torch.Tensor,
    swa_window_size: int,
    verify_capacity: int,
) -> None:
    bs = int(fb_req_pool_indices.shape[0])
    lut_tensor, lut_len, has_swa_lut = _resolve_swa_lut(
        full_to_swa_index_mapping, verify_offsets_scratch.device
    )
    req_to_token_stride0 = int(req_to_token.stride(0))
    write_offsets_len = int(write_offsets.shape[0])
    write_req_capacity = int(write_seed_slot_indices.shape[0])

    _validate_offsets_kernel_inputs(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        lut_tensor=lut_tensor,
        extra_verify_num_valid=extra_verify_num_valid,
        verify_offsets_scratch=verify_offsets_scratch,
        write_offsets=write_offsets,
        write_seed_slot_indices=write_seed_slot_indices,
        verify_num_valid=verify_num_valid,
        verify_enable=verify_enable,
        write_num_valid_reqs=write_num_valid_reqs,
        bs=bs,
        req_to_token_stride0=req_to_token_stride0,
        lut_len=lut_len,
        has_swa_lut=has_swa_lut,
        write_offsets_len=write_offsets_len,
        write_req_capacity=write_req_capacity,
        verify_capacity=verify_capacity,
    )

    _plan_offsets_kernel[(1,)](
        fb_req_pool_indices,
        fb_prefix_lens,
        fb_extend_seq_lens,
        req_to_token,
        lut_tensor,
        extra_verify_num_valid,
        verify_offsets_scratch,
        write_offsets,
        write_seed_slot_indices,
        verify_num_valid,
        verify_enable,
        write_num_valid_reqs,
        bs,
        req_to_token_stride0,
        lut_len,
        BS_BLOCK=_PLAN_BS_BLOCK_SIZE,
        SWA_WINDOW=int(swa_window_size),
        HAS_SWA_LUT=has_swa_lut,
        WRITE_OFFSETS_LEN=write_offsets_len,
        WRITE_REQ_CAPACITY=write_req_capacity,
        VERIFY_CAPACITY=verify_capacity,
    )


def _validate_offsets_kernel_inputs(
    *,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    lut_tensor: torch.Tensor,
    extra_verify_num_valid: torch.Tensor,
    verify_offsets_scratch: torch.Tensor,
    write_offsets: torch.Tensor,
    write_seed_slot_indices: torch.Tensor,
    verify_num_valid: torch.Tensor,
    verify_enable: torch.Tensor,
    write_num_valid_reqs: torch.Tensor,
    bs: int,
    req_to_token_stride0: int,
    lut_len: int,
    has_swa_lut: bool,
    write_offsets_len: int,
    write_req_capacity: int,
    verify_capacity: int,
) -> None:
    _assert_contiguous(fb_req_pool_indices, "fb_req_pool_indices")
    _assert_contiguous(fb_prefix_lens, "fb_prefix_lens")
    _assert_contiguous(fb_extend_seq_lens, "fb_extend_seq_lens")
    _assert_contiguous(req_to_token, "req_to_token")
    _assert_contiguous(lut_tensor, "lut_tensor")
    _assert_contiguous(extra_verify_num_valid, "extra_verify_num_valid")
    _assert_contiguous(verify_offsets_scratch, "verify_offsets_scratch")
    _assert_contiguous(write_offsets, "write_offsets")
    _assert_contiguous(write_seed_slot_indices, "write_seed_slot_indices")
    _assert_contiguous(verify_num_valid, "verify_num_valid")
    _assert_contiguous(verify_enable, "verify_enable")
    _assert_contiguous(write_num_valid_reqs, "write_num_valid_reqs")

    _require_dtype(fb_req_pool_indices, "fb_req_pool_indices", torch.int64)
    _require_dtype(fb_prefix_lens, "fb_prefix_lens", torch.int64)
    _require_dtype(fb_extend_seq_lens, "fb_extend_seq_lens", torch.int64)
    _require_dtype(req_to_token, "req_to_token", torch.int32)
    _require_dtype(lut_tensor, "lut_tensor", torch.int64)
    _require_dtype(extra_verify_num_valid, "extra_verify_num_valid", torch.int32)
    _require_dtype(verify_offsets_scratch, "verify_offsets_scratch", torch.int64)
    _require_dtype(write_offsets, "write_offsets", torch.int64)
    _require_dtype(write_seed_slot_indices, "write_seed_slot_indices", torch.int64)
    _require_dtype(verify_num_valid, "verify_num_valid", torch.int32)
    _require_dtype(verify_enable, "verify_enable", torch.int32)
    _require_dtype(write_num_valid_reqs, "write_num_valid_reqs", torch.int32)

    if bs < 0 or bs > _PLAN_BS_BLOCK_SIZE:
        raise ValueError(
            f"kv-canary: offsets kernel bs must be in [0, {_PLAN_BS_BLOCK_SIZE}], got {bs}"
        )
    if write_offsets_len <= 0:
        raise ValueError(
            f"kv-canary: write_offsets_len must be positive, got {write_offsets_len}"
        )
    if write_req_capacity < 0:
        raise ValueError(
            f"kv-canary: write_req_capacity must be non-negative, got {write_req_capacity}"
        )
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
    _require_len(fb_extend_seq_lens, "fb_extend_seq_lens", bs)
    _require_2d(req_to_token, "req_to_token")
    _require_min_len(lut_tensor, "lut_tensor", max(lut_len, 1))
    _require_min_len(extra_verify_num_valid, "extra_verify_num_valid", 1)
    _require_min_len(
        verify_offsets_scratch,
        "verify_offsets_scratch",
        _PLAN_BS_BLOCK_SIZE + 1,
    )
    _require_len(write_offsets, "write_offsets", write_offsets_len)
    _require_len(write_seed_slot_indices, "write_seed_slot_indices", write_req_capacity)
    _require_len(verify_num_valid, "verify_num_valid", 1)
    _require_len(verify_enable, "verify_enable", 1)
    _require_len(write_num_valid_reqs, "write_num_valid_reqs", 1)
    _require_1d(lut_tensor, "lut_tensor")

    if write_offsets_len != write_req_capacity + 1:
        raise ValueError(
            f"kv-canary: write_offsets_len must equal write_req_capacity + 1, got "
            f"{write_offsets_len} and {write_req_capacity}"
        )
    if bs > write_req_capacity:
        raise ValueError(
            f"kv-canary: bs={bs} exceeds write_req_capacity={write_req_capacity}"
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
            (fb_extend_seq_lens, "fb_extend_seq_lens"),
            (req_to_token, "req_to_token"),
            (lut_tensor, "lut_tensor"),
            (extra_verify_num_valid, "extra_verify_num_valid"),
            (write_offsets, "write_offsets"),
            (write_seed_slot_indices, "write_seed_slot_indices"),
            (verify_num_valid, "verify_num_valid"),
            (verify_enable, "verify_enable"),
            (write_num_valid_reqs, "write_num_valid_reqs"),
        ),
    )


@triton.jit
def _plan_offsets_kernel(
    # Input pointers.
    req_pool_indices_ptr,
    prefix_lens_ptr,
    extend_seq_lens_ptr,
    req_to_token_ptr,
    full_to_swa_lut_ptr,
    extra_verify_num_valid_ptr,
    # Output pointers.
    out_verify_offsets_ptr,
    out_write_offsets_ptr,
    out_write_seed_slot_indices_ptr,
    out_verify_num_valid_ptr,
    out_verify_enable_ptr,
    out_write_num_valid_reqs_ptr,
    # Runtime sizes.
    bs,
    req_to_token_stride0,
    swa_lut_len,
    # Compile-time constants.
    BS_BLOCK: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SWA_LUT: tl.constexpr,
    WRITE_OFFSETS_LEN: tl.constexpr,
    WRITE_REQ_CAPACITY: tl.constexpr,
    VERIFY_CAPACITY: tl.constexpr,
):
    """Offsets kernel: per-req counts, seeds, exclusive-prefix-sum offsets, scalar totals.

    Single program; BLOCK_BS-wide tiles cover the full bs (caller ensures bs <= BS_BLOCK). All cumsum is done
    via block-level ``tl.cumsum`` in one program — no cross-program sync needed.
    """
    bs_offs = tl.arange(0, BS_BLOCK)  # [BS_BLOCK]
    bs_mask = bs_offs < bs  # [BS_BLOCK] bool

    # Per-req inputs (int64 for canary-owned metadata; req_to_token keeps its pool dtype).
    rpi = tl.load(req_pool_indices_ptr + bs_offs, mask=bs_mask, other=0)  # [BS_BLOCK]
    prefix_lens = tl.load(
        prefix_lens_ptr + bs_offs, mask=bs_mask, other=0
    )  # [BS_BLOCK]
    extend_lens = tl.load(
        extend_seq_lens_ptr + bs_offs, mask=bs_mask, other=0
    )  # [BS_BLOCK]

    is_active = (rpi != 0) & bs_mask  # [BS_BLOCK] bool
    has_prefix = is_active & (prefix_lens > 0)  # [BS_BLOCK] bool

    window_starts = _compute_window_start(prefix_lens, SWA_WINDOW)  # [BS_BLOCK]

    verify_lens = prefix_lens - window_starts  # [BS_BLOCK]
    verify_lens = tl.where(verify_lens > 0, verify_lens, 0)
    verify_lens = tl.where(is_active, verify_lens, 0)

    write_lens = tl.where(extend_lens > 0, extend_lens, 0)  # [BS_BLOCK]
    write_lens = tl.where(is_active, write_lens, 0)

    has_write_contribution = has_prefix & (write_lens > 0)  # [BS_BLOCK] bool

    # Seed slot per req. prefix_lens == 0 means no prefix → -1 sentinel. Padding row → no write contribution
    # → -1 sentinel either way; we also mask write_lens onto seed below to match the ref's "no write → -1".
    safe_prefix_pos = tl.where(prefix_lens > 0, prefix_lens - 1, 0)  # [BS_BLOCK]
    stride_i64 = req_to_token_stride0  # scalar
    seed_full = tl.load(  # [BS_BLOCK]
        req_to_token_ptr + rpi.to(tl.int64) * stride_i64 + safe_prefix_pos.to(tl.int64),
        mask=has_prefix,
        other=0,
    )

    if HAS_SWA_LUT:
        seed_translated = _swa_translate_tile(  # [BS_BLOCK]
            seed_full,
            has_prefix,
            full_to_swa_lut_ptr,
            swa_lut_len,
        )
    else:
        seed_translated = seed_full

    # Reqs with no write contribution should expose seed = -1 (ref's _seed_slot is masked by write_lens > 0).
    minus_one = tl.full((BS_BLOCK,), -1, dtype=seed_translated.dtype)  # [BS_BLOCK]
    seed_slot = tl.where(
        has_write_contribution, seed_translated, minus_one
    )  # [BS_BLOCK]

    # Inclusive cumsum → exclusive offsets via subtraction.
    verify_inclusive = tl.cumsum(verify_lens, axis=0)  # [BS_BLOCK]
    write_inclusive = tl.cumsum(write_lens, axis=0)  # [BS_BLOCK]
    verify_exclusive = verify_inclusive - verify_lens  # [BS_BLOCK]
    write_exclusive = write_inclusive - write_lens  # [BS_BLOCK]

    # Scatter exclusive offsets into the [bs+1]-sized output tensor. Positions [0, bs) get the exclusive sum;
    # position bs gets the total (totals = verify_inclusive at index bs - 1 if bs > 0, else 0).
    out_offsets_mask = bs_mask  # [BS_BLOCK] bool
    tl.store(
        out_verify_offsets_ptr + bs_offs,
        verify_exclusive.to(tl.int64),
        mask=out_offsets_mask,
    )
    write_offsets_mask = bs_offs < WRITE_OFFSETS_LEN  # [BS_BLOCK] bool
    tl.store(
        out_write_offsets_ptr + bs_offs,
        write_exclusive.to(tl.int64),
        mask=write_offsets_mask & bs_mask,
    )

    # Scatter seed slots (capped to write_req_capacity).
    seed_mask = bs_mask & (bs_offs < WRITE_REQ_CAPACITY)  # [BS_BLOCK] bool
    tl.store(
        out_write_seed_slot_indices_ptr + bs_offs,
        seed_slot.to(tl.int64),
        mask=seed_mask,
    )

    # Totals: sum of all per-req lens. Same value as the last inclusive entry but tl.sum is robust to bs == 0.
    total_verify = tl.sum(verify_lens, axis=0)  # scalar
    total_write = tl.sum(write_lens, axis=0)  # scalar

    # Store the [bs] slot of verify_offsets and write_offsets (one element past the last per-req entry).
    # verify_offsets scratch has length BS_BLOCK + 1 so the bs slot is always in range.
    tl.store(out_verify_offsets_ptr + bs, total_verify.to(tl.int64))
    # write_offsets has length WRITE_OFFSETS_LEN = write_req_capacity + 1; only store if in range.
    write_tail_in_range = bs < WRITE_OFFSETS_LEN  # scalar bool
    tl.store(
        out_write_offsets_ptr + bs,
        total_write.to(tl.int64),
        mask=write_tail_in_range,
    )

    # Scalar writes: verify_num_valid is clamped to the verify_capacity tensor extent so the verify kernel
    # never indexes past the buffer; enable carries the overflow bit (0 when requested > capacity) so the
    # verify kernel skips the whole launch and the host can warn-log this step.
    extras_count = tl.load(extra_verify_num_valid_ptr)  # scalar
    extras_count = tl.where(extras_count > 0, extras_count, 0)
    requested = total_verify + extras_count  # scalar
    overflow = requested > VERIFY_CAPACITY  # scalar bool
    enable = tl.where(overflow, 0, 1)  # scalar
    clamped = tl.where(overflow, VERIFY_CAPACITY, requested)  # scalar
    tl.store(out_verify_num_valid_ptr, clamped.to(tl.int32))
    tl.store(out_verify_enable_ptr, tl.full((), enable, tl.int32))
    tl.store(out_write_num_valid_reqs_ptr, tl.full((), bs, tl.int32))
