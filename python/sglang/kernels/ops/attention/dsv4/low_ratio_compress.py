"""Triton kernel for DeepSeek V4.1 low-ratio (ratio=2) compression.

Replaces the torch _low_ratio_compress_torch path (prefill/extend only) with
a fused triton kernel that:
  1. Gathers partner kv/score for each odd-positioned token (from the previous
     token in-batch or from per-request state).
  2. Computes the softmax-weighted pool of the pair.
  3. Updates per-request pair state with the last even token of each request.

The kernel produces compacted (pooled, out_loc, group_pos) ready for
_low_ratio_write_group.

Two separate kernels are used to avoid RAW hazards on the per-request state
(odd tokens read old state; even tokens write new state for the next batch).

== Tiled implementation (default) ==

Each program processes BLOCK_N consecutive tokens (instead of 1), reducing
program count by BLOCK_N× and eliminating the in-kernel atomic counter:
  - Output slot offsets are precomputed on the host (cumsum of odd-count per
    block); the kernel uses tl.cumsum for in-block positioning.
  - On NPU this fixes a launch-bound bottleneck (22-29× speedup at N=8K).
  - BLOCK_N is auto-selected to fit the device's on-chip buffer (NPU UB
    ~220 KB, GPU shared memory / register file more permissive).

Set SGLANG_LOW_RATIO_COMPRESS_BLOCK_N to force a specific tiling factor,
or SGLANG_LOW_RATIO_COMPRESS_USE_ORIG=1 to use the original per-token kernel.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Tiled kernels (default): BLOCK_N tokens per program
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["N"])
def _low_ratio_pool_kernel_tiled(
    kv_ptr,
    score_ptr,
    pos_ptr,
    req_ptr,
    out_loc_ptr,
    state_kv_ptr,
    state_score_ptr,
    block_offset_ptr,  # [num_blocks+1] int32, precomputed output offsets
    pooled_ptr,
    compacted_out_loc_ptr,
    compacted_group_pos_ptr,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    token_start = pid * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    token_idx = token_start + offs_n  # [BLOCK_N]
    valid = token_idx < N  # [BLOCK_N]

    # -- Load pos, req for this block --
    pos_block = tl.load(pos_ptr + token_idx, mask=valid, other=0)  # [BLOCK_N]
    req_block = tl.load(req_ptr + token_idx, mask=valid, other=-1)
    is_odd = (pos_block & 1) == 1  # [BLOCK_N]

    # -- Load kv, score: [BLOCK_N, BLOCK_D] --
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    self_ptrs = token_idx[:, None] * D + offs_d[None, :]
    mask2d = valid[:, None] & d_mask[None, :]
    kv_self = tl.load(kv_ptr + self_ptrs, mask=mask2d, other=0.0)  # [BLOCK_N, D]
    score_self = tl.load(score_ptr + self_ptrs, mask=mask2d, other=0.0)

    # -- Partner via shift-by-1 (in-batch pairing) --
    prev_idx = token_idx - 1  # [BLOCK_N]; -1 for pid=0,token=0 (even, skipped)
    prev_valid = (prev_idx >= 0) & valid
    prev_pos = tl.load(pos_ptr + prev_idx, mask=prev_valid, other=-999)
    prev_req = tl.load(req_ptr + prev_idx, mask=prev_valid, other=-1)
    paired_in_batch = prev_valid & (req_block == prev_req) & (pos_block == prev_pos + 1)

    prev_ptrs = prev_idx[:, None] * D + offs_d[None, :]
    prev_mask = prev_valid[:, None] & d_mask[None, :]
    kv_partner_batch = tl.load(kv_ptr + prev_ptrs, mask=prev_mask, other=0.0)
    score_partner_batch = tl.load(score_ptr + prev_ptrs, mask=prev_mask, other=0.0)

    # -- Partner from per-request state (for odd tokens not paired in batch) --
    state_ptrs = req_block[:, None] * D + offs_d[None, :]
    state_mask = valid[:, None] & d_mask[None, :]
    kv_partner_state = tl.load(state_kv_ptr + state_ptrs, mask=state_mask, other=0.0)
    score_partner_state = tl.load(
        state_score_ptr + state_ptrs, mask=state_mask, other=0.0
    )

    # -- Select partner source --
    use_batch = paired_in_batch[:, None]
    kv_partner = tl.where(use_batch, kv_partner_batch, kv_partner_state)
    score_partner = tl.where(use_batch, score_partner_batch, score_partner_state)

    # -- Per-dimension softmax pool --
    smax = tl.maximum(score_partner, score_self)
    e0 = tl.exp(score_partner - smax)
    e1 = tl.exp(score_self - smax)
    pooled_val = (kv_partner * e0 + kv_self * e1) / (e0 + e1)  # [BLOCK_N, D]

    # -- Deterministic output position via block-offset + in-block cumsum --
    out_start = tl.load(block_offset_ptr + pid)
    is_odd_int = is_odd.to(tl.int32)
    rank = tl.cumsum(is_odd_int, axis=0) - is_odd_int  # exclusive prefix sum
    out_idx = out_start + rank  # [BLOCK_N] global output index

    # -- Store only odd rows --
    out_ptrs = pooled_ptr + out_idx[:, None] * D + offs_d[None, :]
    store_mask = is_odd[:, None] & valid[:, None] & d_mask[None, :]
    tl.store(out_ptrs, pooled_val, mask=store_mask)

    out_loc_vals = tl.load(out_loc_ptr + token_idx, mask=valid, other=0)
    tl.store(
        compacted_out_loc_ptr + out_idx,
        out_loc_vals,
        mask=is_odd & valid,
    )
    tl.store(
        compacted_group_pos_ptr + out_idx,
        pos_block - 1,
        mask=is_odd & valid,
    )


@triton.jit(do_not_specialize=["N"])
def _low_ratio_state_kernel_tiled(
    kv_ptr,
    score_ptr,
    pos_ptr,
    req_ptr,
    state_kv_ptr,
    state_score_ptr,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    token_start = pid * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    token_idx = token_start + offs_n  # [BLOCK_N]
    valid = token_idx < N

    pos_block = tl.load(pos_ptr + token_idx, mask=valid, other=0)
    req_block = tl.load(req_ptr + token_idx, mask=valid, other=-1)
    is_even = (pos_block & 1) == 0

    # Look ahead 2 tokens (global load, handles cross-block boundary)
    has_next = (token_idx + 1) < N
    has_next2 = (token_idx + 2) < N
    next_pos = tl.load(pos_ptr + token_idx + 1, mask=has_next & valid, other=-999)
    next_req = tl.load(req_ptr + token_idx + 1, mask=has_next & valid, other=-1)
    next2_req = tl.load(req_ptr + token_idx + 2, mask=has_next2 & valid, other=-1)

    is_last_of_req = (~has_next) | (req_block != next_req)
    next_same = has_next & (next_req == req_block)
    next_odd = (next_pos & 1) == 1
    next_is_last = (~has_next2) | (next_req != next2_req)
    next_last_odd = next_same & next_odd & next_is_last

    is_last_even = is_last_of_req | next_last_odd
    should_write = is_even & is_last_even & valid  # [BLOCK_N]

    # Load kv, score for the block
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    ptrs = token_idx[:, None] * D + offs_d[None, :]
    mask2d = valid[:, None] & d_mask[None, :]
    kv_val = tl.load(kv_ptr + ptrs, mask=mask2d, other=0.0)
    score_val = tl.load(score_ptr + ptrs, mask=mask2d, other=0.0)

    # Store to state (only should_write rows; per-request, no conflict)
    state_ptrs = req_block[:, None] * D + offs_d[None, :]
    store_mask = should_write[:, None] & d_mask[None, :]
    tl.store(state_kv_ptr + state_ptrs, kv_val, mask=store_mask)
    tl.store(state_score_ptr + state_ptrs, score_val, mask=store_mask)


# ---------------------------------------------------------------------------
# Original per-token kernels (fallback)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["N"])
def _low_ratio_pool_kernel_orig(
    kv_ptr,
    score_ptr,
    pos_ptr,
    req_ptr,
    out_loc_ptr,
    state_kv_ptr,
    state_score_ptr,
    counter_ptr,
    pooled_ptr,
    compacted_out_loc_ptr,
    compacted_group_pos_ptr,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    pos_i = tl.load(pos_ptr + pid)
    if (pos_i & 1) == 0:
        return

    req_i = tl.load(req_ptr + pid)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    kv_self = tl.load(kv_ptr + pid * D + d, mask=d_mask, other=0.0)
    score_self = tl.load(score_ptr + pid * D + d, mask=d_mask, other=0.0)

    has_prev = pid > 0
    req_prev = tl.load(req_ptr + pid - 1, mask=has_prev, other=-1)
    pos_prev = tl.load(pos_ptr + pid - 1, mask=has_prev, other=0)
    paired = has_prev & (req_i == req_prev) & (pos_i == pos_prev + 1)

    safe_prev = tl.maximum(pid - 1, 0)
    kv_partner_batch = tl.load(
        kv_ptr + safe_prev * D + d, mask=d_mask, other=0.0
    )
    score_partner_batch = tl.load(
        score_ptr + safe_prev * D + d, mask=d_mask, other=0.0
    )
    kv_partner_state = tl.load(
        state_kv_ptr + req_i * D + d, mask=d_mask, other=0.0
    )
    score_partner_state = tl.load(
        state_score_ptr + req_i * D + d, mask=d_mask, other=0.0
    )

    kv_partner = tl.where(paired, kv_partner_batch, kv_partner_state)
    score_partner = tl.where(paired, score_partner_batch, score_partner_state)

    max_s = tl.maximum(score_partner, score_self)
    exp_partner = tl.exp(score_partner - max_s)
    exp_self = tl.exp(score_self - max_s)
    total = exp_partner + exp_self
    pooled_val = (exp_partner * kv_partner + exp_self * kv_self) / total

    out_idx = tl.atomic_add(counter_ptr, 1)
    tl.store(pooled_ptr + out_idx * D + d, pooled_val, mask=d_mask)
    tl.store(compacted_out_loc_ptr + out_idx, tl.load(out_loc_ptr + pid))
    tl.store(compacted_group_pos_ptr + out_idx, pos_i - 1)


@triton.jit(do_not_specialize=["N"])
def _low_ratio_state_kernel_orig(
    kv_ptr,
    score_ptr,
    pos_ptr,
    req_ptr,
    state_kv_ptr,
    state_score_ptr,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    pos_i = tl.load(pos_ptr + pid)
    if (pos_i & 1) == 1:
        return

    req_i = tl.load(req_ptr + pid)

    has_next = pid < N - 1
    has_next2 = pid < N - 2

    req_next = tl.load(req_ptr + pid + 1, mask=has_next, other=-1)
    req_next2 = tl.load(req_ptr + pid + 2, mask=has_next2, other=-1)
    pos_next = tl.load(pos_ptr + pid + 1, mask=has_next, other=0)

    is_last_of_req = (~has_next) | (req_i != req_next)

    next_same = has_next & (req_next == req_i)
    next_odd = (pos_next & 1) == 1
    next_is_last = (~has_next2) | (req_next != req_next2)
    next_last_odd = next_same & next_odd & next_is_last

    is_last_even = is_last_of_req | next_last_odd
    if not is_last_even:
        return

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    kv_val = tl.load(kv_ptr + pid * D + d, mask=d_mask, other=0.0)
    score_val = tl.load(score_ptr + pid * D + d, mask=d_mask, other=0.0)
    tl.store(state_kv_ptr + req_i * D + d, kv_val, mask=d_mask)
    tl.store(state_score_ptr + req_i * D + d, score_val, mask=d_mask)


# ---------------------------------------------------------------------------
# BLOCK_N selection
# ---------------------------------------------------------------------------

# On-chip buffer budget (bytes) for the pool kernel's ~8 fp32 tiles of
# [BLOCK_N, BLOCK_D].  NPU Unified Buffer ~220 KB; GPU shared memory / register
# file is more permissive but triton still benefits from a conservative cap.
_NPU_UB_BUDGET = 180_000
_GPU_UB_BUDGET = 1_000_000


def _is_npu_device(device: torch.device) -> bool:
    return device.type == "npu"


def _choose_block_n(D: int, device: torch.device, requested: int = 64) -> int:
    """Pick the largest power-of-two BLOCK_N that fits the device buffer."""
    BLOCK_D = triton.next_power_of_2(D)
    budget = _NPU_UB_BUDGET if _is_npu_device(device) else _GPU_UB_BUDGET
    num_buffers = 8
    max_bn = budget // (num_buffers * BLOCK_D * 4)
    bn = 1
    while bn * 2 <= min(requested, max_bn):
        bn *= 2
    return max(1, bn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def low_ratio_compress_triton(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    req: torch.Tensor,
    out_loc: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused ratio-2 compression kernel for prefill/extend.

    Args:
        kv:    [N, D] float32 — projected kv from compressor.project(x).
        score: [N, D] float32 — projected gate score from compressor.project(x).
        pos:   [N] int64 — causal positions.
        req:   [N] int64 — request pool indices (one per token).
        out_loc: [N] int64 — c2_out_loc from compression metadata (-1 = skip).
        state_kv:   [num_req_slots+1, D] float32 — per-request pair state.
        state_score: [num_req_slots+1, D] float32 — per-request pair state.

    Returns:
        pooled:             [num_odd, D] float32 — softmax-weighted pair pool.
        compacted_out_loc:  [num_odd] int64 — cache write slots for odd tokens.
        compacted_group_pos:[num_odd] int64 — group base positions (pos - 1).
    """
    N, D = kv.shape
    device = kv.device

    if N == 0:
        return (
            torch.empty(0, D, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )

    assert kv.dtype == torch.float32 and score.dtype == torch.float32
    assert kv.is_contiguous() and score.is_contiguous()
    assert pos.dtype == torch.int64 and req.dtype == torch.int64
    assert state_kv.dtype == torch.float32 and state_score.dtype == torch.float32

    use_orig = os.environ.get("SGLANG_LOW_RATIO_COMPRESS_USE_ORIG", "0") == "1"
    if use_orig:
        return _low_ratio_compress_triton_orig(
            kv, score, pos, req, out_loc, state_kv, state_score
        )

    # -- Tiled path (default) --
    env_bn = os.environ.get("SGLANG_LOW_RATIO_COMPRESS_BLOCK_N", "")
    BLOCK_N = int(env_bn) if env_bn else _choose_block_n(D, device)

    max_odd = (N + 1) // 2
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    # Host-side: precompute per-block output offsets (eliminates atomic_add)
    odd_mask = ((pos & 1) == 1).to(torch.int32)  # [N]
    padded_len = num_blocks * BLOCK_N
    if padded_len > N:
        odd_padded = torch.zeros(padded_len, dtype=torch.int32, device=device)
        odd_padded[:N] = odd_mask
    else:
        odd_padded = odd_mask
    odd_per_block = odd_padded.view(num_blocks, BLOCK_N).sum(dim=1)  # [num_blocks]
    block_offsets = torch.zeros(num_blocks + 1, dtype=torch.int32, device=device)
    block_offsets[1:] = odd_per_block.cumsum(0)

    pooled = torch.empty((max_odd, D), dtype=torch.float32, device=device)
    compacted_out_loc = torch.empty(max_odd, dtype=torch.int64, device=device)
    compacted_group_pos = torch.empty(max_odd, dtype=torch.int64, device=device)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (num_blocks,)

    _low_ratio_pool_kernel_tiled[grid](
        kv,
        score,
        pos,
        req,
        out_loc,
        state_kv,
        state_score,
        block_offsets,
        pooled,
        compacted_out_loc,
        compacted_group_pos,
        N,
        D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    _low_ratio_state_kernel_tiled[grid](
        kv,
        score,
        pos,
        req,
        state_kv,
        state_score,
        N,
        D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    num_odd = block_offsets[-1].item()
    return (
        pooled[:num_odd],
        compacted_out_loc[:num_odd],
        compacted_group_pos[:num_odd],
    )


def _low_ratio_compress_triton_orig(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    req: torch.Tensor,
    out_loc: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original per-token kernel path (grid=(N,), atomic counter).

    Kept for debugging / fallback via SGLANG_LOW_RATIO_COMPRESS_USE_ORIG=1.
    """
    N, D = kv.shape
    device = kv.device
    max_odd = (N + 1) // 2
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    pooled = torch.empty((max_odd, D), dtype=torch.float32, device=device)
    compacted_out_loc = torch.empty(max_odd, dtype=torch.int64, device=device)
    compacted_group_pos = torch.empty(max_odd, dtype=torch.int64, device=device)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (N,)

    _low_ratio_pool_kernel_orig[grid](
        kv,
        score,
        pos,
        req,
        out_loc,
        state_kv,
        state_score,
        counter,
        pooled,
        compacted_out_loc,
        compacted_group_pos,
        N,
        D,
        BLOCK_D=BLOCK_D,
    )

    _low_ratio_state_kernel_orig[grid](
        kv,
        score,
        pos,
        req,
        state_kv,
        state_score,
        N,
        D,
        BLOCK_D=BLOCK_D,
    )

    num_odd = counter.item()
    return (
        pooled[:num_odd],
        compacted_out_loc[:num_odd],
        compacted_group_pos[:num_odd],
    )
