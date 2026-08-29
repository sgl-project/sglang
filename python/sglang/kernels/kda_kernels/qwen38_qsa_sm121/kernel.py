# SPDX-License-Identifier: Apache-2.0

# KDA provenance: optimized by Codex and Kimi K3 agents through KDA-1.5
# (https://github.com/radixark/KDA-1.5).
# Task: https://github.com/radixark/KDA-1.5/pull/4 @
# 414ce456e14ae8546f77d9356d2c4d955c5bb7f1.
# Winning submission: b4181149c8884ddb; byte-exact submitted source SHA256:
# 4f9977f88abfea4393a2add3a2c9255699f7e13b981dbc1a976b024b3b00e909.
"""Shape-specialized Qwen3.8 packed QSA decode kernel for SM121."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    counter_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    split_program = tl.program_id(1)
    kv_head_idx = split_program // MAX_SPLITS
    split_idx = split_program - kv_head_idx * MAX_SPLITS
    slot = sequence_idx * NUM_KV_HEADS + kv_head_idx
    queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS

    query_idx = tl.load(cu_q_ptr + sequence_idx)
    kv_begin = tl.load(cu_k_ptr + sequence_idx)
    kv_end = tl.load(cu_k_ptr + sequence_idx + 1)
    kv_count = kv_end - kv_begin
    tile_count = tl.cdiv(kv_count, BLOCK_KV)

    # The split count depends only on live device metadata and the static
    # launch geometry, so CUDA Graph replay needs no host readback.
    batch = tl.num_programs(0)
    n_splits = 1
    if tile_count >= 1536 // BLOCK_KV:
        n_splits = 2
    elif tile_count >= 512 // BLOCK_KV and batch * NUM_KV_HEADS <= 4:
        n_splits = 4
    if batch == 1:
        if tile_count >= 1024 // BLOCK_KV:
            n_splits = 8
        elif tile_count >= 512 // BLOCK_KV:
            n_splits = 4
        elif tile_count >= 256 // BLOCK_KV:
            n_splits = 2
    if split_idx >= n_splits:
        return

    tile_lo = (tile_count * split_idx) // n_splits
    tile_hi = (tile_count * (split_idx + 1)) // n_splits
    kv_start = kv_begin + tile_lo * BLOCK_KV
    kv_stop = tl.minimum(kv_begin + tile_hi * BLOCK_KV, kv_end)

    m = tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM)
    q_head = kv_head_idx * queries_per_kv + m
    q_mask = m < queries_per_kv
    query = tl.load(
        q_ptr + query_idx * q_stride_t + q_head[:, None] * q_stride_h + d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )

    n0 = tl.arange(0, BLOCK_KV)
    k_rows = (
        k_ptr
        + kv_head_idx * k_stride_h
        + kv_start.to(tl.int64) * k_stride_t
        + n0[:, None] * k_stride_t
    )
    v_rows = (
        v_ptr
        + kv_head_idx * v_stride_h
        + kv_start.to(tl.int64) * v_stride_t
        + n0[:, None] * v_stride_t
    )

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    kv_len = kv_stop - kv_start
    full_end = kv_start + (kv_len // BLOCK_KV) * BLOCK_KV
    for block_start in range(kv_start, full_end, BLOCK_KV):
        keys = tl.load(k_rows + d[None, :])
        scores = tl.dot(query, tl.trans(keys)) * softmax_scale
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        values = tl.load(v_rows + d[None, :])
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), values
        )
        running_max = new_max
        k_rows += BLOCK_KV * k_stride_t
        v_rows += BLOCK_KV * v_stride_t

    if full_end < kv_stop:
        n = full_end + n0
        n_mask = n < kv_stop
        keys = tl.load(k_rows + d[None, :], mask=n_mask[:, None], other=0.0)
        scores = tl.dot(query, tl.trans(keys)) * softmax_scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        values = tl.load(v_rows + d[None, :], mask=n_mask[:, None], other=0.0)
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), values
        )
        running_max = new_max

    if n_splits == 1:
        output = accumulator / tl.where(running_sum > 0.0, running_sum, 1.0)[:, None]
        tl.store(
            out_ptr
            + query_idx * out_stride_t
            + q_head[:, None] * out_stride_h
            + d[None, :],
            output.to(out_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )
        return

    partial_row = (slot * MAX_SPLITS + split_idx) * BLOCK_M + m
    tl.store(partial_max_ptr + partial_row, running_max)
    tl.store(partial_sum_ptr + partial_row, running_sum)
    tl.store(
        partial_acc_ptr + partial_row[:, None] * HEAD_DIM + d[None, :],
        accumulator,
    )
    tl.debug_barrier()
    arrival = tl.atomic_add(counter_ptr + slot, 1, sem="acq_rel", scope="gpu")
    if arrival == n_splits - 1:
        merged_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
        for j in tl.static_range(MAX_SPLITS):
            j_ok = (j < n_splits) & (m < BLOCK_M)
            row = (slot * MAX_SPLITS + j) * BLOCK_M + m
            mj = tl.load(partial_max_ptr + row, mask=j_ok, other=-float("inf"))
            merged_max = tl.maximum(merged_max, mj)
        merged_sum = tl.zeros([BLOCK_M], tl.float32)
        merged_acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        for j in tl.static_range(MAX_SPLITS):
            j_ok = (j < n_splits) & (m < BLOCK_M)
            row = (slot * MAX_SPLITS + j) * BLOCK_M + m
            mj = tl.load(partial_max_ptr + row, mask=j_ok, other=-float("inf"))
            lj = tl.load(partial_sum_ptr + row, mask=j_ok, other=0.0)
            weight = tl.exp(mj - merged_max)
            merged_sum += weight * lj
            partial = tl.load(
                partial_acc_ptr + row[:, None] * HEAD_DIM + d[None, :],
                mask=j_ok[:, None],
                other=0.0,
            )
            merged_acc += weight[:, None] * partial
        output = merged_acc / tl.where(merged_sum > 0.0, merged_sum, 1.0)[:, None]
        tl.store(
            out_ptr
            + query_idx * out_stride_t
            + q_head[:, None] * out_stride_h
            + d[None, :],
            output.to(out_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )
        tl.atomic_xchg(counter_ptr + slot, 0, sem="release", scope="gpu")


# The worst-case TP1 scratch allocation at the qualified limit is 32.3 MiB.
_MAX_BATCH = 128
_MAX_KV_HEADS = 2
_BLOCK_M = 16
_MAX_SPLITS = 8
_MAX_SLOTS = _MAX_BATCH * _MAX_KV_HEADS
_HEAD_DIM = 256
_scratch: dict[int, tuple[torch.Tensor, ...]] = {}


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, ...]:
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    scratch = _scratch.get(device_index)
    if scratch is None:
        rows = _MAX_SLOTS * _MAX_SPLITS * _BLOCK_M
        scratch = (
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(rows * _HEAD_DIM, dtype=torch.float32, device=device),
            torch.zeros(_MAX_SLOTS, dtype=torch.int32, device=device),
        )
        _scratch[device_index] = scratch
    return scratch


def qwen38_qsa_sm121(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the KDA-generated SM121 Qwen3.8 QSA kernel."""
    output = torch.empty_like(q)
    partial_max, partial_sum, partial_acc, counters = _get_scratch(q.device)
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]

    # This is the measured shape/topology schedule. The TP1 q_rows=4 shape is
    # intentionally kept on BK64 because its short and saturated rows are
    # host-indistinguishable; the live cu_seqlens still choose their split count.
    use_bk32 = (num_kv_heads == 1 and batch < 12) or (num_kv_heads == 2 and batch < 4)
    block_kv = 32 if use_bk32 else 64
    stages = 3 if use_bk32 else 2
    _qsa_split_kernel[(batch, num_kv_heads * _MAX_SPLITS)](
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        partial_max,
        partial_sum,
        partial_acc,
        counters,
        softmax_scale,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_KV=block_kv,
        MAX_SPLITS=_MAX_SPLITS,
        q_stride_t=q.stride(0),
        q_stride_h=q.stride(1),
        k_stride_t=k.stride(0),
        k_stride_h=k.stride(1),
        v_stride_t=v.stride(0),
        v_stride_h=v.stride(1),
        out_stride_t=output.stride(0),
        out_stride_h=output.stride(1),
        num_warps=4,
        num_stages=stages,
    )
    return output
