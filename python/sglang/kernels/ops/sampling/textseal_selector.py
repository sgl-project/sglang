# SPDX-License-Identifier: Apache-2.0
# Adapted from facebookresearch/textseal's Apache-2.0 selector implementation.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix

_BLOCK_SIZE = 8192
_HISTORY_BLOCK_SIZE = 1024
_UINT32_SCALE = tl.constexpr(float(1 << 32))


@triton.jit
def _watermark_partial_argmax_kernel(
    probabilities,
    context_hashes,
    keys,
    partial_scores,
    partial_token_ids,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    token_ids = split * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = token_ids < vocab_size
    candidate_probabilities = tl.load(
        probabilities + row * vocab_size + token_ids,
        mask=in_bounds,
        other=0.0,
    ).to(tl.float32)

    key = tl.load(keys + row).to(tl.uint64)
    state: tl.uint32 = 0
    state = murmur3_mix(state, (key & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, ((key >> 32) & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, tl.load(context_hashes + row).to(tl.uint32))
    state = murmur3_mix(state, token_ids.to(tl.uint32))
    hashed = fmix32(state ^ 16)
    uniform = (hashed.to(tl.float32) + 0.5) / _UINT32_SCALE

    is_candidate = in_bounds & (candidate_probabilities > 0.0)
    safe_probabilities = tl.where(is_candidate, candidate_probabilities, 1.0)
    scores = tl.where(
        is_candidate,
        tl.log(uniform) / safe_probabilities,
        -float("inf"),
    )
    local_index = tl.argmax(scores, axis=0, tie_break_left=True)
    output_offset = row * num_splits + split
    tl.store(partial_scores + output_offset, tl.max(scores, axis=0))
    tl.store(partial_token_ids + output_offset, split * BLOCK_SIZE + local_index)


@triton.jit
def _watermark_finalize_argmax_kernel(
    partial_scores,
    partial_token_ids,
    output_token_ids,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_splits
    scores = tl.load(
        partial_scores + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )
    split = tl.argmax(scores, axis=0, tie_break_left=True)
    token_id = tl.load(partial_token_ids + row * num_splits + split)
    tl.store(output_token_ids + row, token_id)


@triton.jit
def _prepare_watermark_contexts_kernel(
    token_ids,
    lengths,
    write_positions,
    watermarked_context_hashes,
    num_watermarked_contexts,
    req_pool_indices,
    context_windows,
    watermark_enabled,
    top_ks,
    output_context_hashes,
    output_eligible,
    context_window: tl.constexpr,
    max_contexts_per_req: tl.constexpr,
    HISTORY_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    pool_index = tl.load(req_pool_indices + row).to(tl.int64)
    length = tl.load(lengths + pool_index)
    write_position = tl.load(write_positions + pool_index)
    requested_window = tl.load(context_windows + row)
    context_length = tl.minimum(length, requested_window)
    start = tl.where(length == context_window, write_position, 0)
    source_start = length - context_length

    state = tl.full((), 0, tl.uint32)
    for index in range(context_window):
        ring_index = (start + source_start + index) % context_window
        token = tl.load(token_ids + pool_index * context_window + ring_index).to(
            tl.uint32
        )
        mixed = murmur3_mix(state, token).to(tl.uint32)
        state = tl.where(index < context_length, mixed, state).to(tl.uint32)
    context_hash = fmix32(state ^ (context_length * 4).to(tl.uint32))

    count = tl.load(num_watermarked_contexts + pool_index)
    eligible = (
        tl.load(watermark_enabled + row)
        & (tl.load(top_ks + row) > 1)
        & (context_length > 0)
        & (count < max_contexts_per_req)
    )
    repeated = False
    offset = 0
    while (offset < count) & eligible:
        positions = offset + tl.arange(0, HISTORY_BLOCK_SIZE)
        mask = positions < count
        previous_hashes = tl.load(
            watermarked_context_hashes + pool_index * max_contexts_per_req + positions,
            mask=mask,
            other=0,
        )
        repeated |= (
            tl.sum((previous_hashes == context_hash.to(tl.int32)) & mask, axis=0) > 0
        )
        offset += HISTORY_BLOCK_SIZE
    eligible &= ~repeated

    tl.store(output_context_hashes + row, context_hash.to(tl.int64))
    tl.store(output_eligible + row, eligible)
    tl.store(
        watermarked_context_hashes + pool_index * max_contexts_per_req + count,
        context_hash.to(tl.int32),
        mask=eligible,
    )
    tl.store(num_watermarked_contexts + pool_index, count + 1, mask=eligible)


@triton.jit
def _watermark_force_partial_argmax_kernel(
    logits,
    sorted_probabilities,
    sorted_token_ids,
    cumulative_probabilities,
    context_hashes,
    eligible,
    top_ks,
    top_ps,
    min_ps,
    keys,
    partial_scores,
    partial_token_ids,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    ranks = split * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = ranks < vocab_size
    sorted_offsets = row * vocab_size + ranks
    probabilities = tl.load(
        sorted_probabilities + sorted_offsets, mask=in_bounds, other=0.0
    ).to(tl.float32)
    cumulative = tl.load(
        cumulative_probabilities + sorted_offsets, mask=in_bounds, other=0.0
    ).to(tl.float32)
    token_ids = tl.load(sorted_token_ids + sorted_offsets, mask=in_bounds, other=0).to(
        tl.uint32
    )
    row_eligible = tl.load(eligible + row)
    top_k = tl.load(top_ks + row)
    top_p = tl.load(top_ps + row)
    min_p = tl.load(min_ps + row)
    max_probability = tl.load(sorted_probabilities + row * vocab_size).to(tl.float32)
    is_candidate = (
        row_eligible
        & in_bounds
        & (probabilities > 0.0)
        & (ranks < top_k)
        & ((cumulative - probabilities) <= top_p)
        & (probabilities >= max_probability * min_p)
    )

    key = tl.load(keys + row).to(tl.uint64)
    state: tl.uint32 = 0
    state = murmur3_mix(state, (key & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, ((key >> 32) & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, tl.load(context_hashes + row).to(tl.uint32))
    state = murmur3_mix(state, token_ids)
    hashed = fmix32(state ^ 16)
    uniform = (hashed.to(tl.float32) + 0.5) / _UINT32_SCALE
    safe_probabilities = tl.where(is_candidate, probabilities, 1.0)
    scores = tl.where(
        is_candidate,
        tl.log(uniform) / safe_probabilities,
        -float("inf"),
    )
    local_score = tl.max(scores, axis=0)
    local_token = tl.min(tl.where(scores == local_score, token_ids, 0xFFFFFFFF), axis=0)
    output_offset = row * num_splits + split
    tl.store(partial_scores + output_offset, local_score)
    tl.store(partial_token_ids + output_offset, local_token.to(tl.int32))
    tl.store(
        logits + row * vocab_size + ranks,
        -float("inf"),
        mask=row_eligible & in_bounds,
    )


@triton.jit
def _watermark_finalize_and_write_kernel(
    logits,
    eligible,
    partial_scores,
    partial_token_ids,
    output_token_ids,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_splits
    scores = tl.load(
        partial_scores + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )
    max_score = tl.max(scores, axis=0)
    token_ids = tl.load(
        partial_token_ids + row * num_splits + offsets, mask=mask, other=-1
    )
    token_id = tl.min(
        tl.where(mask & (scores == max_score), token_ids, 0x7FFFFFFF), axis=0
    )
    token_id = tl.where(max_score == -float("inf"), 0, token_id)
    row_eligible = tl.load(eligible + row)
    tl.store(logits + row * vocab_size + token_id, 0.0, mask=row_eligible)
    tl.store(output_token_ids + row, tl.where(row_eligible, token_id, -1))


def select_watermark_tokens_triton(
    probabilities: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.dtype != torch.float32:
        raise ValueError("probabilities must be a 2D float32 tensor")
    if not probabilities.is_cuda:
        raise ValueError("Triton watermark selection requires CUDA tensors")
    batch_size, vocab_size = probabilities.shape
    if context_hashes.shape != (batch_size,) or context_hashes.dtype != torch.int64:
        raise ValueError("context_hashes must be int64 with one value per row")
    if keys.shape != (batch_size,) or keys.dtype != torch.int64:
        raise ValueError("keys must be int64 with one value per row")
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (probabilities, context_hashes, keys)
    ):
        raise ValueError("Triton watermark inputs must be contiguous CUDA tensors")

    output_token_ids = torch.empty(
        batch_size, dtype=torch.int32, device=probabilities.device
    )
    if batch_size == 0:
        return output_token_ids

    num_splits = triton.cdiv(vocab_size, _BLOCK_SIZE)
    partial_scores = torch.empty(
        (batch_size, num_splits), dtype=torch.float32, device=probabilities.device
    )
    partial_token_ids = torch.empty(
        (batch_size, num_splits), dtype=torch.int32, device=probabilities.device
    )
    _watermark_partial_argmax_kernel[(batch_size, num_splits)](
        probabilities,
        context_hashes,
        keys,
        partial_scores,
        partial_token_ids,
        vocab_size=vocab_size,
        num_splits=num_splits,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    _watermark_finalize_argmax_kernel[(batch_size,)](
        partial_scores,
        partial_token_ids,
        output_token_ids,
        num_splits=num_splits,
        BLOCK_SIZE=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return output_token_ids


def prepare_watermark_contexts_triton(
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    write_positions: torch.Tensor,
    watermarked_context_hashes: torch.Tensor,
    num_watermarked_contexts: torch.Tensor,
    req_pool_indices: torch.Tensor,
    context_windows: torch.Tensor,
    watermark_enabled: torch.Tensor,
    top_ks: torch.Tensor,
    output_context_hashes: torch.Tensor,
    output_eligible: torch.Tensor,
) -> None:
    batch_size = req_pool_indices.shape[0]
    if batch_size == 0:
        return
    context_window = token_ids.shape[1]
    max_contexts_per_req = watermarked_context_hashes.shape[1]
    _prepare_watermark_contexts_kernel[(batch_size,)](
        token_ids,
        lengths,
        write_positions,
        watermarked_context_hashes,
        num_watermarked_contexts,
        req_pool_indices,
        context_windows,
        watermark_enabled,
        top_ks,
        output_context_hashes,
        output_eligible,
        context_window=context_window,
        max_contexts_per_req=max_contexts_per_req,
        HISTORY_BLOCK_SIZE=_HISTORY_BLOCK_SIZE,
        num_warps=8,
    )


def force_watermark_tokens_triton(
    logits: torch.Tensor,
    context_hashes: torch.Tensor,
    eligible: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits / temperatures, dim=-1)
    sorted_probabilities, sorted_token_ids = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    batch_size, vocab_size = logits.shape
    output_token_ids = torch.empty(batch_size, dtype=torch.int32, device=logits.device)
    if batch_size == 0:
        return output_token_ids

    num_splits = triton.cdiv(vocab_size, _BLOCK_SIZE)
    partial_scores = torch.empty(
        (batch_size, num_splits), dtype=torch.float32, device=logits.device
    )
    partial_token_ids = torch.empty(
        (batch_size, num_splits), dtype=torch.int32, device=logits.device
    )
    _watermark_force_partial_argmax_kernel[(batch_size, num_splits)](
        logits,
        sorted_probabilities,
        sorted_token_ids,
        cumulative_probabilities,
        context_hashes,
        eligible,
        top_ks,
        top_ps,
        min_ps,
        keys,
        partial_scores,
        partial_token_ids,
        vocab_size=vocab_size,
        num_splits=num_splits,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    _watermark_finalize_and_write_kernel[(batch_size,)](
        logits,
        eligible,
        partial_scores,
        partial_token_ids,
        output_token_ids,
        vocab_size=vocab_size,
        num_splits=num_splits,
        BLOCK_SIZE=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return output_token_ids
