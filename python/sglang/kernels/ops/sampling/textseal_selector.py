# SPDX-License-Identifier: Apache-2.0
# Adapted from facebookresearch/textseal's Apache-2.0 selector implementation.

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix

_BLOCK_SIZE = 8192
_HISTORY_BLOCK_SIZE = 1024
_CLEAR_BLOCK_SIZE = 1024
_MAX_FAST_TOP_K = 8192
_UINT32_SCALE = tl.constexpr(float(1 << 32))


def watermark_selector_num_splits(vocab_size: int) -> int:
    return triton.cdiv(vocab_size, _BLOCK_SIZE)


def can_use_finite_topk_watermark(max_top_k: int | None, vocab_size: int) -> bool:
    return (
        max_top_k is not None
        and 1 < max_top_k < vocab_size
        and max_top_k <= _MAX_FAST_TOP_K
        and vocab_size >= 64
    )


@triton.jit
def _log_uniform_from_hash(hashed):
    # fp32 (h + 0.5) / 2^32 rounds to 1.0 for h >= 2^32 - 128; log1p of the
    # complement keeps log(u) strictly negative there.
    lower = tl.log((hashed.to(tl.float32) + 0.5) / _UINT32_SCALE)
    complement = (0xFFFFFFFF - hashed).to(tl.float32) + 0.5
    upper = libdevice.log1p(-complement / _UINT32_SCALE)
    return tl.where(hashed < 0x80000000, lower, upper)


@triton.jit
def _canonicalize_topk_kernel(
    probabilities,
    topk_probabilities,
    topk_token_ids,
    vocab_size: tl.constexpr,
    top_k,
    BLOCK_K: tl.constexpr,
    SCAN_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ranks = tl.arange(0, BLOCK_K)
    rank_mask = ranks < top_k
    row_offset = row * top_k
    candidate_probabilities = tl.load(
        topk_probabilities + row_offset + ranks,
        mask=rank_mask,
        other=float("inf"),
    ).to(tl.float32)
    candidate_token_ids = tl.load(
        topk_token_ids + row_offset + ranks, mask=rank_mask, other=0
    ).to(tl.int64)
    boundary = tl.min(candidate_probabilities, axis=0)
    above_boundary = rank_mask & (candidate_probabilities > boundary)
    above_positions = tl.cumsum(above_boundary.to(tl.int32), axis=0) - 1
    tl.store(
        topk_probabilities + row_offset + above_positions,
        candidate_probabilities,
        mask=above_boundary,
    )
    tl.store(
        topk_token_ids + row_offset + above_positions,
        candidate_token_ids,
        mask=above_boundary,
    )

    num_above = tl.sum(above_boundary.to(tl.int32), axis=0)
    needed_at_boundary = top_k - num_above
    seen_at_boundary = 0
    scan_start = 0
    while scan_start < vocab_size:
        token_ids = scan_start + tl.arange(0, SCAN_BLOCK_SIZE)
        in_bounds = token_ids < vocab_size
        values = tl.load(
            probabilities + row * vocab_size + token_ids,
            mask=in_bounds,
            other=-1.0,
        ).to(tl.float32)
        at_boundary = in_bounds & (values == boundary)
        local_positions = tl.cumsum(at_boundary.to(tl.int32), axis=0) - 1
        selected = at_boundary & (
            seen_at_boundary + local_positions < needed_at_boundary
        )
        output_positions = num_above + seen_at_boundary + local_positions
        tl.store(
            topk_probabilities + row_offset + output_positions,
            boundary,
            mask=selected,
        )
        tl.store(
            topk_token_ids + row_offset + output_positions,
            token_ids,
            mask=selected,
        )
        seen_at_boundary += tl.sum(at_boundary.to(tl.int32), axis=0)
        scan_start += SCAN_BLOCK_SIZE

    tl.debug_barrier()
    corrected_probabilities = tl.load(
        topk_probabilities + row_offset + ranks, mask=rank_mask, other=0.0
    ).to(tl.float32)
    corrected_token_ids = tl.load(
        topk_token_ids + row_offset + ranks, mask=rank_mask, other=0
    ).to(tl.int64)
    probability_bits = corrected_probabilities.to(tl.int32, bitcast=True)
    packed = (probability_bits.to(tl.int64) << 32) | (0xFFFFFFFF - corrected_token_ids)
    packed = tl.where(rank_mask, packed, -1)
    sorted_packed = tl.sort(packed, descending=True)
    sorted_token_ids = 0xFFFFFFFF - (sorted_packed & 0xFFFFFFFF)
    sorted_probabilities = tl.load(
        probabilities + row * vocab_size + sorted_token_ids,
        mask=rank_mask,
        other=0.0,
    )
    tl.store(
        topk_probabilities + row_offset + ranks,
        sorted_probabilities,
        mask=rank_mask,
    )
    tl.store(
        topk_token_ids + row_offset + ranks,
        sorted_token_ids,
        mask=rank_mask,
    )


@triton.jit
def _select_topk_watermark_token(
    topk_probabilities,
    topk_token_ids,
    context_hash,
    top_k,
    top_p,
    min_p,
    key,
    key_b,
    mixing_threshold,
    max_probability_threshold,
    row,
    candidate_count,
    BLOCK_K: tl.constexpr,
    DUAL_KEY: tl.constexpr,
    APPLY_ENTROPY_GATE: tl.constexpr,
):
    ranks = tl.arange(0, BLOCK_K)
    in_bounds = ranks < candidate_count
    offsets = row * candidate_count + ranks
    probabilities = tl.load(topk_probabilities + offsets, mask=in_bounds, other=0.0).to(
        tl.float32
    )
    token_ids = tl.load(topk_token_ids + offsets, mask=in_bounds, other=0).to(tl.uint32)
    cumulative = tl.cumsum(probabilities, axis=0)
    max_probability = tl.load(topk_probabilities + row * candidate_count).to(tl.float32)
    is_candidate = (
        in_bounds
        & (probabilities > 0.0)
        & (ranks < top_k)
        & ((cumulative - probabilities) <= top_p)
        & (probabilities >= max_probability * min_p)
    )
    below_probability_threshold = True
    if APPLY_ENTROPY_GATE:
        candidate_mass = tl.sum(tl.where(is_candidate, probabilities, 0.0), axis=0)
        below_probability_threshold = (
            max_probability / candidate_mass <= max_probability_threshold
        )

    state: tl.uint32 = 0
    state = murmur3_mix(state, (key & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, ((key >> 32) & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, context_hash.to(tl.uint32))
    state = murmur3_mix(state, token_ids)
    hashed = fmix32(state ^ 16)
    if DUAL_KEY:
        mask_state: tl.uint32 = 0
        mask_state = murmur3_mix(mask_state, (key & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, ((key >> 32) & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, (key_b & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, ((key_b >> 32) & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, context_hash.to(tl.uint32))
        use_key_a = fmix32(mask_state ^ 20).to(tl.uint64) < mixing_threshold.to(
            tl.uint64
        )
        state_b: tl.uint32 = 0
        state_b = murmur3_mix(state_b, (key_b & 0xFFFFFFFF).to(tl.uint32))
        state_b = murmur3_mix(state_b, ((key_b >> 32) & 0xFFFFFFFF).to(tl.uint32))
        state_b = murmur3_mix(state_b, context_hash.to(tl.uint32))
        state_b = murmur3_mix(state_b, token_ids)
        hashed = tl.where(use_key_a, hashed, fmix32(state_b ^ 16))
    safe_probabilities = tl.where(is_candidate, probabilities, 1.0)
    scores = tl.where(
        is_candidate,
        _log_uniform_from_hash(hashed) / safe_probabilities,
        -float("inf"),
    )
    max_score = tl.max(scores, axis=0)
    token_id = tl.min(tl.where(scores == max_score, token_ids, 0x7FFFFFFF), axis=0).to(
        tl.int32
    )
    return token_id, below_probability_threshold


@triton.jit
def _force_selected_token(
    logits,
    row,
    token_id,
    eligible,
    vocab_size: tl.constexpr,
    CLEAR_BLOCK_SIZE: tl.constexpr,
):
    clear_start = 0
    while clear_start < vocab_size:
        offsets = clear_start + tl.arange(0, CLEAR_BLOCK_SIZE)
        tl.store(
            logits + row * vocab_size + offsets,
            -float("inf"),
            mask=eligible & (offsets < vocab_size),
        )
        clear_start += CLEAR_BLOCK_SIZE
    tl.debug_barrier()
    tl.store(logits + row * vocab_size + token_id, 0.0, mask=eligible)


@triton.jit
def _watermark_force_topk_kernel(
    logits,
    topk_probabilities,
    topk_token_ids,
    context_hashes,
    eligible,
    top_ks,
    top_ps,
    min_ps,
    keys,
    keys_b,
    mixing_thresholds,
    max_probability_threshold,
    output_token_ids,
    vocab_size: tl.constexpr,
    candidate_count,
    BLOCK_K: tl.constexpr,
    CLEAR_BLOCK_SIZE: tl.constexpr,
    DUAL_KEY: tl.constexpr,
    APPLY_ENTROPY_GATE: tl.constexpr,
    ROWS_PER_CONFIG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    config_row = row // ROWS_PER_CONFIG
    row_eligible = tl.load(eligible + row)
    context_hash = tl.load(context_hashes + row).to(tl.uint32)
    key = tl.load(keys + config_row).to(tl.uint64)
    key_b = tl.load(keys_b + config_row).to(tl.uint64)
    mixing_threshold = tl.load(mixing_thresholds + config_row).to(tl.uint64)
    token_id, below_probability_threshold = _select_topk_watermark_token(
        topk_probabilities,
        topk_token_ids,
        context_hash,
        tl.load(top_ks + config_row),
        tl.load(top_ps + config_row),
        tl.load(min_ps + config_row),
        key,
        key_b,
        mixing_threshold,
        max_probability_threshold,
        row,
        candidate_count,
        BLOCK_K,
        DUAL_KEY,
        APPLY_ENTROPY_GATE,
    )
    row_eligible &= below_probability_threshold
    _force_selected_token(
        logits,
        row,
        token_id,
        row_eligible,
        vocab_size,
        CLEAR_BLOCK_SIZE,
    )
    tl.store(output_token_ids + row, tl.where(row_eligible, token_id, -1))


@triton.jit
def _watermark_partial_argmax_kernel(
    probabilities,
    context_hashes,
    keys,
    keys_b,
    mixing_thresholds,
    partial_scores,
    partial_token_ids,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DUAL_KEY: tl.constexpr,
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
    context_hash = tl.load(context_hashes + row).to(tl.uint32)
    state = murmur3_mix(state, context_hash)
    state = murmur3_mix(state, token_ids.to(tl.uint32))
    hashed = fmix32(state ^ 16)
    if DUAL_KEY:
        key_b = tl.load(keys_b + row).to(tl.uint64)
        mask_state: tl.uint32 = 0
        mask_state = murmur3_mix(mask_state, (key & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, ((key >> 32) & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, (key_b & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, ((key_b >> 32) & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, context_hash)
        use_key_a = fmix32(mask_state ^ 20).to(tl.uint64) < tl.load(
            mixing_thresholds + row
        ).to(tl.uint64)
        state_b: tl.uint32 = 0
        state_b = murmur3_mix(state_b, (key_b & 0xFFFFFFFF).to(tl.uint32))
        state_b = murmur3_mix(state_b, ((key_b >> 32) & 0xFFFFFFFF).to(tl.uint32))
        state_b = murmur3_mix(state_b, context_hash)
        state_b = murmur3_mix(state_b, token_ids.to(tl.uint32))
        hashed = tl.where(use_key_a, hashed, fmix32(state_b ^ 16))
    is_candidate = in_bounds & (candidate_probabilities > 0.0)
    safe_probabilities = tl.where(is_candidate, candidate_probabilities, 1.0)
    scores = tl.where(
        is_candidate,
        _log_uniform_from_hash(hashed) / safe_probabilities,
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
    RECORD_CONTEXT: tl.constexpr,
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
    if RECORD_CONTEXT:
        tl.store(
            watermarked_context_hashes + pool_index * max_contexts_per_req + count,
            context_hash.to(tl.int32),
            mask=eligible,
        )
        tl.store(num_watermarked_contexts + pool_index, count + 1, mask=eligible)


@triton.jit
def _watermark_force_topk_with_state_kernel(
    logits,
    topk_probabilities,
    topk_token_ids,
    token_history,
    lengths,
    write_positions,
    watermarked_context_hashes,
    num_watermarked_contexts,
    req_pool_indices,
    context_windows,
    watermark_enabled,
    top_ks,
    top_ps,
    min_ps,
    keys,
    keys_b,
    mixing_thresholds,
    max_probability_threshold,
    output_context_hashes,
    output_eligible,
    output_token_ids,
    context_window: tl.constexpr,
    max_contexts_per_req: tl.constexpr,
    vocab_size: tl.constexpr,
    candidate_count,
    BLOCK_K: tl.constexpr,
    HISTORY_BLOCK_SIZE: tl.constexpr,
    CLEAR_BLOCK_SIZE: tl.constexpr,
    DUAL_KEY: tl.constexpr,
    APPLY_ENTROPY_GATE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
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
        token = tl.load(token_history + pool_index * context_window + ring_index).to(
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
    history_offset = 0
    while (history_offset < count) & eligible:
        positions = history_offset + tl.arange(0, HISTORY_BLOCK_SIZE)
        mask = positions < count
        previous_hashes = tl.load(
            watermarked_context_hashes + pool_index * max_contexts_per_req + positions,
            mask=mask,
            other=0,
        )
        repeated |= (
            tl.sum((previous_hashes == context_hash.to(tl.int32)) & mask, axis=0) > 0
        )
        history_offset += HISTORY_BLOCK_SIZE
    eligible &= ~repeated

    key = tl.load(keys + row).to(tl.uint64)
    key_b = tl.load(keys_b + row).to(tl.uint64)
    mixing_threshold = tl.load(mixing_thresholds + row).to(tl.uint64)
    token_id, below_probability_threshold = _select_topk_watermark_token(
        topk_probabilities,
        topk_token_ids,
        context_hash,
        tl.load(top_ks + row),
        tl.load(top_ps + row),
        tl.load(min_ps + row),
        key,
        key_b,
        mixing_threshold,
        max_probability_threshold,
        row,
        candidate_count,
        BLOCK_K,
        DUAL_KEY,
        APPLY_ENTROPY_GATE,
    )
    eligible &= below_probability_threshold
    _force_selected_token(
        logits,
        row,
        token_id,
        eligible,
        vocab_size,
        CLEAR_BLOCK_SIZE,
    )
    tl.store(output_context_hashes + row, context_hash.to(tl.int64))
    tl.store(output_eligible + row, eligible)
    tl.store(output_token_ids + row, tl.where(eligible, token_id, -1))
    tl.store(
        watermarked_context_hashes + pool_index * max_contexts_per_req + count,
        context_hash.to(tl.int32),
        mask=eligible,
    )
    tl.store(num_watermarked_contexts + pool_index, count + 1, mask=eligible)


@triton.jit
def _append_watermark_tokens_kernel(
    token_history,
    lengths,
    write_positions,
    req_pool_indices,
    next_token_ids,
    context_window: tl.constexpr,
):
    row = tl.program_id(0)
    pool_index = tl.load(req_pool_indices + row).to(tl.int64)
    write_position = tl.load(write_positions + pool_index)
    next_token_id = tl.load(next_token_ids + row)
    tl.store(
        token_history + pool_index * context_window + write_position,
        next_token_id.to(tl.int32),
    )
    tl.store(
        write_positions + pool_index,
        (write_position + 1) % context_window,
    )
    length = tl.load(lengths + pool_index)
    tl.store(lengths + pool_index, tl.minimum(length + 1, context_window))


@triton.jit
def _speculative_context_hash_kernel(
    token_history,
    lengths,
    write_positions,
    req_pool_indices,
    draft_tokens,
    custom_mask,
    positions,
    context_windows,
    output_context_hashes,
    output_context_lengths,
    context_window: tl.constexpr,
    draft_token_num: tl.constexpr,
    FULL_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    request_row = row // draft_token_num
    draft_position = row % draft_token_num
    pool_index = tl.load(req_pool_indices + request_row).to(tl.int64)
    length = tl.load(lengths + pool_index)
    write_position = tl.load(write_positions + pool_index)
    requested_window = tl.load(context_windows + request_row)
    base_length = tl.minimum(length, requested_window)
    start = tl.where(length == context_window, write_position, 0)
    source_start = length - base_length

    if FULL_MASK:
        prefix_length = tl.load(positions + request_row * draft_token_num).to(tl.int64)
        request_offset = tl.full((), 0, tl.int64)
        prior_request = 0
        while prior_request < request_row:
            prior_prefix_length = tl.load(
                positions + prior_request * draft_token_num
            ).to(tl.int64)
            request_offset += draft_token_num * (prior_prefix_length + draft_token_num)
            prior_request += 1
        mask_row_offset = (
            request_offset
            + draft_position * (prefix_length + draft_token_num)
            + prefix_length
        )
    else:
        mask_row_offset = row * draft_token_num

    ancestor_count = 0
    for candidate in range(1, draft_token_num):
        ancestor_count += tl.load(custom_mask + mask_row_offset + candidate).to(
            tl.int32
        )
    total_length = base_length + ancestor_count
    context_length = tl.minimum(total_length, requested_window)
    skip = total_length - context_length

    state = tl.full((), 0, tl.uint32)
    rank = 0
    for candidate in range(context_window):
        valid = candidate < base_length
        ring_index = (start + source_start + candidate) % context_window
        token = tl.load(token_history + pool_index * context_window + ring_index).to(
            tl.uint32
        )
        mixed = murmur3_mix(state, token).to(tl.uint32)
        state = tl.where(valid & (rank >= skip), mixed, state).to(tl.uint32)
        rank += valid
    for candidate in range(1, draft_token_num):
        valid = tl.load(custom_mask + mask_row_offset + candidate)
        token = tl.load(draft_tokens + request_row * draft_token_num + candidate).to(
            tl.uint32
        )
        mixed = murmur3_mix(state, token).to(tl.uint32)
        state = tl.where(valid & (rank >= skip), mixed, state).to(tl.uint32)
        rank += valid

    context_hash = fmix32(state ^ (context_length * 4).to(tl.uint32))
    tl.store(output_context_hashes + row, context_hash.to(tl.int64))
    tl.store(output_context_lengths + row, context_length)


@triton.jit
def _speculative_context_eligible_kernel(
    watermarked_context_hashes,
    num_watermarked_contexts,
    req_pool_indices,
    context_hashes,
    context_lengths,
    watermark_enabled,
    top_ks,
    output_eligible,
    draft_token_num: tl.constexpr,
    max_contexts_per_req: tl.constexpr,
    HISTORY_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    request_row = row // draft_token_num
    draft_position = row % draft_token_num
    pool_index = tl.load(req_pool_indices + request_row).to(tl.int64)
    context_hash = tl.load(context_hashes + row).to(tl.int32)
    count = tl.load(num_watermarked_contexts + pool_index)
    eligible = (
        tl.load(watermark_enabled + request_row)
        & (tl.load(top_ks + request_row) > 1)
        & (tl.load(context_lengths + row) > 0)
        & (count < max_contexts_per_req)
    )

    repeated = False
    history_offset = 0
    while (history_offset < count) & eligible:
        history_positions = history_offset + tl.arange(0, HISTORY_BLOCK_SIZE)
        history_mask = history_positions < count
        previous_hashes = tl.load(
            watermarked_context_hashes
            + pool_index * max_contexts_per_req
            + history_positions,
            mask=history_mask,
            other=0,
        )
        repeated |= tl.sum((previous_hashes == context_hash) & history_mask, axis=0) > 0
        history_offset += HISTORY_BLOCK_SIZE
    for prior_position in range(draft_token_num):
        prior_hash = tl.load(
            context_hashes + request_row * draft_token_num + prior_position
        ).to(tl.int32)
        repeated |= (prior_position < draft_position) & (prior_hash == context_hash)
    tl.store(output_eligible + row, eligible & ~repeated)


@triton.jit
def _record_speculative_contexts_kernel(
    watermarked_context_hashes,
    num_watermarked_contexts,
    req_pool_indices,
    context_hashes,
    selected,
    accept_indices,
    accept_lens,
    max_contexts_per_req: tl.constexpr,
    max_accept_tokens: tl.constexpr,
):
    request_row = tl.program_id(0)
    pool_index = tl.load(req_pool_indices + request_row).to(tl.int64)
    count = tl.load(num_watermarked_contexts + pool_index)
    accept_length = tl.load(accept_lens + request_row)
    for position in range(max_accept_tokens):
        row = tl.load(accept_indices + request_row * max_accept_tokens + position).to(
            tl.int64
        )
        row = tl.maximum(row, 0)
        should_record = (
            (position < accept_length)
            & tl.load(selected + row)
            & (count < max_contexts_per_req)
        )
        context_hash = tl.load(context_hashes + row).to(tl.int32)
        tl.store(
            watermarked_context_hashes + pool_index * max_contexts_per_req + count,
            context_hash,
            mask=should_record,
        )
        count += should_record
    tl.store(num_watermarked_contexts + pool_index, count)


@triton.jit
def _append_speculative_watermark_tokens_kernel(
    token_history,
    lengths,
    write_positions,
    req_pool_indices,
    accept_tokens,
    accept_lens,
    context_window: tl.constexpr,
    max_accept_tokens: tl.constexpr,
):
    request_row = tl.program_id(0)
    pool_index = tl.load(req_pool_indices + request_row).to(tl.int64)
    write_position = tl.load(write_positions + pool_index)
    length = tl.load(lengths + pool_index)
    accept_length = tl.load(accept_lens + request_row)
    for position in range(max_accept_tokens):
        valid = position < accept_length
        token = tl.load(accept_tokens + request_row * max_accept_tokens + position)
        tl.store(
            token_history + pool_index * context_window + write_position,
            token.to(tl.int32),
            mask=valid,
        )
        write_position = tl.where(
            valid, (write_position + 1) % context_window, write_position
        )
        length = tl.where(valid, tl.minimum(length + 1, context_window), length)
    tl.store(write_positions + pool_index, write_position)
    tl.store(lengths + pool_index, length)


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
    keys_b,
    mixing_thresholds,
    partial_scores,
    partial_token_ids,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DUAL_KEY: tl.constexpr,
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
    context_hash = tl.load(context_hashes + row).to(tl.uint32)
    state = murmur3_mix(state, context_hash)
    state = murmur3_mix(state, token_ids)
    hashed = fmix32(state ^ 16)
    if DUAL_KEY:
        key_b = tl.load(keys_b + row).to(tl.uint64)
        mask_state: tl.uint32 = 0
        mask_state = murmur3_mix(mask_state, (key & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, ((key >> 32) & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, (key_b & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, ((key_b >> 32) & 0xFFFFFFFF).to(tl.uint32))
        mask_state = murmur3_mix(mask_state, context_hash)
        use_key_a = fmix32(mask_state ^ 20).to(tl.uint64) < tl.load(
            mixing_thresholds + row
        ).to(tl.uint64)
        state_b: tl.uint32 = 0
        state_b = murmur3_mix(state_b, (key_b & 0xFFFFFFFF).to(tl.uint32))
        state_b = murmur3_mix(state_b, ((key_b >> 32) & 0xFFFFFFFF).to(tl.uint32))
        state_b = murmur3_mix(state_b, context_hash)
        state_b = murmur3_mix(state_b, token_ids)
        hashed = tl.where(use_key_a, hashed, fmix32(state_b ^ 16))
    safe_probabilities = tl.where(is_candidate, probabilities, 1.0)
    scores = tl.where(
        is_candidate,
        _log_uniform_from_hash(hashed) / safe_probabilities,
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
    keys_b: torch.Tensor | None = None,
    mixing_thresholds: torch.Tensor | None = None,
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
    if (keys_b is None) != (mixing_thresholds is None):
        raise ValueError(
            "dual-key watermark selection requires both key B and mixing thresholds"
        )
    dual_key = keys_b is not None
    if dual_key and (
        keys_b.shape != (batch_size,)
        or keys_b.dtype != torch.int64
        or mixing_thresholds.shape != (batch_size,)
        or mixing_thresholds.dtype != torch.int64
    ):
        raise ValueError("dual-key inputs must be int64 with one value per row")
    keys_b = keys if keys_b is None else keys_b
    mixing_thresholds = (
        context_hashes if mixing_thresholds is None else mixing_thresholds
    )
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (probabilities, context_hashes, keys, keys_b, mixing_thresholds)
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
        keys_b,
        mixing_thresholds,
        partial_scores,
        partial_token_ids,
        vocab_size=vocab_size,
        num_splits=num_splits,
        BLOCK_SIZE=_BLOCK_SIZE,
        DUAL_KEY=dual_key,
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
    *,
    record_context: bool = True,
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
        RECORD_CONTEXT=record_context,
        num_warps=8,
    )


def append_watermark_tokens_triton(
    token_history: torch.Tensor,
    lengths: torch.Tensor,
    write_positions: torch.Tensor,
    req_pool_indices: torch.Tensor,
    next_token_ids: torch.Tensor,
) -> None:
    batch_size = req_pool_indices.shape[0]
    if batch_size == 0:
        return
    _append_watermark_tokens_kernel[(batch_size,)](
        token_history,
        lengths,
        write_positions,
        req_pool_indices,
        next_token_ids,
        context_window=token_history.shape[1],
        num_warps=1,
    )


def prepare_speculative_watermark_contexts_triton(
    token_history: torch.Tensor,
    lengths: torch.Tensor,
    write_positions: torch.Tensor,
    watermarked_context_hashes: torch.Tensor,
    num_watermarked_contexts: torch.Tensor,
    req_pool_indices: torch.Tensor,
    draft_tokens: torch.Tensor,
    custom_mask: torch.Tensor,
    positions: torch.Tensor,
    context_windows: torch.Tensor,
    watermark_enabled: torch.Tensor,
    top_ks: torch.Tensor,
    output_context_hashes: torch.Tensor,
    output_context_lengths: torch.Tensor,
    output_eligible: torch.Tensor,
    draft_token_num: int,
    full_mask: bool,
) -> None:
    batch_size = req_pool_indices.shape[0]
    num_rows = batch_size * draft_token_num
    if num_rows == 0:
        return
    _speculative_context_hash_kernel[(num_rows,)](
        token_history,
        lengths,
        write_positions,
        req_pool_indices,
        draft_tokens,
        custom_mask,
        positions,
        context_windows,
        output_context_hashes,
        output_context_lengths,
        context_window=token_history.shape[1],
        draft_token_num=draft_token_num,
        FULL_MASK=full_mask,
        num_warps=1,
    )
    _speculative_context_eligible_kernel[(num_rows,)](
        watermarked_context_hashes,
        num_watermarked_contexts,
        req_pool_indices,
        output_context_hashes,
        output_context_lengths,
        watermark_enabled,
        top_ks,
        output_eligible,
        draft_token_num=draft_token_num,
        max_contexts_per_req=watermarked_context_hashes.shape[1],
        HISTORY_BLOCK_SIZE=_HISTORY_BLOCK_SIZE,
        num_warps=8,
    )


def record_speculative_watermark_contexts_triton(
    watermarked_context_hashes: torch.Tensor,
    num_watermarked_contexts: torch.Tensor,
    req_pool_indices: torch.Tensor,
    context_hashes: torch.Tensor,
    selected: torch.Tensor,
    accept_indices: torch.Tensor,
    accept_lens: torch.Tensor,
) -> None:
    batch_size = req_pool_indices.shape[0]
    if batch_size == 0:
        return
    _record_speculative_contexts_kernel[(batch_size,)](
        watermarked_context_hashes,
        num_watermarked_contexts,
        req_pool_indices,
        context_hashes,
        selected,
        accept_indices,
        accept_lens,
        max_contexts_per_req=watermarked_context_hashes.shape[1],
        max_accept_tokens=accept_indices.shape[1],
        num_warps=1,
    )


def append_speculative_watermark_tokens_triton(
    token_history: torch.Tensor,
    lengths: torch.Tensor,
    write_positions: torch.Tensor,
    req_pool_indices: torch.Tensor,
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
) -> None:
    batch_size = req_pool_indices.shape[0]
    if batch_size == 0:
        return
    _append_speculative_watermark_tokens_kernel[(batch_size,)](
        token_history,
        lengths,
        write_positions,
        req_pool_indices,
        accept_tokens,
        accept_lens,
        context_window=token_history.shape[1],
        max_accept_tokens=accept_tokens.shape[1],
        num_warps=1,
    )


def _topk_probabilities(
    probabilities: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_probabilities, topk_token_ids = torch.topk(
        probabilities, top_k, dim=-1, largest=True, sorted=True
    )
    _canonicalize_topk_kernel[(probabilities.shape[0],)](
        probabilities,
        topk_probabilities,
        topk_token_ids,
        vocab_size=probabilities.shape[1],
        top_k=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        SCAN_BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return topk_probabilities, topk_token_ids


def force_speculative_watermark_tokens_triton(
    logits: torch.Tensor,
    context_hashes: torch.Tensor,
    eligible: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    keys: torch.Tensor,
    keys_b: torch.Tensor | None,
    mixing_thresholds: torch.Tensor | None,
    draft_token_num: int,
    max_top_k: int,
    output_token_ids: torch.Tensor,
    max_probability: float = 1.0,
) -> torch.Tensor:
    batch_size = temperatures.shape[0]
    num_rows, vocab_size = logits.shape
    if num_rows != batch_size * draft_token_num:
        raise ValueError("speculative watermark rows do not match the batch layout")
    if not can_use_finite_topk_watermark(max_top_k, vocab_size):
        raise ValueError("speculative watermark fast path requires finite top-k")
    if (keys_b is None) != (mixing_thresholds is None):
        raise ValueError(
            "dual-key watermark selection requires both key B and mixing thresholds"
        )
    dual_key = keys_b is not None
    keys_b = keys if keys_b is None else keys_b
    mixing_thresholds = (
        context_hashes[:batch_size] if mixing_thresholds is None else mixing_thresholds
    )
    output_token_ids = output_token_ids[:num_rows]
    if num_rows == 0:
        return output_token_ids
    probabilities = torch.softmax(
        logits.view(batch_size, draft_token_num, vocab_size)
        / temperatures.view(batch_size, 1, 1),
        dim=-1,
    ).view(num_rows, vocab_size)
    topk_probabilities, topk_token_ids = _topk_probabilities(probabilities, max_top_k)
    _watermark_force_topk_kernel[(num_rows,)](
        logits,
        topk_probabilities,
        topk_token_ids,
        context_hashes,
        eligible,
        top_ks,
        top_ps,
        min_ps,
        keys,
        keys_b,
        mixing_thresholds,
        max_probability,
        output_token_ids,
        vocab_size=vocab_size,
        candidate_count=max_top_k,
        BLOCK_K=triton.next_power_of_2(max_top_k),
        CLEAR_BLOCK_SIZE=_CLEAR_BLOCK_SIZE,
        DUAL_KEY=dual_key,
        APPLY_ENTROPY_GATE=max_probability < 1.0,
        ROWS_PER_CONFIG=draft_token_num,
        num_warps=8,
    )
    return output_token_ids


def force_watermark_tokens_with_state_triton(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    lengths: torch.Tensor,
    write_positions: torch.Tensor,
    watermarked_context_hashes: torch.Tensor,
    num_watermarked_contexts: torch.Tensor,
    req_pool_indices: torch.Tensor,
    context_windows: torch.Tensor,
    watermark_enabled: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    keys: torch.Tensor,
    keys_b: torch.Tensor | None,
    mixing_thresholds: torch.Tensor | None,
    output_context_hashes: torch.Tensor,
    output_eligible: torch.Tensor,
    output_token_ids: torch.Tensor,
    max_top_k: int,
    max_probability: float = 1.0,
) -> None:
    batch_size, vocab_size = logits.shape
    if batch_size == 0:
        return
    if not can_use_finite_topk_watermark(max_top_k, vocab_size):
        raise ValueError("finite top-k watermark forcing requires 1 < top_k < vocab")
    if (keys_b is None) != (mixing_thresholds is None):
        raise ValueError(
            "dual-key watermark selection requires both key B and mixing thresholds"
        )
    dual_key = keys_b is not None
    keys_b = keys if keys_b is None else keys_b
    mixing_thresholds = (
        output_context_hashes if mixing_thresholds is None else mixing_thresholds
    )
    probabilities = torch.softmax(logits / temperatures, dim=-1)
    topk_probabilities, topk_token_ids = _topk_probabilities(probabilities, max_top_k)
    _watermark_force_topk_with_state_kernel[(batch_size,)](
        logits,
        topk_probabilities,
        topk_token_ids,
        token_history,
        lengths,
        write_positions,
        watermarked_context_hashes,
        num_watermarked_contexts,
        req_pool_indices,
        context_windows,
        watermark_enabled,
        top_ks,
        top_ps,
        min_ps,
        keys,
        keys_b,
        mixing_thresholds,
        max_probability,
        output_context_hashes,
        output_eligible,
        output_token_ids,
        context_window=token_history.shape[1],
        max_contexts_per_req=watermarked_context_hashes.shape[1],
        vocab_size=vocab_size,
        candidate_count=max_top_k,
        BLOCK_K=triton.next_power_of_2(max_top_k),
        HISTORY_BLOCK_SIZE=_HISTORY_BLOCK_SIZE,
        CLEAR_BLOCK_SIZE=_CLEAR_BLOCK_SIZE,
        DUAL_KEY=dual_key,
        APPLY_ENTROPY_GATE=max_probability < 1.0,
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
    keys_b: torch.Tensor | None = None,
    mixing_thresholds: torch.Tensor | None = None,
    *,
    max_top_k: int | None = None,
    partial_scores: torch.Tensor | None = None,
    partial_token_ids: torch.Tensor | None = None,
    output_token_ids: torch.Tensor | None = None,
    max_probability: float = 1.0,
) -> torch.Tensor:
    if (keys_b is None) != (mixing_thresholds is None):
        raise ValueError(
            "dual-key watermark selection requires both key B and mixing thresholds"
        )
    dual_key = keys_b is not None
    keys_b = keys if keys_b is None else keys_b
    mixing_thresholds = (
        context_hashes if mixing_thresholds is None else mixing_thresholds
    )
    batch_size, vocab_size = logits.shape
    if output_token_ids is None:
        output_token_ids = torch.empty(
            batch_size, dtype=torch.int32, device=logits.device
        )
    elif output_token_ids.numel() < batch_size:
        raise ValueError("output token buffer is smaller than the watermark batch")
    else:
        output_token_ids = output_token_ids[:batch_size]
    if batch_size == 0:
        return output_token_ids

    probabilities = torch.softmax(logits / temperatures, dim=-1)
    if can_use_finite_topk_watermark(max_top_k, vocab_size):
        topk_probabilities, topk_token_ids = _topk_probabilities(
            probabilities, max_top_k
        )
        _watermark_force_topk_kernel[(batch_size,)](
            logits,
            topk_probabilities,
            topk_token_ids,
            context_hashes,
            eligible,
            top_ks,
            top_ps,
            min_ps,
            keys,
            keys_b,
            mixing_thresholds,
            max_probability,
            output_token_ids,
            vocab_size=vocab_size,
            candidate_count=max_top_k,
            BLOCK_K=triton.next_power_of_2(max_top_k),
            CLEAR_BLOCK_SIZE=_CLEAR_BLOCK_SIZE,
            DUAL_KEY=dual_key,
            APPLY_ENTROPY_GATE=max_probability < 1.0,
            ROWS_PER_CONFIG=1,
            num_warps=8,
        )
        return output_token_ids

    sorted_probabilities, sorted_token_ids = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    if max_probability < 1.0:
        ranks = torch.arange(vocab_size, device=logits.device).view(1, -1)
        keep = ranks < top_ks.view(-1, 1)
        keep &= (cumulative_probabilities - sorted_probabilities) <= top_ps.view(-1, 1)
        keep &= sorted_probabilities >= (
            sorted_probabilities[:, :1] * min_ps.view(-1, 1)
        )
        candidate_mass = torch.where(keep, sorted_probabilities, 0.0).sum(dim=-1)
        eligible &= sorted_probabilities[:, 0] / candidate_mass <= max_probability

    num_splits = watermark_selector_num_splits(vocab_size)
    partial_size = batch_size * num_splits
    if partial_scores is None:
        partial_scores = torch.empty(
            partial_size, dtype=torch.float32, device=logits.device
        )
    elif partial_scores.numel() < partial_size:
        raise ValueError("partial score buffer is smaller than the watermark grid")
    else:
        partial_scores = partial_scores[:partial_size]
    if partial_token_ids is None:
        partial_token_ids = torch.empty(
            partial_size, dtype=torch.int32, device=logits.device
        )
    elif partial_token_ids.numel() < partial_size:
        raise ValueError("partial token buffer is smaller than the watermark grid")
    else:
        partial_token_ids = partial_token_ids[:partial_size]
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
        keys_b,
        mixing_thresholds,
        partial_scores,
        partial_token_ids,
        vocab_size=vocab_size,
        num_splits=num_splits,
        BLOCK_SIZE=_BLOCK_SIZE,
        DUAL_KEY=dual_key,
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
