# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
# Modified by the SGLang Team: adapted from google-deepmind/synthid-text@addb4a15
# (src/synthid_text/logits_processing.py and src/synthid_text/hashing_function.py).

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.sampling.synthid import (
    G_VALUE_SHIFT,
    HASH_INCREMENT,
    HASH_MULTIPLIER,
    HASH_ROUNDS,
    HASH_SHIFT,
    MAX_BLOCK_CANDIDATES,
    MAX_CHAIN_WIDTH,
    MAX_CONTEXT_HISTORY_SIZE,
    MAX_CONTEXT_WIDTH,
    MAX_KEYS,
)

_HASH_MULTIPLIER = tl.constexpr(HASH_MULTIPLIER)
_HASH_INCREMENT = tl.constexpr(HASH_INCREMENT)
_HASH_ROUNDS = tl.constexpr(HASH_ROUNDS)
_HASH_SHIFT = tl.constexpr(HASH_SHIFT)
_G_VALUE_SHIFT = tl.constexpr(G_VALUE_SHIFT)
_FLT_MIN = tl.constexpr(1.1754943508222875e-38)


@triton.jit
def _accumulate_hash(value, data):
    return (value + data) * _HASH_MULTIPLIER + _HASH_INCREMENT


@triton.jit
def _g_value(candidate_hashes, key):
    hashes = _accumulate_hash(candidate_hashes, key)
    for _ in range(_HASH_ROUNDS):
        hashes = _accumulate_hash(hashes, 1) >> _HASH_SHIFT
    return (hashes >> _G_VALUE_SHIFT) & 1


@triton.jit
def _eligible(
    call_number,
    context_width: tl.constexpr,
    skip_first_ngram_calls: tl.constexpr,
):
    """Whether the ``call_number``-th generated token (from 1) may be watermarked."""
    if skip_first_ngram_calls:
        return call_number > context_width
    return call_number > 0


@triton.jit
def _in_history(history_values, history_mask, context_hash):
    return (
        tl.max(tl.where(history_mask & (history_values == context_hash), 1, 0), axis=0)
        != 0
    )


@triton.jit
def _projected_context_hash(
    context_ptr,
    verify_tokens_ptr,
    slot,
    request,
    step,
    context_stride,
    verify_stride,
    context_width: tl.constexpr,
    hash_iv,
):
    """The context hash of chain position ``step``: the slot's committed context
    followed by the request's draft tokens ``1..step``."""
    context_hash = hash_iv.to(tl.int64)
    for context_index in range(context_width):
        base_index = context_index + step
        draft_index = 1 + context_index + step - context_width
        from_base = base_index < context_width
        base_token = tl.load(
            context_ptr + slot * context_stride + base_index,
            mask=from_base,
            other=0,
        )
        draft_token = tl.load(
            verify_tokens_ptr + request * verify_stride + draft_index,
            mask=~from_base,
            other=0,
        )
        context_hash = _accumulate_hash(
            context_hash,
            tl.where(from_base, base_token, draft_token).to(tl.int64),
        )
    return context_hash


@triton.jit
def _synthid_tournament_kernel(
    scores_ptr,
    token_ids_ptr,
    context_hashes_ptr,
    keys_ptr,
    apply_mask_ptr,
    output_ptr,
    score_stride,
    token_stride,
    output_stride,
    num_candidates: tl.constexpr,
    depth: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_candidates

    scores = tl.load(
        scores_ptr + row * score_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    max_score = tl.max(scores, axis=0)
    probs = tl.exp(scores - max_score)
    probs = tl.where(mask, probs, 0.0)
    probs /= tl.sum(probs, axis=0)

    token_ids = tl.load(
        token_ids_ptr + row * token_stride + offsets, mask=mask, other=0
    ).to(tl.int64)
    context_hash = tl.load(context_hashes_ptr + row).to(tl.int64)
    candidate_hashes = _accumulate_hash(context_hash, token_ids)

    for key_index in range(depth):
        key = tl.load(keys_ptr + key_index).to(tl.int64)
        g_values = _g_value(candidate_hashes, key).to(tl.float32)
        g_values = tl.where(mask, g_values, 0.0)
        g_mass = tl.sum(g_values * probs, axis=0)
        probs *= 1.0 + g_values - g_mass

    should_apply = tl.load(apply_mask_ptr + row)
    output = tl.log(probs)
    output = tl.where(probs > 0.0, output, -1.0e12)
    output = tl.where(should_apply, output, scores)
    tl.store(output_ptr + row * output_stride + offsets, output, mask=mask)


@triton.jit
def _synthid_wide_tournament_kernel(
    scores_ptr,
    token_ids_ptr,
    context_hashes_ptr,
    keys_ptr,
    apply_mask_ptr,
    output_ptr,
    g_bits_ptr,
    score_stride,
    token_stride,
    output_stride,
    g_bits_stride,
    num_candidates,
    depth,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    score_base = scores_ptr + row * score_stride
    token_base = token_ids_ptr + row * token_stride
    output_base = output_ptr + row * output_stride
    g_bits_base = g_bits_ptr + row * g_bits_stride

    max_score = -float("inf")
    for start in tl.range(0, num_candidates, BLOCK_SIZE):
        mask = start + offsets < num_candidates
        scores = tl.load(score_base + start + offsets, mask=mask, other=-float("inf"))
        max_score = tl.maximum(max_score, tl.max(scores, axis=0))

    should_apply = tl.load(apply_mask_ptr + row)
    context_hash = tl.load(context_hashes_ptr + row).to(tl.int64)
    denominator = 0.0
    for start in tl.range(0, num_candidates, BLOCK_SIZE):
        candidate_offsets = start + offsets
        mask = candidate_offsets < num_candidates
        scores = tl.load(
            score_base + candidate_offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        exp_scores = tl.where(mask, tl.exp(scores - max_score), 0.0)
        denominator += tl.sum(exp_scores, axis=0)
        tl.store(output_base + candidate_offsets, exp_scores, mask=mask)

        token_ids = tl.load(
            token_base + candidate_offsets,
            mask=mask,
            other=0,
        ).to(tl.int64)
        candidate_hashes = _accumulate_hash(context_hash, token_ids)
        g_bits = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        for key_index in tl.range(0, depth):
            key = tl.load(keys_ptr + key_index).to(tl.int64)
            g_bits |= _g_value(candidate_hashes, key) << key_index
        g_bits = tl.where(should_apply, g_bits, 0)
        tl.store(g_bits_base + candidate_offsets, g_bits, mask=mask)

    denominator = tl.maximum(denominator, _FLT_MIN)
    current_mass = 0.0
    for start in tl.range(0, num_candidates, BLOCK_SIZE):
        candidate_offsets = start + offsets
        mask = candidate_offsets < num_candidates
        probs = tl.load(output_base + candidate_offsets, mask=mask, other=0.0)
        probs /= denominator
        g_bits = tl.load(g_bits_base + candidate_offsets, mask=mask, other=0)
        current_mass += tl.sum(probs * (g_bits & 1).to(tl.float32), axis=0)
        tl.store(output_base + candidate_offsets, probs, mask=mask)

    for key_index in tl.range(0, depth):
        next_mass = 0.0
        for start in tl.range(0, num_candidates, BLOCK_SIZE):
            candidate_offsets = start + offsets
            mask = candidate_offsets < num_candidates
            probs = tl.load(output_base + candidate_offsets, mask=mask, other=0.0)
            g_bits = tl.load(g_bits_base + candidate_offsets, mask=mask, other=0)
            g_value = ((g_bits >> key_index) & 1).to(tl.float32)
            probs *= 1.0 + g_value - current_mass
            if key_index + 1 < depth:
                next_g_value = ((g_bits >> (key_index + 1)) & 1).to(tl.float32)
                next_mass += tl.sum(probs * next_g_value, axis=0)
                tl.store(output_base + candidate_offsets, probs, mask=mask)
            else:
                output = tl.where(probs > 0.0, tl.log(probs), -1.0e12)
                original_scores = tl.load(
                    score_base + candidate_offsets,
                    mask=mask,
                    other=-float("inf"),
                )
                output = tl.where(should_apply, output, original_scores)
                tl.store(output_base + candidate_offsets, output, mask=mask)
        current_mass = next_mass


@triton.jit
def _synthid_sample_kernel(
    scores_ptr,
    token_ids_ptr,
    exponential_noise_ptr,
    temperatures_ptr,
    slot_indices_ptr,
    context_ptr,
    context_history_ptr,
    num_calls_ptr,
    keys_ptr,
    output_token_ids_ptr,
    score_stride,
    token_stride,
    noise_stride,
    temperature_stride,
    context_stride,
    history_stride,
    hash_iv,
    num_candidates: tl.constexpr,
    context_width: tl.constexpr,
    history_size: tl.constexpr,
    depth: tl.constexpr,
    skip_first_ngram_calls: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    HISTORY_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_indices_ptr + row).to(tl.int64)
    candidate_offsets = tl.arange(0, CANDIDATE_BLOCK_SIZE)
    candidate_mask = candidate_offsets < num_candidates

    scores = tl.load(
        scores_ptr + row * score_stride + candidate_offsets,
        mask=candidate_mask,
        other=-float("inf"),
    ).to(tl.float32)
    temperature = tl.load(temperatures_ptr + row * temperature_stride).to(tl.float32)
    scores /= temperature

    valid_mask = candidate_mask & (scores > -float("inf"))
    max_score = tl.max(scores, axis=0)
    base_probs = tl.exp(scores - max_score)
    base_probs = tl.where(valid_mask, base_probs, 0.0)
    base_probs /= tl.sum(base_probs, axis=0)
    probs = base_probs

    # Position 0 of a chain reads only the committed context.
    context_hash = _projected_context_hash(
        context_ptr,
        context_ptr,
        slot,
        slot,
        0,
        context_stride=context_stride,
        verify_stride=context_stride,
        context_width=context_width,
        hash_iv=hash_iv,
    )
    call_number = tl.load(num_calls_ptr + slot).to(tl.int64) + 1
    eligible = _eligible(call_number, context_width, skip_first_ngram_calls)

    history_offsets = tl.arange(0, HISTORY_BLOCK_SIZE)
    history_mask = history_offsets < history_size
    history_values = tl.load(
        context_history_ptr + slot * history_stride + history_offsets,
        mask=history_mask,
        other=0,
    ).to(tl.int64)
    repeated = _in_history(history_values, history_mask, context_hash)

    token_ids = tl.load(
        token_ids_ptr + row * token_stride + candidate_offsets,
        mask=candidate_mask,
        other=0,
    ).to(tl.int64)
    candidate_hashes = _accumulate_hash(context_hash, token_ids)
    for key_index in range(depth):
        key = tl.load(keys_ptr + key_index).to(tl.int64)
        g_values = _g_value(candidate_hashes, key).to(tl.float32)
        g_values = tl.where(valid_mask, g_values, 0.0)
        g_mass = tl.sum(g_values * probs, axis=0)
        probs *= 1.0 + g_values - g_mass

    selection_probs = tl.where(eligible & ~repeated, probs, base_probs)
    noise = tl.load(
        exponential_noise_ptr + row * noise_stride + candidate_offsets,
        mask=valid_mask,
        other=1.0,
    ).to(tl.float32)
    selection_scores = selection_probs / tl.maximum(noise, _FLT_MIN)
    selection_scores = tl.where(
        valid_mask,
        selection_scores,
        -float("inf"),
    )
    selected_index = tl.argmax(selection_scores, axis=0)
    selected_token_id = tl.sum(
        tl.where(candidate_offsets == selected_index, token_ids, 0),
        axis=0,
    )
    tl.store(output_token_ids_ptr + row, selected_token_id)


@triton.jit
def _synthid_project_state_kernel(
    slot_indices_ptr,
    verify_tokens_ptr,
    context_ptr,
    context_history_ptr,
    history_cursor_ptr,
    num_calls_ptr,
    context_hashes_ptr,
    apply_mask_ptr,
    verify_stride,
    context_stride,
    history_stride,
    hash_iv,
    verify_width: tl.constexpr,
    context_width: tl.constexpr,
    history_size: tl.constexpr,
    skip_first_ngram_calls: tl.constexpr,
    HISTORY_BLOCK_SIZE: tl.constexpr,
):
    flat_row = tl.program_id(0).to(tl.int64)
    request = flat_row // verify_width
    step = flat_row % verify_width
    slot = tl.load(slot_indices_ptr + request).to(tl.int64)
    context_hash = _projected_context_hash(
        context_ptr,
        verify_tokens_ptr,
        slot,
        request,
        step,
        context_stride=context_stride,
        verify_stride=verify_stride,
        context_width=context_width,
        hash_iv=hash_iv,
    )

    base_num_calls = tl.load(num_calls_ptr + slot).to(tl.int64)
    eligible = _eligible(
        base_num_calls + step + 1, context_width, skip_first_ngram_calls
    )

    history_offsets = tl.arange(0, HISTORY_BLOCK_SIZE)
    history_mask = history_offsets < history_size
    history_values = tl.load(
        context_history_ptr + slot * history_stride + history_offsets,
        mask=history_mask,
        other=0,
    ).to(tl.int64)
    projected_cursor = tl.load(history_cursor_ptr + slot).to(tl.int64)
    for previous_step in range(verify_width):
        previous_eligible = (previous_step < step) & _eligible(
            base_num_calls + previous_step + 1, context_width, skip_first_ngram_calls
        )
        previous_history_hash = _projected_context_hash(
            context_ptr,
            verify_tokens_ptr,
            slot,
            request,
            previous_step,
            context_stride=context_stride,
            verify_stride=verify_stride,
            context_width=context_width,
            hash_iv=hash_iv,
        )
        history_values = tl.where(
            history_mask & previous_eligible & (history_offsets == projected_cursor),
            previous_history_hash,
            history_values,
        )
        projected_cursor = tl.where(
            previous_eligible,
            (projected_cursor + 1) % history_size,
            projected_cursor,
        )

    repeated = _in_history(history_values, history_mask, context_hash)
    tl.store(context_hashes_ptr + flat_row, context_hash)
    tl.store(apply_mask_ptr + flat_row, eligible & ~repeated)


@triton.jit
def _synthid_commit_state_kernel(
    token_ids_ptr,
    lengths_ptr,
    slot_indices_ptr,
    context_ptr,
    context_history_ptr,
    history_cursor_ptr,
    num_calls_ptr,
    token_stride,
    context_stride,
    history_stride,
    hash_iv,
    has_lengths: tl.constexpr,
    max_tokens: tl.constexpr,
    context_width: tl.constexpr,
    history_size: tl.constexpr,
    skip_first_ngram_calls: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_indices_ptr + row).to(tl.int64)
    length = max_tokens
    if has_lengths:
        length = tl.load(lengths_ptr + row).to(tl.int64)
    history_cursor = tl.load(history_cursor_ptr + slot).to(tl.int64)
    num_calls = tl.load(num_calls_ptr + slot).to(tl.int64)

    for step in range(max_tokens):
        committed = step < length
        context_hash = _projected_context_hash(
            context_ptr,
            context_ptr,
            slot,
            slot,
            0,
            context_stride=context_stride,
            verify_stride=context_stride,
            context_width=context_width,
            hash_iv=hash_iv,
        )
        record = committed & _eligible(
            num_calls + 1, context_width, skip_first_ngram_calls
        )
        tl.store(
            context_history_ptr + slot * history_stride + history_cursor,
            context_hash,
            mask=record,
        )
        history_cursor = tl.where(
            record,
            (history_cursor + 1) % history_size,
            history_cursor,
        )

        for context_index in range(context_width - 1):
            next_context_token = tl.load(
                context_ptr + slot * context_stride + context_index + 1
            )
            tl.store(
                context_ptr + slot * context_stride + context_index,
                next_context_token,
                mask=committed,
            )
        token_id = tl.load(
            token_ids_ptr + row * token_stride + step,
            mask=committed,
            other=0,
        )
        tl.store(
            context_ptr + slot * context_stride + context_width - 1,
            token_id,
            mask=committed,
        )
        num_calls += committed.to(tl.int64)

    tl.store(history_cursor_ptr + slot, history_cursor)
    tl.store(num_calls_ptr + slot, num_calls)


def _check_candidates(scores: torch.Tensor, token_ids: torch.Tensor) -> None:
    if not scores.is_cuda:
        raise ValueError("The Triton SynthID kernels require CUDA tensors.")
    if scores.dtype != torch.float32:
        raise ValueError(f"Scores must be float32, got {scores.dtype}.")
    if scores.ndim != 2 or token_ids.shape != scores.shape:
        raise ValueError("scores and token_ids must be matching 2D tensors.")
    if scores.stride(1) != 1 or token_ids.stride(1) != 1:
        raise ValueError("scores and token_ids must be contiguous by row.")


def synthid_tournament_triton(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
    apply_mask: torch.Tensor,
) -> torch.Tensor:
    _check_candidates(scores, token_ids)
    if context_hashes.shape != (scores.shape[0],):
        raise ValueError("context_hashes must have one value per score row.")
    if apply_mask.shape != (scores.shape[0],):
        raise ValueError("apply_mask must have one value per score row.")

    output = torch.empty_like(scores)
    block_size = triton.next_power_of_2(scores.shape[1])
    if block_size > MAX_BLOCK_CANDIDATES:
        if keys.numel() > MAX_KEYS:
            raise ValueError(
                f"The wide Triton SynthID path supports at most {MAX_KEYS} keys."
            )
        g_bits = torch.empty_like(token_ids, dtype=torch.int64)
        _synthid_wide_tournament_kernel[(scores.shape[0],)](
            scores,
            token_ids,
            context_hashes,
            keys,
            apply_mask,
            output,
            g_bits,
            scores.stride(0),
            token_ids.stride(0),
            output.stride(0),
            g_bits.stride(0),
            scores.shape[1],
            keys.numel(),
            BLOCK_SIZE=MAX_BLOCK_CANDIDATES,
            num_warps=8,
            num_stages=1,
        )
        return output
    _synthid_tournament_kernel[(scores.shape[0],)](
        scores,
        token_ids,
        context_hashes,
        keys,
        apply_mask,
        output,
        scores.stride(0),
        token_ids.stride(0),
        output.stride(0),
        num_candidates=scores.shape[1],
        depth=keys.numel(),
        BLOCK_SIZE=block_size,
        num_warps=1 if block_size <= 64 else 4,
    )
    return output


def synthid_sample_triton(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    temperatures: torch.Tensor,
    slot_indices: torch.Tensor,
    context: torch.Tensor,
    context_history: torch.Tensor,
    num_calls: torch.Tensor,
    keys: torch.Tensor,
    exponential_noise: torch.Tensor,
    *,
    hash_iv: int,
    skip_first_ngram_calls: bool,
) -> torch.Tensor:
    """Sample each row from the SynthID law with exponential noise; the state is
    read, not committed."""
    _check_candidates(scores, token_ids)
    if exponential_noise.shape != scores.shape:
        raise ValueError("exponential_noise must match scores.")
    if exponential_noise.dtype != torch.float32:
        raise ValueError("exponential_noise must be float32.")
    if slot_indices.shape != (scores.shape[0],):
        raise ValueError("slot_indices must have one value per score row.")
    if temperatures.shape != (scores.shape[0], 1):
        raise ValueError("temperatures must have shape [rows, 1].")

    output_token_ids = torch.empty(
        scores.shape[0], dtype=torch.int32, device=scores.device
    )
    candidate_block_size = triton.next_power_of_2(scores.shape[1])
    history_block_size = triton.next_power_of_2(context_history.shape[1])
    if candidate_block_size > MAX_BLOCK_CANDIDATES:
        raise ValueError(
            "The fused SynthID sampling path supports "
            f"num_candidates <= {MAX_BLOCK_CANDIDATES}."
        )
    if history_block_size > MAX_CONTEXT_HISTORY_SIZE:
        raise ValueError(
            "The fused SynthID sampling path supports "
            f"context history <= {MAX_CONTEXT_HISTORY_SIZE}."
        )
    if context.shape[1] > MAX_CONTEXT_WIDTH:
        raise ValueError(
            "The fused SynthID sampling path supports "
            f"ngram_len <= {MAX_CONTEXT_WIDTH + 1}."
        )

    _synthid_sample_kernel[(scores.shape[0],)](
        scores,
        token_ids,
        exponential_noise,
        temperatures,
        slot_indices,
        context,
        context_history,
        num_calls,
        keys,
        output_token_ids,
        scores.stride(0),
        token_ids.stride(0),
        exponential_noise.stride(0),
        temperatures.stride(0),
        context.stride(0),
        context_history.stride(0),
        hash_iv,
        num_candidates=scores.shape[1],
        context_width=context.shape[1],
        history_size=context_history.shape[1],
        depth=keys.numel(),
        skip_first_ngram_calls=skip_first_ngram_calls,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        HISTORY_BLOCK_SIZE=history_block_size,
        num_warps=4,
        num_stages=1,
    )
    return output_token_ids


def synthid_project_state_triton(
    slot_indices: torch.Tensor,
    verify_tokens: torch.Tensor,
    context: torch.Tensor,
    context_history: torch.Tensor,
    history_cursor: torch.Tensor,
    num_calls: torch.Tensor,
    *,
    hash_iv: int,
    skip_first_ngram_calls: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, verify_width = verify_tokens.shape
    if slot_indices.shape != (batch_size,):
        raise ValueError("slot_indices must have one value per request.")

    history_block_size = triton.next_power_of_2(context_history.shape[1])
    if history_block_size > MAX_CONTEXT_HISTORY_SIZE:
        raise ValueError(
            "The fused SynthID projection supports "
            f"context history <= {MAX_CONTEXT_HISTORY_SIZE}."
        )
    if context.shape[1] > MAX_CONTEXT_WIDTH or verify_width > MAX_CHAIN_WIDTH:
        raise ValueError(
            "The fused SynthID projection supports "
            f"ngram_len <= {MAX_CONTEXT_WIDTH + 1} "
            f"and verify width <= {MAX_CHAIN_WIDTH}."
        )

    context_hashes = torch.empty(
        batch_size * verify_width,
        dtype=torch.int64,
        device=verify_tokens.device,
    )
    apply_mask = torch.empty_like(context_hashes, dtype=torch.bool)
    _synthid_project_state_kernel[(context_hashes.shape[0],)](
        slot_indices,
        verify_tokens,
        context,
        context_history,
        history_cursor,
        num_calls,
        context_hashes,
        apply_mask,
        verify_tokens.stride(0),
        context.stride(0),
        context_history.stride(0),
        hash_iv,
        verify_width=verify_width,
        context_width=context.shape[1],
        history_size=context_history.shape[1],
        skip_first_ngram_calls=skip_first_ngram_calls,
        HISTORY_BLOCK_SIZE=history_block_size,
        num_warps=8,
        num_stages=1,
    )
    return context_hashes, apply_mask


def synthid_commit_state_triton(
    token_ids: torch.Tensor,
    lengths: torch.Tensor | None,
    slot_indices: torch.Tensor,
    context: torch.Tensor,
    context_history: torch.Tensor,
    history_cursor: torch.Tensor,
    num_calls: torch.Tensor,
    *,
    hash_iv: int,
    skip_first_ngram_calls: bool,
) -> None:
    """Commit the first ``lengths[i]`` tokens of row ``i`` (all when None)."""
    if token_ids.ndim != 2 or slot_indices.shape != (token_ids.shape[0],):
        raise ValueError("token_ids must be [rows, width] with one slot per row.")
    if lengths is not None and lengths.shape != slot_indices.shape:
        raise ValueError("lengths must have one value per row.")
    if token_ids.shape[1] > MAX_CHAIN_WIDTH:
        raise ValueError(
            f"The SynthID state commit supports at most {MAX_CHAIN_WIDTH} tokens."
        )

    _synthid_commit_state_kernel[(token_ids.shape[0],)](
        token_ids,
        lengths if lengths is not None else token_ids,
        slot_indices,
        context,
        context_history,
        history_cursor,
        num_calls,
        token_ids.stride(0),
        context.stride(0),
        context_history.stride(0),
        hash_iv,
        has_lengths=lengths is not None,
        max_tokens=token_ids.shape[1],
        context_width=context.shape[1],
        history_size=context_history.shape[1],
        skip_first_ngram_calls=skip_first_ngram_calls,
        num_warps=1,
        num_stages=1,
    )
