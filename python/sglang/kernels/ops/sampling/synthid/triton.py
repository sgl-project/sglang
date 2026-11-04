# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused Triton implementation of the SynthID sparse tournament transform."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_HASH_MULTIPLIER = tl.constexpr(6364136223846793005)


@triton.jit
def _accumulate_hash(value, data):
    return (value + data) * _HASH_MULTIPLIER + 1


@triton.jit
def _projected_context_hash(
    context_ptr,
    verify_tokens_ptr,
    slot,
    request,
    step,
    context_stride: tl.constexpr,
    verify_stride: tl.constexpr,
    context_width: tl.constexpr,
    hash_iv,
):
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
def _synthid_topk_kernel(
    scores_ptr,
    token_ids_ptr,
    context_hashes_ptr,
    keys_ptr,
    apply_mask_ptr,
    temperatures_ptr,
    exponential_noise_ptr,
    output_ptr,
    output_token_ids_ptr,
    output_sampled_probs_ptr,
    output_sorted_tokens_ptr,
    score_stride,
    token_stride,
    noise_stride,
    output_stride,
    temperature_stride,
    num_candidates: tl.constexpr,
    depth: tl.constexpr,
    return_probs: tl.constexpr,
    sample_token: tl.constexpr,
    sort_output: tl.constexpr,
    apply_temperature: tl.constexpr,
    rows_per_request: tl.constexpr,
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
    if apply_temperature:
        temperature = tl.load(
            temperatures_ptr + (row // rows_per_request) * temperature_stride
        )
        scores = tl.div_rn(scores, temperature)
    max_score = tl.max(scores, axis=0)
    base_probs = tl.exp(scores - max_score)
    base_probs = tl.where(mask, base_probs, 0.0)
    base_probs /= tl.sum(base_probs, axis=0)
    probs = base_probs

    token_ids = tl.load(
        token_ids_ptr + row * token_stride + offsets, mask=mask, other=0
    ).to(tl.int64)
    context_hash = tl.load(context_hashes_ptr + row).to(tl.int64)
    candidate_hashes = _accumulate_hash(context_hash, token_ids)

    for key_index in range(depth):
        key = tl.load(keys_ptr + key_index).to(tl.int64)
        hashes = _accumulate_hash(candidate_hashes, key)
        for _ in range(12):
            hashes = _accumulate_hash(hashes, 1) >> 5
        g_values = ((hashes >> 30) & 1).to(tl.float32)
        g_values = tl.where(mask, g_values, 0.0)
        g_mass = tl.sum(g_values * probs, axis=0)
        probs *= 1.0 + g_values - g_mass

    should_apply = tl.load(apply_mask_ptr + row)
    selected_probs = tl.where(should_apply, probs, base_probs)
    if return_probs:
        output = tl.where(mask, selected_probs, 0.0)
    else:
        output = tl.log(probs)
        output = tl.where(probs > 0.0, output, -1.0e12)
        output = tl.where(should_apply, output, scores)
    if sort_output:
        # Keep the original candidate/noise pairing; reorder only stored proposals.
        packed = token_ids.to(tl.uint64) * BLOCK_SIZE + offsets.to(tl.uint64)
        packed = tl.where(mask, packed, 0xFFFFFFFFFFFFFFFF)
        packed = tl.sort(packed, descending=False)
        order = tl.where(mask, packed % BLOCK_SIZE, 0).to(tl.int32)
        sorted_output = tl.gather(output, order, axis=0)
        tl.store(
            output_sorted_tokens_ptr + row * output_stride + offsets,
            (packed // BLOCK_SIZE).to(tl.int64),
            mask=mask,
        )
        tl.store(output_ptr + row * output_stride + offsets, sorted_output, mask=mask)
    else:
        tl.store(output_ptr + row * output_stride + offsets, output, mask=mask)

    if sample_token:
        noise = tl.load(
            exponential_noise_ptr + row * noise_stride + offsets,
            mask=mask,
            other=1.0,
        ).to(tl.float32)
        sampling_scores = tl.where(
            mask,
            selected_probs / tl.maximum(noise, 1.1754943508222875e-38),
            -float("inf"),
        )
        sampled_index = tl.argmax(sampling_scores, axis=0)
        sampled_token_id = tl.sum(
            tl.where(offsets == sampled_index, token_ids, 0),
            axis=0,
        )
        sampled_prob = tl.sum(
            tl.where(offsets == sampled_index, selected_probs, 0.0),
            axis=0,
        )
        tl.store(output_token_ids_ptr + row, sampled_token_id)
        tl.store(output_sampled_probs_ptr + row, sampled_prob)


@triton.jit
def _synthid_wide_topk_kernel(
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
    return_probs: tl.constexpr,
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
        g_bits = token_ids ^ token_ids
        for key_index in tl.range(0, depth):
            key = tl.load(keys_ptr + key_index).to(tl.int64)
            hashes = _accumulate_hash(candidate_hashes, key)
            for _ in range(12):
                hashes = _accumulate_hash(hashes, 1) >> 5
            g_value = ((hashes >> 30) & 1).to(tl.int64)
            g_bits |= g_value << key_index
        g_bits = tl.where(should_apply, g_bits, 0)
        tl.store(g_bits_base + candidate_offsets, g_bits, mask=mask)

    denominator = tl.maximum(denominator, 1.1754943508222875e-38)
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
                if return_probs:
                    output = probs
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
def _synthid_sample_and_commit_kernel(
    scores_ptr,
    token_ids_ptr,
    exponential_noise_ptr,
    temperatures_ptr,
    slot_indices_ptr,
    context_ptr,
    context_history_ptr,
    history_cursor_ptr,
    num_calls_ptr,
    keys_ptr,
    commit_mask_ptr,
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
    commit_state: tl.constexpr,
    use_exponential_noise: tl.constexpr,
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

    context_hash = hash_iv.to(tl.int64)
    for context_index in range(context_width):
        context_token = tl.load(context_ptr + slot * context_stride + context_index).to(
            tl.int64
        )
        context_hash = _accumulate_hash(context_hash, context_token)

    call_number = tl.load(num_calls_ptr + slot).to(tl.int64) + 1
    eligible = True
    if skip_first_ngram_calls:
        eligible = call_number >= context_width + 1

    history_offsets = tl.arange(0, HISTORY_BLOCK_SIZE)
    history_mask = history_offsets < history_size
    history_values = tl.load(
        context_history_ptr + slot * history_stride + history_offsets,
        mask=history_mask,
        other=0,
    ).to(tl.int64)
    repeated = (
        tl.max(
            tl.where(
                history_mask & (history_values == context_hash),
                1,
                0,
            ),
            axis=0,
        )
        != 0
    )

    token_ids = tl.load(
        token_ids_ptr + row * token_stride + candidate_offsets,
        mask=candidate_mask,
        other=0,
    ).to(tl.int64)
    candidate_hashes = _accumulate_hash(context_hash, token_ids)
    for key_index in range(depth):
        key = tl.load(keys_ptr + key_index).to(tl.int64)
        hashes = _accumulate_hash(candidate_hashes, key)
        for _ in range(12):
            hashes = _accumulate_hash(hashes, 1) >> 5
        g_values = ((hashes >> 30) & 1).to(tl.float32)
        g_values = tl.where(valid_mask, g_values, 0.0)
        g_mass = tl.sum(g_values * probs, axis=0)
        probs *= 1.0 + g_values - g_mass

    selection_probs = tl.where(eligible & ~repeated, probs, base_probs)
    selection_scores = selection_probs
    if use_exponential_noise:
        noise = tl.load(
            exponential_noise_ptr + row * noise_stride + candidate_offsets,
            mask=valid_mask,
            other=1.0,
        ).to(tl.float32)
        selection_scores /= tl.maximum(noise, 1.1754943508222875e-38)
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

    if commit_state:
        should_commit = tl.load(commit_mask_ptr + row)
        history_cursor = tl.load(history_cursor_ptr + slot).to(tl.int64)
        tl.store(
            context_history_ptr + slot * history_stride + history_cursor,
            context_hash,
            mask=should_commit & eligible,
        )
        tl.store(
            history_cursor_ptr + slot,
            (history_cursor + 1) % history_size,
            mask=should_commit & eligible,
        )
        for context_index in range(context_width - 1):
            next_context_token = tl.load(
                context_ptr + slot * context_stride + context_index + 1
            )
            tl.store(
                context_ptr + slot * context_stride + context_index,
                next_context_token,
                mask=should_commit,
            )
        tl.store(
            context_ptr + slot * context_stride + context_width - 1,
            selected_token_id,
            mask=should_commit,
        )
        tl.store(num_calls_ptr + slot, call_number, mask=should_commit)


@triton.jit
def _synthid_project_kernel(
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
    call_number = base_num_calls + step + 1
    eligible = True
    if skip_first_ngram_calls:
        eligible = call_number >= context_width + 1

    history_offsets = tl.arange(0, HISTORY_BLOCK_SIZE)
    history_mask = history_offsets < history_size
    history_values = tl.load(
        context_history_ptr + slot * history_stride + history_offsets,
        mask=history_mask,
        other=0,
    ).to(tl.int64)
    projected_cursor = tl.load(history_cursor_ptr + slot).to(tl.int64)
    for previous_step in range(verify_width):
        active_previous = previous_step < step
        previous_call_number = base_num_calls + previous_step + 1
        previous_eligible = active_previous
        if skip_first_ngram_calls:
            previous_eligible &= previous_call_number >= context_width + 1
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

    repeated = (
        tl.max(
            tl.where(
                history_mask & (history_values == context_hash),
                1,
                0,
            ),
            axis=0,
        )
        != 0
    )
    tl.store(context_hashes_ptr + flat_row, context_hash)
    tl.store(apply_mask_ptr + flat_row, eligible & ~repeated)


@triton.jit
def _synthid_commit_kernel(
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
    use_lengths: tl.constexpr,
    max_tokens: tl.constexpr,
    context_width: tl.constexpr,
    history_size: tl.constexpr,
    skip_first_ngram_calls: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_indices_ptr + row).to(tl.int64)
    length = 1
    if use_lengths:
        length = tl.load(lengths_ptr + row).to(tl.int64)
    history_cursor = tl.load(history_cursor_ptr + slot).to(tl.int64)
    num_calls = tl.load(num_calls_ptr + slot).to(tl.int64)

    for step in range(max_tokens):
        committed = step < length
        context_hash = hash_iv.to(tl.int64)
        for context_index in range(context_width):
            context_token = tl.load(
                context_ptr + slot * context_stride + context_index
            ).to(tl.int64)
            context_hash = _accumulate_hash(context_hash, context_token)

        call_number = num_calls + 1
        record = committed
        if skip_first_ngram_calls:
            record &= call_number >= context_width + 1
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


def watermark_topk_scores_triton(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
    apply_mask: torch.Tensor,
) -> torch.Tensor:
    if not scores.is_cuda:
        raise ValueError("The Triton SynthID backend requires CUDA tensors.")
    if scores.dtype != torch.float32:
        raise ValueError(f"Scores must be float32, got {scores.dtype}.")
    if scores.ndim != 2 or token_ids.shape != scores.shape:
        raise ValueError("scores and token_ids must be matching 2D tensors.")
    if scores.stride(1) != 1 or token_ids.stride(1) != 1:
        raise ValueError("scores and token_ids must be contiguous by row.")
    if context_hashes.shape != (scores.shape[0],):
        raise ValueError("context_hashes must have one value per score row.")
    if apply_mask.shape != (scores.shape[0],):
        raise ValueError("apply_mask must have one value per score row.")

    output = torch.empty_like(scores)
    if scores.shape[0] == 0:
        return output
    block_size = triton.next_power_of_2(scores.shape[1])
    if block_size > 1024:
        if keys.numel() > 64:
            raise ValueError("The wide Triton SynthID path supports at most 64 keys.")
        g_bits = torch.empty_like(token_ids)
        _synthid_wide_topk_kernel[(scores.shape[0],)](
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
            return_probs=False,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=1,
        )
        return output
    _synthid_topk_kernel[(scores.shape[0],)](
        scores,
        token_ids,
        context_hashes,
        keys,
        apply_mask,
        scores,
        scores,
        output,
        output,
        output,
        output,
        scores.stride(0),
        token_ids.stride(0),
        0,
        output.stride(0),
        0,
        num_candidates=scores.shape[1],
        depth=keys.numel(),
        return_probs=False,
        sample_token=False,
        sort_output=False,
        apply_temperature=False,
        rows_per_request=1,
        BLOCK_SIZE=block_size,
        num_warps=1 if block_size <= 64 else 4,
    )
    return output


def watermark_topk_probs_and_sample_triton(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
    apply_mask: torch.Tensor,
    exponential_noise: torch.Tensor | None = None,
    *,
    temperatures: torch.Tensor | None = None,
    rows_per_request: int = 1,
    sorted_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Optionally store proposals in vocabulary order without reordering sampling."""
    if not scores.is_cuda:
        raise ValueError("The fused SynthID probability path requires CUDA tensors.")
    if scores.dtype != torch.float32:
        raise ValueError(f"Scores must be float32, got {scores.dtype}.")
    if scores.ndim != 2 or token_ids.shape != scores.shape:
        raise ValueError("scores and token_ids must be matching 2D tensors.")
    if context_hashes.shape != (scores.shape[0],):
        raise ValueError("context_hashes must have one value per score row.")
    if apply_mask.shape != (scores.shape[0],):
        raise ValueError("apply_mask must have one value per score row.")
    if exponential_noise is not None and exponential_noise.shape != scores.shape:
        raise ValueError("exponential_noise must match scores.")

    sample_token = exponential_noise is not None
    output_probs = torch.empty_like(scores)
    if sorted_tokens is not None:
        if (
            sorted_tokens.shape != token_ids.shape
            or sorted_tokens.dtype != torch.int64
            or sorted_tokens.device != scores.device
            or sorted_tokens.stride() != output_probs.stride()
        ):
            raise ValueError("sorted_tokens must match the contiguous output layout.")
        if scores.shape[1] > 1024:
            raise ValueError("Fused proposal sorting requires top-k <= 1024.")
    output_token_ids = (
        torch.empty(scores.shape[0], dtype=torch.int64, device=scores.device)
        if sample_token
        else None
    )
    output_sampled_probs = (
        torch.empty(scores.shape[0], dtype=torch.float32, device=scores.device)
        if sample_token
        else None
    )
    if scores.shape[0] == 0:
        return output_probs, output_token_ids, output_sampled_probs

    block_size = triton.next_power_of_2(scores.shape[1])
    if temperatures is not None:
        if (
            rows_per_request < 1
            or scores.shape[0] % rows_per_request != 0
            or temperatures.shape != (scores.shape[0] // rows_per_request, 1)
        ):
            raise ValueError("Temperatures must match the SynthID request rows.")
        if block_size > 1024:
            raise ValueError("Fused temperature scaling requires top-k <= 1024.")
    if block_size > 1024:
        if keys.numel() > 64:
            raise ValueError("The wide Triton SynthID path supports at most 64 keys.")
        g_bits = torch.empty_like(token_ids)
        _synthid_wide_topk_kernel[(scores.shape[0],)](
            scores,
            token_ids,
            context_hashes,
            keys,
            apply_mask,
            output_probs,
            g_bits,
            scores.stride(0),
            token_ids.stride(0),
            output_probs.stride(0),
            g_bits.stride(0),
            scores.shape[1],
            keys.numel(),
            return_probs=True,
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=1,
        )
        if exponential_noise is not None:
            sampled_indices = (
                output_probs / exponential_noise.clamp_min(1.1754943508222875e-38)
            ).argmax(dim=-1, keepdim=True)
            output_token_ids = token_ids.gather(1, sampled_indices).view(-1)
            output_sampled_probs = output_probs.gather(1, sampled_indices).view(-1)
        return output_probs, output_token_ids, output_sampled_probs
    if exponential_noise is None:
        exponential_noise = scores
        noise_stride = 0
    else:
        noise_stride = exponential_noise.stride(0)
    output_token_ids_ptr = (
        output_token_ids if output_token_ids is not None else output_probs
    )
    output_sampled_probs_ptr = (
        output_sampled_probs if output_sampled_probs is not None else output_probs
    )

    _synthid_topk_kernel[(scores.shape[0],)](
        scores,
        token_ids,
        context_hashes,
        keys,
        apply_mask,
        temperatures if temperatures is not None else scores,
        exponential_noise,
        output_probs,
        output_token_ids_ptr,
        output_sampled_probs_ptr,
        sorted_tokens if sorted_tokens is not None else output_probs,
        scores.stride(0),
        token_ids.stride(0),
        noise_stride,
        output_probs.stride(0),
        temperatures.stride(0) if temperatures is not None else 0,
        num_candidates=scores.shape[1],
        depth=keys.numel(),
        return_probs=True,
        sample_token=sample_token,
        sort_output=sorted_tokens is not None,
        apply_temperature=temperatures is not None,
        rows_per_request=rows_per_request,
        BLOCK_SIZE=block_size,
        num_warps=1 if block_size <= 64 else 4,
    )
    return output_probs, output_token_ids, output_sampled_probs


def synthid_sample_and_commit_triton(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    temperatures: torch.Tensor,
    slot_indices: torch.Tensor,
    context: torch.Tensor,
    context_history: torch.Tensor,
    history_cursor: torch.Tensor,
    num_calls: torch.Tensor,
    keys: torch.Tensor,
    commit_mask: torch.Tensor,
    exponential_noise: torch.Tensor | None = None,
    *,
    hash_iv: int,
    skip_first_ngram_calls: bool,
) -> torch.Tensor:
    if not scores.is_cuda:
        raise ValueError("The fused SynthID sampling path requires CUDA tensors.")
    if scores.dtype != torch.float32:
        raise ValueError(f"Scores must be float32, got {scores.dtype}.")
    if scores.ndim != 2 or token_ids.shape != scores.shape:
        raise ValueError("scores and token_ids must be matching 2D tensors.")
    if exponential_noise is not None:
        if exponential_noise.shape != scores.shape:
            raise ValueError("exponential_noise must match scores.")
        if exponential_noise.dtype != torch.float32:
            raise ValueError("exponential_noise must be float32.")
    if slot_indices.shape != (scores.shape[0],):
        raise ValueError("slot_indices must have one value per score row.")
    if commit_mask.shape != slot_indices.shape:
        raise ValueError("commit_mask must have one value per score row.")
    if temperatures.shape != (scores.shape[0], 1):
        raise ValueError("temperatures must have shape [rows, 1].")

    output_token_ids = torch.empty(
        scores.shape[0], dtype=torch.int32, device=scores.device
    )
    if scores.shape[0] == 0:
        return output_token_ids

    candidate_block_size = triton.next_power_of_2(scores.shape[1])
    history_block_size = triton.next_power_of_2(context_history.shape[1])
    if candidate_block_size > 1024:
        raise ValueError("The fused SynthID sampling path supports top_k <= 1024.")
    if history_block_size > 2048:
        raise ValueError(
            "The fused SynthID sampling path supports context history <= 2048."
        )
    if context.shape[1] > 16:
        raise ValueError("The fused SynthID sampling path supports ngram_len <= 17.")

    use_exponential_noise = exponential_noise is not None
    if exponential_noise is None:
        exponential_noise = scores
        noise_stride = 0
    else:
        noise_stride = exponential_noise.stride(0)
    _synthid_sample_and_commit_kernel[(scores.shape[0],)](
        scores,
        token_ids,
        exponential_noise,
        temperatures,
        slot_indices,
        context,
        context_history,
        history_cursor,
        num_calls,
        keys,
        commit_mask,
        output_token_ids,
        scores.stride(0),
        token_ids.stride(0),
        noise_stride,
        temperatures.stride(0),
        context.stride(0),
        context_history.stride(0),
        hash_iv,
        num_candidates=scores.shape[1],
        context_width=context.shape[1],
        history_size=context_history.shape[1],
        depth=keys.numel(),
        commit_state=True,
        use_exponential_noise=use_exponential_noise,
        skip_first_ngram_calls=skip_first_ngram_calls,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        HISTORY_BLOCK_SIZE=history_block_size,
        num_warps=4,
        num_stages=1,
    )
    return output_token_ids


def project_synthid_speculative_state_triton(
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
    if history_block_size > 2048:
        raise ValueError(
            "The fused SynthID projection supports context history <= 2048."
        )
    if context.shape[1] > 16 or verify_width > 8:
        raise ValueError(
            "The fused SynthID projection supports ngram_len <= 17 "
            "and verify width <= 8."
        )

    context_hashes = torch.empty(
        batch_size * verify_width,
        dtype=torch.int64,
        device=verify_tokens.device,
    )
    apply_mask = torch.empty_like(context_hashes, dtype=torch.bool)
    _synthid_project_kernel[(context_hashes.shape[0],)](
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


def commit_synthid_state_triton(
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    slot_indices: torch.Tensor,
    context: torch.Tensor,
    context_history: torch.Tensor,
    history_cursor: torch.Tensor,
    num_calls: torch.Tensor,
    *,
    hash_iv: int,
    skip_first_ngram_calls: bool,
) -> None:
    if token_ids.ndim != 2 or lengths.shape != (token_ids.shape[0],):
        raise ValueError("token_ids and lengths must describe one batch.")
    if slot_indices.shape != lengths.shape:
        raise ValueError("slot_indices must have one value per request.")

    _synthid_commit_kernel[(token_ids.shape[0],)](
        token_ids,
        lengths,
        slot_indices,
        context,
        context_history,
        history_cursor,
        num_calls,
        token_ids.stride(0),
        context.stride(0),
        context_history.stride(0),
        hash_iv,
        use_lengths=True,
        max_tokens=token_ids.shape[1],
        context_width=context.shape[1],
        history_size=context_history.shape[1],
        skip_first_ngram_calls=skip_first_ngram_calls,
        num_warps=1,
        num_stages=1,
    )


def commit_synthid_token_triton(
    token_ids: torch.Tensor,
    slot_indices: torch.Tensor,
    commit_mask: torch.Tensor,
    context: torch.Tensor,
    context_history: torch.Tensor,
    history_cursor: torch.Tensor,
    num_calls: torch.Tensor,
    *,
    hash_iv: int,
    skip_first_ngram_calls: bool,
) -> None:
    if (
        token_ids.ndim != 1
        or slot_indices.shape != token_ids.shape
        or commit_mask.shape != token_ids.shape
    ):
        raise ValueError(
            "token_ids, slot_indices, and commit_mask must be matching 1D tensors."
        )

    _synthid_commit_kernel[(token_ids.shape[0],)](
        token_ids,
        commit_mask,
        slot_indices,
        context,
        context_history,
        history_cursor,
        num_calls,
        token_ids.stride(0),
        context.stride(0),
        context_history.stride(0),
        hash_iv,
        use_lengths=True,
        max_tokens=1,
        context_width=context.shape[1],
        history_size=context_history.shape[1],
        skip_first_ngram_calls=skip_first_ngram_calls,
        num_warps=1,
        num_stages=1,
    )


@triton.jit
def _lookup_sorted_sparse_prob(tokens, probs, token, TOP_K: tl.constexpr):
    low = tl.zeros_like(token).to(tl.int32)
    high = low + TOP_K
    for _ in range(11):
        mid = (low + high) // 2
        in_bounds = mid < TOP_K
        mid_token = tl.load(tokens + mid, mask=in_bounds, other=2147483647)
        go_right = mid_token < token
        low = tl.where(go_right, mid + 1, low)
        high = tl.where(go_right, high, mid)
    in_bounds = low < TOP_K
    matched_token = tl.load(tokens + low, mask=in_bounds, other=-1)
    matched_prob = tl.load(probs + low, mask=in_bounds, other=0.0)
    return tl.where(in_bounds & (matched_token == token), matched_prob, 0.0)


@triton.jit
def _synthid_speculative_sampling_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetrieveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    TargetTokens,
    DraftProbs,
    DraftTokens,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_k,
    stride_tt_b,
    stride_tt_s,
    stride_tt_k,
    stride_dp_b,
    stride_dp_s,
    stride_dp_k,
    stride_dt_b,
    stride_dt_s,
    stride_dt_k,
    NUM_SLOTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    candidate_mask = offsets < TOP_K
    candidate_base = Candidates + pid * stride_cand_b
    index_base = RetrieveIndex + pid * stride_idx_b
    uniform_base = UniformSamples + pid * stride_uni_b
    current_row = 0
    num_correct_drafts = 0
    continue_verifying = 1

    root_index = tl.load(index_base)
    tl.store(AcceptIndex + pid * stride_idx_b, root_index)
    last_accept_index = root_index

    step = 1
    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(candidate_base + step * stride_cand_s)
        target_tokens = tl.load(
            TargetTokens
            + pid * stride_tt_b
            + current_row * stride_tt_s
            + offsets * stride_tt_k,
            mask=candidate_mask,
            other=-1,
        )
        target_probs = tl.load(
            TargetProbs
            + pid * stride_tp_b
            + current_row * stride_tp_s
            + offsets * stride_tp_k,
            mask=candidate_mask,
            other=0.0,
        )
        target_prob = tl.sum(
            tl.where(
                candidate_mask & (target_tokens == draft_token),
                target_probs,
                0.0,
            ),
            axis=0,
        )
        draft_prob = _lookup_sorted_sparse_prob(
            DraftTokens + pid * stride_dt_b + current_row * stride_dt_s,
            DraftProbs + pid * stride_dp_b + current_row * stride_dp_s,
            draft_token,
            TOP_K=TOP_K,
        )
        coin = tl.load(uniform_base + (step - 1) * stride_uni_s)
        if coin * draft_prob < target_prob:
            num_correct_drafts += 1
            current_row = step
            tl.store(Predicts + last_accept_index, draft_token)
            current_index = tl.load(index_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_correct_drafts * stride_idx_s,
                current_index,
            )
            last_accept_index = current_index
            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_correct_drafts)
    target_tokens = tl.load(
        TargetTokens
        + pid * stride_tt_b
        + current_row * stride_tt_s
        + offsets * stride_tt_k,
        mask=candidate_mask,
        other=-1,
    )
    target_probs = tl.load(
        TargetProbs
        + pid * stride_tp_b
        + current_row * stride_tp_s
        + offsets * stride_tp_k,
        mask=candidate_mask,
        other=0.0,
    )
    final_probs = target_probs
    if continue_verifying == 0:
        draft_probs = _lookup_sorted_sparse_prob(
            DraftTokens + pid * stride_dt_b + current_row * stride_dt_s,
            DraftProbs + pid * stride_dp_b + current_row * stride_dp_s,
            target_tokens,
            TOP_K=TOP_K,
        )
        final_probs = tl.maximum(target_probs - draft_probs, 0.0)
        residual_sum = tl.sum(final_probs, axis=0)
        final_probs = tl.where(residual_sum > 0.0, final_probs, target_probs)

    target = tl.load(UniformSamplesFinal + pid) * tl.sum(final_probs, axis=0)
    cumulative = tl.cumsum(final_probs, axis=0)
    matches = candidate_mask & (cumulative > target)
    has_match = tl.max(matches.to(tl.int32), axis=0) != 0
    sampled_index = tl.argmax(matches.to(tl.int32), axis=0)
    sampled_index = tl.where(has_match, sampled_index, TOP_K - 1)
    bonus_token = tl.sum(
        tl.where(offsets == sampled_index, target_tokens, 0),
        axis=0,
    )
    tl.store(Predicts + last_accept_index, bonus_token)


@triton.jit
def _synthid_zero_step_kernel(
    Probs,
    UniformSamples,
    OutputTokens,
    stride_p_b,
    stride_p_v,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    prob_base = Probs + pid * stride_p_b
    norm_sum = 0.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = v_start + tl.arange(0, BLOCK_V)
        norm_sum += tl.sum(
            tl.load(
                prob_base + offsets * stride_p_v,
                mask=offsets < VOCAB_SIZE,
                other=0.0,
            )
        )

    target = tl.load(UniformSamples + pid) * norm_sum
    cumulative = 0.0
    output_token = VOCAB_SIZE - 1
    found = 0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            offsets = v_start + tl.arange(0, BLOCK_V)
            values = tl.load(
                prob_base + offsets * stride_p_v,
                mask=offsets < VOCAB_SIZE,
                other=0.0,
            )
            block_cumsum = cumulative + tl.cumsum(values, axis=0)
            matches = block_cumsum > target
            if tl.max(matches, axis=0):
                output_token = v_start + tl.argmax(matches.to(tl.int32), axis=0)
                found = 1
            cumulative += tl.sum(values)
    tl.store(OutputTokens + pid, output_token)


def sample_synthid_zero_step_triton(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    retrieve_index: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
) -> None:
    probs = target_probs[:, 0]
    final_tokens = probs.new_empty((probs.shape[0],), dtype=torch.int64)
    _synthid_zero_step_kernel[(probs.shape[0],)](
        probs,
        uniform_samples_for_final_sampling,
        final_tokens,
        probs.stride(0),
        probs.stride(1),
        VOCAB_SIZE=probs.shape[1],
        BLOCK_V=4096,
    )
    accept_index[:, 0].copy_(retrieve_index[:, 0].to(accept_index.dtype))
    accept_token_num.zero_()
    predicts[retrieve_index[:, 0]] = final_tokens.to(predicts.dtype)


def chain_synthid_speculative_sampling_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    target_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> None:
    top_k = target_probs.shape[-1]
    if draft_probs.shape[-1] != top_k or draft_tokens.shape != draft_probs.shape:
        raise ValueError("Sparse target and draft distributions must share top_k.")
    block_k = triton.next_power_of_2(top_k)
    if block_k > 1024:
        raise ValueError("SynthID speculative sampling supports top_k <= 1024.")

    _synthid_speculative_sampling_kernel[(candidates.shape[0],)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        target_tokens,
        draft_probs,
        draft_tokens,
        candidates.stride(0),
        candidates.stride(1),
        retrieve_index.stride(0),
        retrieve_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        target_tokens.stride(0),
        target_tokens.stride(1),
        target_tokens.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        draft_tokens.stride(2),
        NUM_SLOTS=candidates.shape[1],
        TOP_K=top_k,
        BLOCK_K=block_k,
        num_warps=4,
    )


__all__ = [
    "chain_synthid_speculative_sampling_triton",
    "commit_synthid_token_triton",
    "commit_synthid_state_triton",
    "project_synthid_speculative_state_triton",
    "synthid_sample_and_commit_triton",
    "watermark_topk_probs_and_sample_triton",
    "watermark_topk_scores_triton",
]
