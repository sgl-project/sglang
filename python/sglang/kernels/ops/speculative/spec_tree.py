# Copyright 2023-2026 SGLang Team
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
import triton
import triton.language as tl


@triton.jit
def sgl_build_tree_kernel_efficient_triton(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    seq_len_prefix_sum_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    topk: tl.constexpr,
    depth: tl.constexpr,
    draft_token_num: tl.constexpr,
    tree_mask_mode: tl.constexpr,
    batch_size: tl.constexpr,
    parent_list_stride: tl.constexpr,
    selected_index_stride: tl.constexpr,
):
    """
    Triton kernel for building EAGLE tree structure.
    Each program handles one batch item (batch_idx).
    """
    batch_idx = tl.program_id(0)

    # Calculate seq_tree_idx
    seq_len = tl.load(verified_seq_len_ptr + batch_idx)
    seq_len_prefix_sum = tl.load(seq_len_prefix_sum_ptr + batch_idx)

    # Cast initial value to match the dtype of loaded tensors to avoid type inconsistency
    seq_tree_idx = (
        tl.cast(draft_token_num * draft_token_num * batch_idx, seq_len.dtype)
        + seq_len_prefix_sum * draft_token_num
    )

    positions_offset = batch_idx * draft_token_num
    tl.store(positions_ptr + positions_offset, seq_len)

    retrieve_index_offset = batch_idx * draft_token_num

    # Build retrieval index structure (reverse loop from draft_token_num-1 to 1)
    for i in range(draft_token_num - 1, 0, -1):
        current_token_idx = retrieve_index_offset + i
        tl.store(
            retrieve_index_ptr + batch_idx * draft_token_num + i,
            current_token_idx,
        )

        parent_tb_idx = (
            tl.load(selected_index_ptr + batch_idx * selected_index_stride + (i - 1))
            // topk
        )
        parent_position = 0
        found = 0

        if parent_tb_idx == 0:
            found = 1
        else:
            parent_token_idx = tl.load(
                parent_list_ptr + batch_idx * parent_list_stride + parent_tb_idx
            )

            # Find parent position
            for pp in range(draft_token_num - 1):
                if found == 0:
                    sel_idx = tl.load(
                        selected_index_ptr + batch_idx * selected_index_stride + pp
                    )
                    if sel_idx == parent_token_idx:
                        parent_position = pp + 1
                        found = 1

        if found == 1:
            # Update next token links
            next_tok_addr = (
                retrieve_next_token_ptr + batch_idx * draft_token_num + parent_position
            )
            next_tok = tl.load(next_tok_addr)

            if next_tok == -1:
                tl.store(next_tok_addr, i)
            else:
                tl.store(next_tok_addr, i)
                tl.store(
                    retrieve_next_sibling_ptr + batch_idx * draft_token_num + i,
                    next_tok,
                )

    tl.store(retrieve_index_ptr + batch_idx * draft_token_num, retrieve_index_offset)

    # Process all draft token indices for tree mask
    for draft_tokenx in range(draft_token_num):
        if tree_mask_mode == 0:  # FULL_MASK
            token_tree_idx = (
                seq_tree_idx + (seq_len + draft_token_num) * draft_tokenx + seq_len + 1
            )
        else:
            token_tree_idx = (
                draft_token_num * draft_token_num * batch_idx
                + draft_token_num * draft_tokenx
                + 1
            )

        tl.store(tree_mask_ptr + token_tree_idx - 1, 1)
        for i in range(draft_token_num - 1):
            tl.store(tree_mask_ptr + token_tree_idx + i, 0)

        if draft_tokenx > 0:
            # Build tree path for draft_tokenx > 0
            cur_position = draft_tokenx - 1
            position = 0
            should_continue = 1

            for _ in range(depth):
                if should_continue:
                    position += 1
                    tl.store(tree_mask_ptr + token_tree_idx + cur_position, 1)

                    parent_tb_idx = (
                        tl.load(
                            selected_index_ptr
                            + batch_idx * selected_index_stride
                            + cur_position
                        )
                        // topk
                    )
                    if parent_tb_idx == 0:
                        should_continue = 0
                    else:
                        parent_token_idx = tl.load(
                            parent_list_ptr
                            + batch_idx * parent_list_stride
                            + parent_tb_idx
                        )

                        # Find cur_position for next iteration
                        found = 0
                        for cp in range(draft_token_num - 1):
                            if found == 0:
                                if (
                                    tl.load(
                                        selected_index_ptr
                                        + batch_idx * selected_index_stride
                                        + cp
                                    )
                                    == parent_token_idx
                                ):
                                    cur_position = cp
                                    found = 1
                        if found == 0:
                            should_continue = 0

            tl.store(
                positions_ptr + batch_idx * draft_token_num + draft_tokenx,
                position + seq_len,
            )


@triton.jit
def tree_speculative_sampling_target_only_kernel_triton(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    uniform_samples_ptr,
    uniform_samples_for_final_sampling_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    threshold_single,
    threshold_acc,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_acc_b,
    stride_acc_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    num_speculative_tokens: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Triton port of sgl-kernel's TreeSpeculativeSamplingTargetOnly.

    Walks the draft tree accepting nodes against the target distribution, then
    samples the first rejected (or bonus) token from the residual
    ``relu(target_probs - draft_probs)``. One program per batch item.

    Kept semantically identical to
    ``sgl-kernel/csrc/speculative/speculative_sampling.cuh`` so devices without
    that CUDA kernel (ROCm) keep the same sampling distribution. In particular
    the rejected-token write-back into ``draft_probs`` is what makes the
    "target only" variant exclude already-rejected tokens from the residual.
    """
    bx = tl.program_id(0)

    cand_base = bx * stride_cand_b
    idx_base = bx * stride_idx_b
    uni_base = bx * stride_uni_b

    # Current row of target_probs/draft_probs, as a step index within the batch item.
    cur_prob_step = tl.cast(0, tl.int32)
    prob_acc = tl.cast(0.0, tl.float32)
    coin = tl.load(uniform_samples_ptr + uni_base)

    last_accept_retrieve_idx = tl.load(retrieve_index_ptr + idx_base)
    tl.store(accept_index_ptr + bx * stride_acc_b, last_accept_retrieve_idx)
    num_accept_tokens = tl.cast(0, tl.int32)
    cur_index = tl.cast(0, last_accept_retrieve_idx.dtype)

    walking = tl.cast(1, tl.int32)
    for _level in range(1, num_speculative_tokens):
        if walking == 1:
            cur_index = tl.load(
                retrieve_next_token_ptr + idx_base + cur_index * stride_idx_s
            )
            accepted_level = tl.cast(0, tl.int32)
            # Sibling scan: at most num_draft_tokens siblings on a level.
            for _sibling in range(num_draft_tokens):
                if accepted_level == 0:
                    if cur_index != -1:
                        draft_index = tl.load(
                            retrieve_index_ptr + idx_base + cur_index * stride_idx_s
                        )
                        draft_token = tl.load(
                            candidates_ptr + cand_base + cur_index * stride_cand_s
                        )
                        prob_offset = (
                            bx * stride_tp_b
                            + cur_prob_step * stride_tp_s
                            + draft_token * stride_tp_v
                        )
                        target_prob_single = tl.load(target_probs_ptr + prob_offset)
                        prob_acc += target_prob_single

                        if (coin <= prob_acc / threshold_acc) | (
                            target_prob_single >= threshold_single
                        ):
                            prob_acc = tl.cast(0.0, tl.float32)
                            cur_prob_step = cur_index.to(tl.int32)
                            coin = tl.load(
                                uniform_samples_ptr
                                + uni_base
                                + cur_index * stride_uni_s
                            )
                            tl.store(
                                predicts_ptr + last_accept_retrieve_idx,
                                draft_token.to(tl.int32),
                            )
                            num_accept_tokens += 1
                            tl.store(
                                accept_index_ptr
                                + bx * stride_acc_b
                                + num_accept_tokens * stride_acc_s,
                                draft_index,
                            )
                            last_accept_retrieve_idx = draft_index
                            accepted_level = tl.cast(1, tl.int32)
                        else:
                            # Mask this token out of the residual distribution.
                            tl.store(
                                draft_probs_ptr
                                + bx * stride_dp_b
                                + cur_prob_step * stride_dp_s
                                + draft_token * stride_dp_v,
                                target_prob_single,
                            )
                            cur_index = tl.load(
                                retrieve_next_sibling_ptr
                                + idx_base
                                + cur_index * stride_idx_s
                            )
            if accepted_level == 0:
                walking = tl.cast(0, tl.int32)

    tl.store(accept_token_num_ptr + bx, num_accept_tokens)

    # Sample the first rejected (or bonus) token from relu(target - draft).
    coin_final = tl.load(uniform_samples_for_final_sampling_ptr + bx)
    # No draft distribution exists for the bonus token: every level was accepted.
    all_drafts_accepted = num_accept_tokens == num_speculative_tokens - 1

    tp_base_ptr = target_probs_ptr + bx * stride_tp_b + cur_prob_step * stride_tp_s
    dp_base_ptr = draft_probs_ptr + bx * stride_dp_b + cur_prob_step * stride_dp_s

    # Pass 1: normalization constant.
    norm_sum = tl.cast(0.0, tl.float32)
    for v_start in range(0, vocab_size, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < vocab_size
        p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
        if all_drafts_accepted:
            val = p_val
        else:
            q_val = tl.load(dp_base_ptr + v_offsets * stride_dp_v, mask=mask, other=0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)
        norm_sum += tl.sum(val)

    # Pass 2: inverse CDF. A degenerate residual (norm_sum == 0) leaves the
    # cumsum at 0 <= target_u and falls back to vocab_size - 1, matching
    # reject_sampling.py's chain kernel.
    target_u = coin_final * norm_sum
    cum_sum = tl.cast(0.0, tl.float32)
    final_token = vocab_size - 1
    found = 0

    for v_start in range(0, vocab_size, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < vocab_size
            p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
            if all_drafts_accepted:
                val = p_val
            else:
                q_val = tl.load(
                    dp_base_ptr + v_offsets * stride_dp_v, mask=mask, other=0.0
                )
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            total_cumsum = cum_sum + tl.cumsum(val, axis=0)
            hits = total_cumsum > target_u
            if tl.max(hits, axis=0):
                final_token = v_start + tl.argmax(hits.to(tl.int32), axis=0)
                found = 1
            cum_sum += tl.sum(val)

    tl.store(predicts_ptr + last_accept_retrieve_idx, final_token)


@triton.jit
def verify_tree_greedy_kernel_triton(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    target_predict_ptr,
    batch_size: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    num_draft_tokens: tl.constexpr,
):
    """
    Triton kernel for verifying EAGLE tree in greedy mode.
    Each program handles one batch item.
    """
    bx = tl.program_id(0)

    # Initialize
    last_accept_retrieve_idx = tl.load(retrieve_index_ptr + bx * num_draft_tokens)
    tl.store(accept_index_ptr + bx * num_speculative_tokens, last_accept_retrieve_idx)
    # Cast to match dtype of loaded tensors to avoid type inconsistency
    num_accept_tokens = tl.cast(0, last_accept_retrieve_idx.dtype)
    cur_index = tl.cast(0, last_accept_retrieve_idx.dtype)

    # Tree traversal loop
    should_continue = 1
    for j in range(1, num_speculative_tokens):
        if should_continue:  # Early exit guard
            cur_index = tl.load(
                retrieve_next_token_ptr + bx * num_draft_tokens + cur_index
            )

            # Load target token once per level (before sibling search)
            # last_accept_retrieve_idx is constant during sibling traversal
            target_row = last_accept_retrieve_idx // num_draft_tokens
            target_col = last_accept_retrieve_idx % num_draft_tokens
            target_token = tl.load(
                target_predict_ptr + target_row * num_draft_tokens + target_col
            )

            # Traverse siblings
            found_match = 0
            for _ in range(num_draft_tokens):  # Max iterations = num_draft_tokens
                if found_match == 0:  # Early exit guard
                    # Check if we've reached end of sibling list
                    is_valid = cur_index != -1

                    # Use masked loads with safe address (0 when invalid)
                    safe_cur_index = (
                        cur_index * is_valid
                    )  # 0 if invalid, cur_index if valid
                    safe_index = bx * num_draft_tokens + safe_cur_index

                    # Load draft token info (loads from index 0 when invalid, but we won't use it)
                    draft_index = tl.load(retrieve_index_ptr + safe_index)
                    draft_token = tl.load(candidates_ptr + safe_index)

                    # Check for token match (only valid when is_valid is True)
                    token_match = is_valid & (draft_token == target_token)

                    # Accept token using predicated stores (only write if matched)
                    tl.store(
                        predicts_ptr + last_accept_retrieve_idx,
                        target_token,
                        mask=token_match,
                    )
                    next_num_accept_tokens = num_accept_tokens + 1
                    tl.store(
                        accept_index_ptr
                        + bx * num_speculative_tokens
                        + next_num_accept_tokens,
                        draft_index,
                        mask=token_match,
                    )

                    num_accept_tokens = num_accept_tokens + token_match
                    last_accept_retrieve_idx = (
                        token_match * draft_index
                        + (~token_match) * last_accept_retrieve_idx
                    )
                    found_match = token_match * 1 + (~is_valid) * (-1)

                    # Masked load: only load next sibling when no match (hardware predication)
                    # When matched: returns cur_index (other); when not matched: loads sibling
                    cur_index = tl.load(
                        retrieve_next_sibling_ptr + safe_index,
                        mask=~token_match
                        & is_valid,  # Only load when valid and NOT matched
                        other=cur_index,  # Keep cur_index when matched or invalid
                    )

            if found_match != 1:
                should_continue = 0

    # Store final results
    tl.store(accept_token_num_ptr + bx, num_accept_tokens)

    target_row = last_accept_retrieve_idx // num_draft_tokens
    target_col = last_accept_retrieve_idx % num_draft_tokens
    final_target = tl.load(
        target_predict_ptr + target_row * num_draft_tokens + target_col
    )
    tl.store(predicts_ptr + last_accept_retrieve_idx, final_target)
