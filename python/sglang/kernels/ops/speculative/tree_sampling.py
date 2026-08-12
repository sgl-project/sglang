import torch
import triton
import triton.language as tl


@triton.jit
def _tree_speculative_sampling_target_only_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetrieveIndex,
    RetrieveNextToken,
    RetrieveNextSibling,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    stride_predict,
    stride_accept_b,
    stride_accept_s,
    stride_accept_num,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_next_b,
    stride_next_s,
    stride_sibling_b,
    stride_sibling_s,
    stride_uni_b,
    stride_uni_s,
    stride_uni_final,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    threshold_single,
    threshold_acc,
    MAX_TREE_DEPTH: tl.constexpr,
    NUM_DRAFT_TOKENS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    request_idx = tl.program_id(0)

    candidate_base = Candidates + request_idx * stride_cand_b
    retrieve_base = RetrieveIndex + request_idx * stride_idx_b
    next_token_base = RetrieveNextToken + request_idx * stride_next_b
    next_sibling_base = RetrieveNextSibling + request_idx * stride_sibling_b
    uniform_base = UniformSamples + request_idx * stride_uni_b

    current_tree_idx = tl.cast(0, tl.int64)
    current_prob_row = tl.cast(0, tl.int64)
    last_accept_global_idx = tl.load(retrieve_base)
    tl.store(
        AcceptIndex + request_idx * stride_accept_b,
        last_accept_global_idx,
    )

    num_correct_drafts = tl.cast(0, tl.int32)
    coin = tl.load(uniform_base)
    continue_tree = 1
    safe_threshold_acc = tl.maximum(threshold_acc, 1.0e-9)

    for _ in range(1, MAX_TREE_DEPTH):
        if continue_tree:
            current_tree_idx = tl.load(
                next_token_base + current_tree_idx * stride_next_s
            )
            sibling_prob_acc = 0.0
            sibling_status = 0

            for _ in range(NUM_DRAFT_TOKENS):
                if sibling_status == 0:
                    is_valid = current_tree_idx != -1
                    safe_tree_idx = tl.where(is_valid, current_tree_idx, 0)

                    draft_token = tl.load(
                        candidate_base + safe_tree_idx * stride_cand_s
                    )
                    draft_global_idx = tl.load(
                        retrieve_base + safe_tree_idx * stride_idx_s
                    )
                    target_prob_offset = (
                        request_idx * stride_tp_b
                        + current_prob_row * stride_tp_s
                        + draft_token * stride_tp_v
                    )
                    target_prob = tl.load(
                        TargetProbs + target_prob_offset,
                        mask=is_valid,
                        other=0.0,
                    )
                    next_prob_acc = sibling_prob_acc + target_prob
                    accept_draft = is_valid & (
                        (coin <= next_prob_acc / safe_threshold_acc)
                        | (target_prob >= threshold_single)
                    )
                    reject_draft = is_valid & ~accept_draft

                    # Match the target-only CUDA kernel: rejected sibling mass is
                    # removed from the residual distribution in-place.
                    draft_prob_offset = (
                        request_idx * stride_dp_b
                        + current_prob_row * stride_dp_s
                        + draft_token * stride_dp_v
                    )
                    tl.store(
                        DraftProbs + draft_prob_offset,
                        target_prob,
                        mask=reject_draft,
                    )

                    next_num_correct_drafts = num_correct_drafts + 1
                    tl.store(
                        Predicts + last_accept_global_idx * stride_predict,
                        draft_token,
                        mask=accept_draft,
                    )
                    tl.store(
                        AcceptIndex
                        + request_idx * stride_accept_b
                        + next_num_correct_drafts * stride_accept_s,
                        draft_global_idx,
                        mask=accept_draft,
                    )

                    num_correct_drafts = tl.where(
                        accept_draft,
                        next_num_correct_drafts,
                        num_correct_drafts,
                    )
                    last_accept_global_idx = tl.where(
                        accept_draft,
                        draft_global_idx,
                        last_accept_global_idx,
                    )
                    current_prob_row = tl.where(
                        accept_draft,
                        safe_tree_idx,
                        current_prob_row,
                    )
                    coin = tl.where(
                        accept_draft,
                        tl.load(
                            uniform_base + safe_tree_idx * stride_uni_s,
                            mask=is_valid,
                            other=0.0,
                        ),
                        coin,
                    )
                    sibling_prob_acc = tl.where(
                        accept_draft,
                        0.0,
                        next_prob_acc,
                    )
                    sibling_status = tl.where(
                        accept_draft,
                        1,
                        tl.where(is_valid, 0, -1),
                    )
                    current_tree_idx = tl.load(
                        next_sibling_base + safe_tree_idx * stride_sibling_s,
                        mask=reject_draft,
                        other=current_tree_idx,
                    )

            if sibling_status != 1:
                continue_tree = 0

    tl.store(
        AcceptTokenNum + request_idx * stride_accept_num,
        num_correct_drafts,
    )

    all_drafts_accept = num_correct_drafts == MAX_TREE_DEPTH - 1
    target_row = (
        TargetProbs + request_idx * stride_tp_b + current_prob_row * stride_tp_s
    )
    draft_row = DraftProbs + request_idx * stride_dp_b + current_prob_row * stride_dp_s

    residual_sum = 0.0
    for vocab_start in range(0, VOCAB_SIZE, BLOCK_V):
        vocab_offsets = vocab_start + tl.arange(0, BLOCK_V)
        mask = vocab_offsets < VOCAB_SIZE
        target = tl.load(
            target_row + vocab_offsets * stride_tp_v,
            mask=mask,
            other=0.0,
        )
        draft = tl.load(
            draft_row + vocab_offsets * stride_dp_v,
            mask=mask & ~all_drafts_accept,
            other=0.0,
        )
        residual = tl.where(
            all_drafts_accept,
            target,
            tl.maximum(target - draft, 0.0),
        )
        residual_sum += tl.sum(residual)

    target_cdf = (
        tl.load(UniformSamplesFinal + request_idx * stride_uni_final) * residual_sum
    )
    cumulative_sum = 0.0
    bonus_token = VOCAB_SIZE - 1
    last_valid_token = -1
    found_bonus = 0

    for vocab_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found_bonus == 0:
            vocab_offsets = vocab_start + tl.arange(0, BLOCK_V)
            mask = vocab_offsets < VOCAB_SIZE
            target = tl.load(
                target_row + vocab_offsets * stride_tp_v,
                mask=mask,
                other=0.0,
            )
            draft = tl.load(
                draft_row + vocab_offsets * stride_dp_v,
                mask=mask & ~all_drafts_accept,
                other=0.0,
            )
            residual = tl.where(
                all_drafts_accept,
                target,
                tl.maximum(target - draft, 0.0),
            )

            positive = mask & (residual > 0.0)
            block_last_valid = tl.max(
                tl.where(positive, vocab_offsets, -1),
                axis=0,
            )
            last_valid_token = tl.maximum(last_valid_token, block_last_valid)

            block_cdf = cumulative_sum + tl.cumsum(residual, axis=0)
            crosses_target = positive & (block_cdf > target_cdf)
            has_match = tl.max(crosses_target, axis=0)
            if has_match:
                match_idx = tl.argmax(crosses_target.to(tl.int32), axis=0)
                bonus_token = vocab_start + match_idx
                found_bonus = 1

            cumulative_sum += tl.sum(residual)

    bonus_token = tl.where(
        (found_bonus == 0) & (last_valid_token >= 0),
        last_valid_token,
        bonus_token,
    )
    tl.store(Predicts + last_accept_global_idx * stride_predict, bonus_token)


def tree_speculative_sampling_target_only_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Triton port of sgl_kernel's target-only tree verifier.

    The misspelled ``retrive_*`` kwargs and ``accept_token_num`` are retained at
    this compatibility boundary to match the frozen sgl_kernel schema.

    Tree construction guarantees that candidates are valid vocabulary indices,
    retrieve indices address ``predicts``, and child/sibling links are either
    ``-1`` or valid tree slots. These topology bounds are kernel preconditions;
    checking them here would introduce a device synchronization on every verify.
    """
    # Retained for API parity. The fixed Triton reduction/scan order is
    # repeatable on one backend, but is not bitwise-equivalent to FlashInfer's
    # deterministic scan at floating-point CDF boundaries.
    del deterministic

    assert predicts.ndim == 1
    assert accept_index.ndim == 2
    assert accept_token_num.ndim == 1
    assert candidates.ndim == 2
    assert retrive_index.ndim == 2
    assert retrive_next_token.ndim == 2
    assert retrive_next_sibling.ndim == 2
    assert uniform_samples.ndim == 2
    assert uniform_samples_for_final_sampling.ndim == 1
    assert target_probs.ndim == 3
    assert draft_probs.ndim == 3
    assert predicts.dtype == torch.int32
    assert accept_index.dtype == torch.int32
    assert accept_token_num.dtype == torch.int32
    assert candidates.dtype == torch.int64
    assert retrive_index.dtype == torch.int64
    assert retrive_next_token.dtype == torch.int64
    assert retrive_next_sibling.dtype == torch.int64
    assert uniform_samples.dtype == torch.float32
    assert uniform_samples_for_final_sampling.dtype == torch.float32
    assert target_probs.dtype == torch.float32
    assert draft_probs.dtype == torch.float32

    batch_size, num_draft_tokens = candidates.shape
    max_tree_depth = accept_index.shape[1]
    vocab_size = target_probs.shape[2]
    if batch_size == 0:
        return

    assert num_draft_tokens > 0
    assert 1 <= max_tree_depth <= num_draft_tokens
    assert vocab_size > 0
    assert accept_index.shape[0] == batch_size
    assert accept_token_num.shape == (batch_size,)
    assert retrive_index.shape == candidates.shape
    assert retrive_next_token.shape == candidates.shape
    assert retrive_next_sibling.shape == candidates.shape
    assert uniform_samples.shape == candidates.shape
    assert uniform_samples_for_final_sampling.shape == (batch_size,)
    assert target_probs.shape[:2] == candidates.shape
    assert draft_probs.shape == target_probs.shape
    assert predicts.numel() >= batch_size * num_draft_tokens
    assert 0.0 <= threshold_single <= 1.0
    assert 0.0 <= threshold_acc <= 1.0

    device = target_probs.device
    tensors = (
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        draft_probs,
    )
    assert all(tensor.device == device for tensor in tensors)

    _tree_speculative_sampling_target_only_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        predicts.stride(0),
        accept_index.stride(0),
        accept_index.stride(1),
        accept_token_num.stride(0),
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        retrive_next_token.stride(0),
        retrive_next_token.stride(1),
        retrive_next_sibling.stride(0),
        retrive_next_sibling.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        uniform_samples_for_final_sampling.stride(0),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        threshold_single,
        threshold_acc,
        MAX_TREE_DEPTH=max_tree_depth,
        NUM_DRAFT_TOKENS=num_draft_tokens,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
    )
