import torch
import triton
import triton.language as tl


@triton.jit
def _chain_speculative_sampling_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrieve_index_ptr,
    uniform_samples_ptr,
    uniform_final_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uniform_b,
    stride_uniform_s,
    stride_target_b,
    stride_target_s,
    stride_target_v,
    stride_draft_b,
    stride_draft_s,
    stride_draft_v,
    num_slots: tl.constexpr,
    vocab_size: tl.constexpr,
    block_v: tl.constexpr,
):
    """Lossless classic speculative sampling for one linear draft chain."""
    batch = tl.program_id(0)
    current_prob_row = 0

    candidate_base = candidates_ptr + batch * stride_cand_b
    retrieve_base = retrieve_index_ptr + batch * stride_idx_b
    uniform_base = uniform_samples_ptr + batch * stride_uniform_b

    root_index = tl.load(retrieve_base)
    tl.store(accept_index_ptr + batch * stride_idx_b, root_index)
    last_accepted_index = root_index
    num_accepted = 0

    slot = 1
    keep_verifying = 1
    while (slot < num_slots) and (keep_verifying == 1):
        draft_token = tl.load(candidate_base + slot * stride_cand_s)
        target_offset = (
            batch * stride_target_b
            + current_prob_row * stride_target_s
            + draft_token * stride_target_v
        )
        draft_offset = (
            batch * stride_draft_b
            + current_prob_row * stride_draft_s
            + draft_token * stride_draft_v
        )
        target_probability = tl.load(target_probs_ptr + target_offset)
        draft_probability = tl.load(draft_probs_ptr + draft_offset)
        coin = tl.load(uniform_base + (slot - 1) * stride_uniform_s)

        if coin * draft_probability < target_probability:
            num_accepted += 1
            current_prob_row = slot
            tl.store(predicts_ptr + last_accepted_index, draft_token)
            current_index = tl.load(retrieve_base + slot * stride_idx_s)
            tl.store(
                accept_index_ptr + batch * stride_idx_b + num_accepted * stride_idx_s,
                current_index,
            )
            last_accepted_index = current_index
            slot += 1
        else:
            keep_verifying = 0

    tl.store(accept_token_num_ptr + batch, num_accepted)

    # If every draft is accepted, sample the bonus from target p. Otherwise,
    # sample from normalized relu(p-q), as required by lossless verification.
    all_drafts_accepted = keep_verifying
    target_base = (
        target_probs_ptr + batch * stride_target_b + current_prob_row * stride_target_s
    )
    # On the all-accepted path current_prob_row is the target-only bonus row.
    # This draft pointer is out of range conceptually, but that branch never loads it.
    draft_base = (
        draft_probs_ptr + batch * stride_draft_b + current_prob_row * stride_draft_s
    )

    normalizer = 0.0
    for vocab_start in range(0, vocab_size, block_v):
        offsets = vocab_start + tl.arange(0, block_v)
        mask = offsets < vocab_size
        target_probability = tl.load(
            target_base + offsets * stride_target_v, mask=mask, other=0.0
        )
        if all_drafts_accepted:
            residual = target_probability
        else:
            draft_probability = tl.load(
                draft_base + offsets * stride_draft_v, mask=mask, other=0.0
            )
            draft_probability = tl.where(
                draft_probability == draft_probability, draft_probability, 0.0
            )
            residual = tl.maximum(target_probability - draft_probability, 0.0)
        normalizer += tl.sum(residual)

    target_uniform = tl.load(uniform_final_ptr + batch) * normalizer
    cumulative = 0.0
    final_token = vocab_size - 1
    found = 0
    for vocab_start in range(0, vocab_size, block_v):
        if found == 0:
            offsets = vocab_start + tl.arange(0, block_v)
            mask = offsets < vocab_size
            target_probability = tl.load(
                target_base + offsets * stride_target_v, mask=mask, other=0.0
            )
            if all_drafts_accepted:
                residual = target_probability
            else:
                draft_probability = tl.load(
                    draft_base + offsets * stride_draft_v, mask=mask, other=0.0
                )
                draft_probability = tl.where(
                    draft_probability == draft_probability, draft_probability, 0.0
                )
                residual = tl.maximum(target_probability - draft_probability, 0.0)

            cdf = cumulative + tl.cumsum(residual, axis=0)
            matches = cdf > target_uniform
            if tl.max(matches, axis=0):
                final_token = vocab_start + tl.argmax(matches.to(tl.int32), axis=0)
                found = 1
            cumulative += tl.sum(residual)

    tl.store(predicts_ptr + last_accepted_index, final_token)


def chain_speculative_sampling_triton(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> None:
    """Verify a DFlash2 path with its selector distribution in one program/request."""
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    _chain_speculative_sampling_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrieve_index.stride(0),
        retrieve_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        num_slots=num_slots,
        vocab_size=vocab_size,
        block_v=4096,
    )
