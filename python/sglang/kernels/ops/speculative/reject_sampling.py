import triton
import triton.language as tl


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_VERIFICATION: tl.constexpr = False,
    BLOCK_REDUCE: tl.constexpr = 4096,
    # The draft is a point mass on each proposed token (a top-1 chain); DraftProbs is unread.
    ONE_HOT_DRAFT: tl.constexpr = False,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    residual_scale = 1.0
    norm_sum = -1.0

    if BLOCK_VERIFICATION:
        # Algorithm 2: https://arxiv.org/abs/2403.10444
        # Keep the short recurrence in registers; only the longest accepted prefix matters.
        prefix_prob = 1.0
        prefix_probs = ()
        for step in tl.static_range(1, NUM_SLOTS):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            p = tl.load(
                TargetProbs
                + pid * stride_tp_b
                + (step - 1) * stride_tp_s
                + draft_token * stride_tp_v
            )
            q = tl.load(
                DraftProbs
                + pid * stride_dp_b
                + (step - 1) * stride_dp_s
                + draft_token * stride_dp_v
            )
            q_is_prob = (q > 0.0) & (q <= 1.0)
            prefix_prob = tl.where(q_is_prob, tl.minimum(prefix_prob * p / q, 1.0), 0.0)
            prefix_probs += (prefix_prob,)

        for step in tl.static_range(NUM_SLOTS - 1, 0, -1):
            if num_accept == 0:
                prefix_prob = prefix_probs[step - 1]
                coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
                # For normalized distributions h_i <= r_i; reject before scanning.
                if coin < prefix_prob:
                    accept_prob = prefix_prob
                    residual_mass = -1.0
                    if (
                        (step < NUM_SLOTS - 1)
                        & (prefix_prob > 0.0)
                        & (prefix_prob < 1.0)
                    ):
                        residual_mass = 0.0
                        for v_start in range(0, VOCAB_SIZE, BLOCK_REDUCE):
                            v_offsets = v_start + tl.arange(0, BLOCK_REDUCE)
                            mask = v_offsets < VOCAB_SIZE
                            p_next = tl.load(
                                TargetProbs
                                + pid * stride_tp_b
                                + step * stride_tp_s
                                + v_offsets * stride_tp_v,
                                mask=mask,
                                other=0.0,
                            )
                            q_next = tl.load(
                                DraftProbs
                                + pid * stride_dp_b
                                + step * stride_dp_s
                                + v_offsets * stride_dp_v,
                                mask=mask,
                                other=0.0,
                            )
                            residual_mass += tl.sum(
                                tl.maximum(prefix_prob * p_next - q_next, 0.0)
                            )
                        accept_prob = residual_mass / (
                            residual_mass + 1.0 - prefix_prob
                        )
                    if coin < accept_prob:
                        num_accept = step
                        residual_scale = prefix_prob
                        norm_sum = residual_mass

        cur_prob_row = num_accept
        for step in range(1, num_accept + 1):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + step * stride_idx_s,
                last_accepted_global_idx,
            )
    else:
        step = 1
        continue_verifying = 1
        # At each token-verification loop entry, cur_prob_row == num_accept == step - 1.
        while (step < NUM_SLOTS) and (continue_verifying == 1):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            p = tl.load(
                TargetProbs
                + pid * stride_tp_b
                + cur_prob_row * stride_tp_s
                + draft_token * stride_tp_v
            )
            if ONE_HOT_DRAFT:
                q = 1.0
            else:
                q = tl.load(
                    DraftProbs
                    + pid * stride_dp_b
                    + cur_prob_row * stride_dp_s
                    + draft_token * stride_dp_v
                )
            coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
            # A proposal token must have positive probability under its draft distribution.
            q_is_prob = (q > 0.0) & (q <= 1.0)
            if q_is_prob & (coin * q < p):
                num_accept += 1
                cur_prob_row = step
                tl.store(Predicts + last_accepted_global_idx, draft_token)
                last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
                tl.store(
                    AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                    last_accepted_global_idx,
                )
                step += 1
            else:
                continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    coin_final = tl.load(UniformSamplesFinal + pid)

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)
    # On rejection the residual removes the rejected draft token (slot cur_prob_row + 1).
    rejected_token = tl.load(
        cand_ptr_base + tl.minimum(cur_prob_row + 1, NUM_SLOTS - 1) * stride_cand_s
    )

    # Reuse the selected prefix mass as the correction normalizer.
    if norm_sum < 0.0:
        norm_sum = 0.0
        # Pass 1: Sum
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                if ONE_HOT_DRAFT:
                    q_val = (v_offsets == rejected_token).to(tl.float32)
                else:
                    q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                    q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Treat any non-probability q (NaN, +-inf, negative) as 0: the
                # residual falls back to p. A comparison against NaN is false, so
                # the range test rejects it along with the infinities.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = residual_scale * p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            norm_sum += tl.sum(val)

    # Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on
    # rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to
    # VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                if ONE_HOT_DRAFT:
                    q_val = (v_offsets == rejected_token).to(tl.float32)
                else:
                    q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                    q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Same guard as pass 1.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = residual_scale * p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
    *,
    block_verification: bool = False,
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    # Keep CDF tiles small, and cap multi-tile reductions to avoid register spills.
    block_reduce = min(triton.next_power_of_2(vocab_size), 32768)
    num_warps = 8 if block_verification and block_reduce > 4096 else 4

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        BLOCK_VERIFICATION=block_verification,
        BLOCK_REDUCE=block_reduce,
        num_warps=num_warps,
    )


def chain_speculative_sampling_target_only_triton(
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
    draft_probs,  # not used: the draft is a point mass on each proposed token
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    """Lossless sampled verification of a top-1 draft chain against target probs.

    A deterministic draft proposes each token with probability 1, so classic
    speculative sampling with a one-hot proposal accepts token ``t`` with
    probability ``p(t)`` and resamples a rejection from ``p`` with ``t`` removed.
    Needs no draft distribution, so it also serves reduced-vocab drafts.
    """
    if threshold_single != 1.0 or threshold_acc != 1.0:
        raise ValueError(
            "Target-only chain sampling supports only exact verification "
            "(speculative accept thresholds of 1.0)."
        )
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    speculative_sampling_classic_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        target_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        BLOCK_REDUCE=min(triton.next_power_of_2(vocab_size), 32768),
        ONE_HOT_DRAFT=True,
        num_warps=4,
    )
