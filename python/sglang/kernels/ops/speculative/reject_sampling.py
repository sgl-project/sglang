import triton
import triton.language as tl


@triton.jit
def _draft_probability(value, row_max, inv_sum, temperature, FROM_LOGITS: tl.constexpr):
    if FROM_LOGITS:
        value = tl.exp(value.to(tl.float32) / temperature - row_max) * inv_sum
    return value


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
    DraftProbs,  # Probabilities, or pre-temperature logits when DRAFT_FROM_LOGITS.
    DraftSoftmaxStats,
    DraftTemperatures,
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
    NUM_DRAFT_STEPS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    DRAFT_FROM_LOGITS: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0
    temperature = 1.0
    draft_max = 0.0
    draft_inv_sum = 1.0
    if DRAFT_FROM_LOGITS:
        temperature = tl.load(DraftTemperatures + pid).to(tl.float32)

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = tl.load(DraftProbs + offset_draft)
        if DRAFT_FROM_LOGITS:
            stats_offset = (pid.to(tl.int64) * NUM_DRAFT_STEPS + cur_prob_row) * 2
            draft_max = tl.load(DraftSoftmaxStats + stats_offset)
            draft_inv_sum = tl.load(DraftSoftmaxStats + stats_offset + 1)
        q = _draft_probability(
            q, draft_max, draft_inv_sum, temperature, DRAFT_FROM_LOGITS
        )

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        # X was sampled from q, so q(X) has to be a positive probability.
        # Anything else means this row is not the distribution X came from, and
        # `coin * q < p` would then accept unconditionally -- -inf < p for an
        # -inf q, 0 < p for a zero one, and the range guard the residual passes
        # use lets zero through. Reject instead: the residual path resamples
        # from the target, which is the safe direction to fail in.
        q_is_prob = (q > 0.0) & (q <= 1.0)

        if q_is_prob & (coin * q < p):
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
            q_val = tl.load(
                q_ptr, mask=mask, other=float("-inf") if DRAFT_FROM_LOGITS else 0.0
            )
            q_val = _draft_probability(
                q_val, draft_max, draft_inv_sum, temperature, DRAFT_FROM_LOGITS
            )
            # Treat any non-probability q (NaN, +-inf, negative) as 0: the
            # residual falls back to p. A comparison against NaN is false, so
            # the range test rejects it along with the infinities.
            q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
            diff = p_val - q_val
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
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(
                    q_ptr, mask=mask, other=float("-inf") if DRAFT_FROM_LOGITS else 0.0
                )
                q_val = _draft_probability(
                    q_val, draft_max, draft_inv_sum, temperature, DRAFT_FROM_LOGITS
                )
                # Same guard as pass 1.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = p_val - q_val
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
    draft_logits=None,
    draft_softmax_stats=None,
    draft_temperatures=None,
):
    """Verify using dense q or logits with per-row (scaled max, inverse sum)."""
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    from_logits = draft_logits is not None
    draft_values = draft_logits if from_logits else draft_probs
    if from_logits:
        assert draft_probs is None
        assert draft_softmax_stats is not None and draft_temperatures is not None
        assert draft_softmax_stats.shape == (*draft_logits.shape[:2], 2)
        assert draft_softmax_stats.is_contiguous()
        assert draft_temperatures.numel() == batch_size
        assert draft_temperatures.is_contiguous()
    else:
        assert draft_probs is not None
        assert draft_softmax_stats is None and draft_temperatures is None

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
        draft_values,
        draft_softmax_stats,
        draft_temperatures,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_values.stride(0),
        draft_values.stride(1),
        draft_values.stride(2),
        NUM_SLOTS=num_slots,
        NUM_DRAFT_STEPS=draft_values.shape[1],
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        DRAFT_FROM_LOGITS=from_logits,
        # Match the separately rounded scaling in the normalization kernel.
        enable_fp_fusion=not from_logits,
    )
