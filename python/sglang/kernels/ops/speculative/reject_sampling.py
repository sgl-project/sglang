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
            q_val = tl.load(q_ptr, mask=mask, other=0.0)
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
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
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
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]

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
    )


@triton.jit
def _topk_support_probs(
    TopkLogits,
    row,
    stride_row,
    offs,
    k_mask,
    temperature,
    top_k,
    top_p,
    USE_TOP_P: tl.constexpr,
):
    # Same kept set as softmax -> top_k_renorm_probs -> top_p_renorm_probs on the full row:
    # logits are sorted descending, top-k keeps ties with the k-th logit, top-p keeps ties with its pivot.
    z = tl.load(TopkLogits + row * stride_row + offs, mask=k_mask, other=float("-inf"))
    z = z.to(tl.float32) / temperature
    z_k = tl.max(tl.where(offs == top_k - 1, z, float("-inf")))
    e = tl.where(z >= z_k, tl.exp(z - tl.max(z)), 0.0)
    probs = e / tl.sum(e)
    if USE_TOP_P:
        # pivot = first probability at which the running sum reaches top_p (0 if never: keep all)
        reached = tl.cumsum(probs, axis=0) >= top_p
        first = tl.min(tl.where(reached, offs, 2147483647))
        pivot = tl.sum(tl.where(offs == first, probs, 0.0))
        probs = tl.where(probs >= pivot, probs, 0.0)
        probs = probs / tl.sum(probs)
    return probs


@triton.jit
def speculative_sampling_topk_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TopkLogits,
    TopkIds,
    Temperatures,
    TopKs,
    TopPs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tl_row,
    stride_ti_row,
    stride_temp,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    support_width,
    # Constants
    NUM_SLOTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TOP_P: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0
    offs = tl.arange(0, BLOCK_K)
    k_mask = offs < support_width

    temperature = tl.load(Temperatures + pid * stride_temp).to(tl.float32)
    top_k = tl.load(TopKs + pid)
    top_p = 1.0
    if USE_TOP_P:
        top_p = tl.load(TopPs + pid)

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop: same accept rule as speculative_sampling_classic_kernel,
    # with p = 0 off the row's top-k support.
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        row = pid * NUM_SLOTS + cur_prob_row
        ids = tl.load(TopkIds + row * stride_ti_row + offs, mask=k_mask, other=-1)
        probs = _topk_support_probs(
            TopkLogits,
            row,
            stride_tl_row,
            offs,
            k_mask,
            temperature,
            top_k,
            top_p,
            USE_TOP_P,
        )
        p = tl.sum(tl.where(ids == draft_token, probs, 0.0))
        q = tl.load(
            DraftProbs
            + pid * stride_dp_b
            + cur_prob_row * stride_dp_s
            + draft_token * stride_dp_v
        )

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

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

    # Final Sampling: max(p - q, 0) is 0 off the support, so the support is enough.
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)

    row = pid * NUM_SLOTS + cur_prob_row
    ids = tl.load(TopkIds + row * stride_ti_row + offs, mask=k_mask, other=-1)
    p_val = _topk_support_probs(
        TopkLogits,
        row,
        stride_tl_row,
        offs,
        k_mask,
        temperature,
        top_k,
        top_p,
        USE_TOP_P,
    )
    if all_drafts_accepted:
        val = p_val
    else:
        # DraftProbs has num_steps rows; on rejection cur_prob_row <= num_steps - 1.
        q_val = tl.load(
            DraftProbs
            + pid * stride_dp_b
            + cur_prob_row * stride_dp_s
            + ids * stride_dp_v,
            mask=k_mask,
            other=0.0,
        )
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        diff = p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    norm_sum = tl.sum(val)
    # A degenerate residual (q >= p on the whole support) falls back to p.
    if norm_sum <= 0.0:
        val = p_val
        norm_sum = tl.sum(val)

    # Walk the CDF in token-id order, the order the dense kernel scans the vocab:
    # sort (id << 32 | bits of val); val >= 0, so its bits never reach the id half.
    ids = tl.where(k_mask, ids, 2147483647).to(tl.int64)
    keys = tl.sort((ids << 32) | val.to(tl.int32, bitcast=True).to(tl.int64))
    sorted_ids = (keys >> 32).to(tl.int32)
    sorted_val = (keys & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    cdf = tl.cumsum(sorted_val, axis=0)
    target_u = coin_final * norm_sum
    final_token = tl.min(tl.where(cdf > target_u, sorted_ids, 2147483647))
    if final_token == 2147483647:
        # Rounding left target_u at the top of the CDF: take the last token with mass.
        final_token = tl.max(tl.where(sorted_val > 0.0, sorted_ids, -1))

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_topk_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    topk_logits,
    topk_ids,
    temperatures,
    top_ks,
    top_ps,
    draft_probs,
):
    """Chain rejection sampling with the target restricted to each row's top-k.

    topk_logits / topk_ids: [bs * num_slots, width], each row sorted by
    descending logit, width from topk_support_width(max top_k, vocab).
    temperatures: [bs, 1]; top_ks: [bs]; top_ps: [bs], or None to skip top-p.
    draft_probs: [bs, num_slots - 1, vocab], the dense draft proposal q.
    Same accept and resample rule as chain_speculative_sampling_triton on
    softmax -> top_k_renorm -> top_p_renorm target probabilities.
    """
    batch_size, num_slots = candidates.shape
    support_width = topk_logits.shape[-1]
    assert topk_logits.stride(-1) == 1 and topk_ids.stride(-1) == 1

    grid = (batch_size,)
    speculative_sampling_topk_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        topk_logits,
        topk_ids,
        temperatures,
        top_ks,
        top_ps if top_ps is not None else temperatures,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        topk_logits.stride(0),
        topk_ids.stride(0),
        temperatures.stride(0),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        support_width,
        NUM_SLOTS=num_slots,
        BLOCK_K=triton.next_power_of_2(support_width),
        USE_TOP_P=top_ps is not None,
    )


def topk_support_width(max_top_k: int, vocab_size: int) -> int:
    """Top-k width to fetch for chain_speculative_sampling_topk_triton.

    top_k_renorm_probs keeps every logit tied with the k-th one, and bf16 logits tie
    often; the extra room keeps those ties (16 is arbitrary, ties that long are rare).
    """
    return min(triton.next_power_of_2(max_top_k + 16), vocab_size)
