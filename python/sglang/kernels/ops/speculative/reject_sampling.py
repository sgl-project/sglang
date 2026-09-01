import triton
import triton.language as tl


@triton.jit
def _vocab_scan_residual_mass(
    TargetProbsRow,
    DraftProbsRow,
    scale,
    stride_tp_v,
    stride_dp_v,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Row-wise sum of max(scale * M_b(x) - M_s(x), 0); NaN q treated as 0."""
    acc = 0.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        t_val = tl.load(TargetProbsRow + v_offsets * stride_tp_v, mask=mask, other=0.0)
        d_val = tl.load(DraftProbsRow + v_offsets * stride_dp_v, mask=mask, other=0.0)
        d_val = tl.where(d_val == d_val, d_val, 0.0)
        diff = scale * t_val - d_val
        acc += tl.sum(tl.where(diff > 0.0, diff, 0.0))
    return acc


@triton.jit
def _vocab_scan_target_mass(
    TargetProbsRow,
    stride_tp_v,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Sum over the vocab of a target-prob row (draft-free variant)."""
    acc = 0.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        t_val = tl.load(TargetProbsRow + v_offsets * stride_tp_v, mask=mask, other=0.0)
        acc += tl.sum(t_val)
    return acc


@triton.jit
def speculative_sampling_block_kernel(
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
    """Block verification (arXiv:2403.10444, Algorithm 2).

    Per-request kernel over a linear (topk=1) chain of gamma = NUM_SLOTS - 1
    drafted tokens; same tensor contract as speculative_sampling_classic_kernel.
    Accepts via h_i = Z_{i+1} / (Z_{i+1} + 1 - p_i) with cumulative prefix
    ratio p_i and residual mass Z_{i+1}; tau = argmax_i{coin_i <= h_i}
    (no early stop), then resamples from the p_tau-scaled residual.
    One-hot draft rows use Z_{i+1} = p_i * (1 - M_b(X_{i+1})) in closed form.
    """
    pid = tl.program_id(0)

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)

    tau = 0
    p = 1.0
    # Residual snapshot at tau: z_res = Z_{tau+1}, p_res = p_tau.
    z_res = 0.0
    p_res = 1.0

    for step in range(1, NUM_SLOTS):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        t_prob = tl.load(
            TargetProbs
            + (pid * stride_tp_b)
            + ((step - 1) * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        d_prob = tl.load(
            DraftProbs
            + (pid * stride_dp_b)
            + ((step - 1) * stride_dp_s)
            + (draft_token * stride_dp_v)
        )
        # q=0 or NaN mirrors the classic kernel: q=0 & p>0 accepts, q=0 & p=0
        # hard-rejects.
        d_ok = (d_prob == d_prob) & (d_prob > 0.0)
        ratio = tl.where(
            d_ok, t_prob / tl.where(d_ok, d_prob, 1.0), tl.where(t_prob > 0.0, 1.0, 0.0)
        )
        p = tl.minimum(p * ratio, 1.0)

        # Z_{step+1}: feeds h_step (step < gamma) and the residual normalizer
        # if tau lands here.
        z = 0.0
        if (step < NUM_SLOTS - 1) & (p > 0.0):
            next_token = tl.load(cand_ptr_base + (step + 1) * stride_cand_s)
            next_draft_prob = tl.load(
                DraftProbs
                + (pid * stride_dp_b)
                + (step * stride_dp_s)
                + (next_token * stride_dp_v)
            )
            if next_draft_prob >= 1.0:
                # One-hot draft row: Z_{step+1} = p * (1 - M_b(X_{step+1})).
                next_target_prob = tl.load(
                    TargetProbs
                    + (pid * stride_tp_b)
                    + (step * stride_tp_s)
                    + (next_token * stride_tp_v)
                )
                # Clamp: 1 - M_b(x) can round slightly below 0 after
                # top-k/top-p renorm; Z is a sum of non-negative terms.
                z = tl.maximum(p * (1.0 - next_target_prob), 0.0)
            else:
                z = _vocab_scan_residual_mass(
                    TargetProbs + (pid * stride_tp_b) + (step * stride_tp_s),
                    DraftProbs + (pid * stride_dp_b) + (step * stride_dp_s),
                    p,
                    stride_tp_v,
                    stride_dp_v,
                    VOCAB_SIZE,
                    BLOCK_V,
                )

        # h_gamma = p_gamma; denom == 0 means scaled p == q, accept
        # unconditionally.
        denom = z + 1.0 - p
        h_safe = tl.where(denom > 0.0, z / tl.where(denom > 0.0, denom, 1.0), 1.0)
        h = tl.where(step == NUM_SLOTS - 1, p, h_safe)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
        if coin <= h:
            tau = step
            z_res = z
            p_res = p

    tl.store(AcceptTokenNum + pid, tau)

    # Same predict layout as the classic kernel: predict at the previous
    # accepted slot's global index.
    last_accepted_global_idx = root_global_idx
    for step in range(1, NUM_SLOTS):
        if step <= tau:
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + step * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

    # Final (bonus/correction) token.
    all_drafts_accepted = tau == NUM_SLOTS - 1
    coin_final = tl.load(UniformSamplesFinal + pid)

    cur_prob_row = tau
    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only gamma rows (TargetProbs has gamma + 1). The
    # all-accepted branch never dereferences this pointer (tau == gamma).
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    if all_drafts_accepted:
        # Draft row gamma does not exist; keep this branch draft-pointer-free.
        residual_norm = _vocab_scan_target_mass(
            tp_base_ptr,
            stride_tp_v,
            VOCAB_SIZE,
            BLOCK_V,
        )
    else:
        # tau == 0 never snapshotted z_res; scan the root row with scale 1.
        if tau == 0:
            z_res = _vocab_scan_residual_mass(
                tp_base_ptr,
                dp_base_ptr_safe,
                1.0,
                stride_tp_v,
                stride_dp_v,
                VOCAB_SIZE,
                BLOCK_V,
            )
        residual_norm = z_res

    # Degenerate residual (norm == 0, scaled p == q everywhere) leaves the
    # cumsum at 0 <= target_u: final_token falls back to VOCAB_SIZE - 1, same
    # as the classic kernel.
    target_u = coin_final * residual_norm
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            t_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = t_val
            else:
                d_val = tl.load(
                    dp_base_ptr_safe + v_offsets * stride_dp_v, mask=mask, other=0.0
                )
                # Same NaN-q guard as the classic kernel.
                d_val = tl.where(d_val == d_val, d_val, 0.0)
                diff = p_res * t_val - d_val
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

        if coin * q < p:
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
            # Treat NaN q (degenerate draft rows) as 0: residual falls back to p.
            q_val = tl.where(q_val == q_val, q_val, 0.0)
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
                # Same NaN-q guard as pass 1.
                q_val = tl.where(q_val == q_val, q_val, 0.0)
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


def chain_block_speculative_sampling_triton(
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
    threshold_single,  # not used: block verification ignores accept thresholds
    threshold_acc,  # not used: block verification ignores accept thresholds
    deterministic,  # not used
):
    """Chain verification with block verification (arXiv:2403.10444).

    Drop-in replacement for chain_speculative_sampling_triton (same tensor
    contract); verifies the draft block jointly instead of token-by-token.
    """
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]

    grid = (batch_size,)
    speculative_sampling_block_kernel[grid](
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
