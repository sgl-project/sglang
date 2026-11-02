import torch
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


@triton.jit
def speculative_sampling_target_only_chain_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
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
    threshold_single,
    threshold_acc,
    NUM_SPEC_STEPS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Single-kernel fused chain (topk=1) target-only speculative sampler.

    Verifies a linear chain of draft tokens using only ``target_probs`` (no
    ``draft_probs`` allocation) and samples the bonus token in the same launch.
    Mirrors the acceptance logic of the CUDA tree kernel
    (``sampling.cuh:80-81``) specialised to the no-sibling linear case, and
    the final-sample CDF guard from ``sampling.cuh:155-171``.
    """
    pid = tl.program_id(0)

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b
    acc_ptr_base = AcceptIndex + pid * stride_acc_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(acc_ptr_base + 0 * stride_acc_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    cur_prob_row = 0
    rejected_token = tl.full([], -1, dtype=tl.int32)
    coin = tl.load(uni_ptr_base + 0 * stride_uni_s)

    # ---- Chain verification (linear, topk == 1, no siblings) ----
    step = 1
    continue_verifying = 1
    while (step < NUM_SPEC_STEPS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s).to(tl.int32)
        p = tl.load(
            TargetProbs
            + pid * stride_tp_b
            + cur_prob_row * stride_tp_s
            + draft_token * stride_tp_v
        )

        # Acceptance: target has mass AND
        #   (coin < p / threshold_acc  OR  p >= threshold_single).
        has_target_mass = p > 0.0
        accept = has_target_mass & (
            (coin < p / threshold_acc) | (p >= threshold_single)
        )

        if accept:
            cur_prob_row = step
            coin = tl.load(uni_ptr_base + step * stride_uni_s)
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            num_accept += 1
            tl.store(acc_ptr_base + num_accept * stride_acc_s, last_accepted_global_idx)
            step += 1
        else:
            rejected_token = draft_token
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # ---- Final sampling from target_probs[cur_prob_row] ----
    # When a draft was rejected, mask its probability to 0 (the tree kernel
    # achieves this via draft_probs[rejected] = target_probs[rejected] and
    # relu(target_probs - draft_probs); we skip the allocation and mask inline).
    coin_final = tl.load(UniformSamplesFinal + pid)
    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    has_rejection = rejected_token >= 0

    # Pass 1: Sum of positive probs (rejected token masked).
    norm_sum = 0.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
        if has_rejection:
            p_val = tl.where(v_offsets == rejected_token, 0.0, p_val)
        val = tl.where(p_val > 0.0, p_val, 0.0)
        norm_sum += tl.sum(val)

    # Pass 2: CDF selection — require (total_cumsum > target_u) & (val > 0.0)
    # so a leading zero-mass token is never picked when coin_final == 0.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0
    last_valid_token = -1

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE
            p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
            if has_rejection:
                p_val = tl.where(v_offsets == rejected_token, 0.0, p_val)
            valid_mask = p_val > 0.0
            val = tl.where(valid_mask, p_val, 0.0)

            # Track the largest vocab index with val > 0 across all blocks.
            valid_indices = tl.where(valid_mask, v_offsets, -1)
            block_last_valid = tl.max(valid_indices, axis=0)
            if block_last_valid >= 0:
                last_valid_token = block_last_valid

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = (total_cumsum > target_u) & valid_mask
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    # Fallback for upper-endpoint coin_final where floating-point
    # total_cumsum <= target_u (sampling.cuh:165-171).
    if found == 0:
        if last_valid_token >= 0:
            final_token = last_valid_token

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_target_only_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor | None = None,
    retrive_next_sibling: torch.Tensor | None = None,
    uniform_samples: torch.Tensor | None = None,
    uniform_samples_for_final_sampling: torch.Tensor | None = None,
    target_probs: torch.Tensor | None = None,
    draft_probs: torch.Tensor | None = None,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Target-only chain (topk=1) speculative sampling without ``draft_probs``.

    Fuses draft verification and bonus-token final sampling into a single
    Triton kernel launch.  ``draft_probs`` is accepted for signature
    compatibility but never dereferenced — the rejected-token mask is applied
    inline, avoiding the ``torch.zeros_like(target_probs)`` allocation.
    """
    batch_size, num_slots = candidates.shape
    if batch_size == 0:
        return

    vocab_size = target_probs.shape[-1]
    # Bound the chain loop by the narrower of accept_index and candidates.
    num_spec_steps = min(accept_index.shape[1], num_slots)
    threshold_acc = max(float(threshold_acc), 1e-9)

    grid = (batch_size,)
    speculative_sampling_target_only_chain_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        accept_index.stride(0),
        accept_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        float(threshold_single),
        threshold_acc,
        NUM_SPEC_STEPS=num_spec_steps,
        VOCAB_SIZE=vocab_size,
        # 8192 halves the two vocab passes to 19 blocks each at vocab 152064;
        # 16384 and above spill registers and lose badly (0.30x).
        BLOCK_V=8192,
        num_warps=4,
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
