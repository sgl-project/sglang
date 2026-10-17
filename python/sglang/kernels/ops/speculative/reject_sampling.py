import torch
import triton
import triton.language as tl

# Vocabulary tile of the final-token draw. Each (request, tile) pair runs in its own program.
_FINAL_BLOCK_V = 4096


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    ResidualScale,
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
    tl.store(ResidualScale + pid, residual_scale)


@triton.jit
def _final_token_weights(
    TargetProbs,
    DraftProbs,
    ResidualScale,
    pid,
    num_accept,
    v_offsets,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    # The final token is drawn from row num_accept: from p if all drafts were
    # accepted, else from the residual max(residual_scale * p - q, 0).
    mask = v_offsets < VOCAB_SIZE
    p_val = tl.load(
        TargetProbs
        + pid * stride_tp_b
        + num_accept * stride_tp_s
        + v_offsets * stride_tp_v,
        mask=mask,
        other=0.0,
    )
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted num_accept == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences DraftProbs; on rejection num_accept <= num_steps - 1.
    if num_accept == NUM_SLOTS - 1:
        val = p_val
    else:
        q_val = tl.load(
            DraftProbs
            + pid * stride_dp_b
            + num_accept * stride_dp_s
            + v_offsets * stride_dp_v,
            mask=mask,
            other=0.0,
        )
        # Treat any non-probability q (NaN, +-inf, negative) as 0: the
        # residual falls back to p. A comparison against NaN is false, so
        # the range test rejects it along with the infinities.
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        diff = tl.load(ResidualScale + pid) * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    return val


@triton.jit
def _final_token_block_sums_kernel(
    AcceptTokenNum,
    ResidualScale,
    TargetProbs,
    DraftProbs,
    BlockSums,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    num_blocks,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # One program per (request, vocabulary block): the weight of the block.
    pid = tl.program_id(0)
    block = tl.program_id(1)
    val = _final_token_weights(
        TargetProbs,
        DraftProbs,
        ResidualScale,
        pid,
        tl.load(AcceptTokenNum + pid),
        block * BLOCK_V + tl.arange(0, BLOCK_V),
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        NUM_SLOTS,
        VOCAB_SIZE,
    )
    tl.store(BlockSums + pid * num_blocks + block, tl.sum(val, axis=0))


@triton.jit
def _final_token_kernel(
    Predicts,
    AcceptTokenNum,
    RetriveIndex,
    UniformSamplesFinal,
    ResidualScale,
    TargetProbs,
    DraftProbs,
    BlockSums,
    stride_idx_b,
    stride_idx_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    num_blocks,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BLOCKS_POW2: tl.constexpr,
):
    # Inverse-CDF draw: the block sums locate the block where the CDF passes
    # coin_final * total, then a cumsum inside that block locates the token.
    pid = tl.program_id(0)
    num_accept = tl.load(AcceptTokenNum + pid)
    blocks = tl.arange(0, NUM_BLOCKS_POW2)
    block_sums = tl.load(
        BlockSums + pid * num_blocks + blocks, mask=blocks < num_blocks, other=0.0
    )
    block_cdf = tl.cumsum(block_sums, axis=0)
    target_u = tl.load(UniformSamplesFinal + pid) * tl.max(block_cdf, axis=0)
    crossed = (block_cdf > target_u) & (block_sums > 0.0)

    # Degenerate residual (total == 0, i.e. p == q everywhere on rejection)
    # crosses no block, so final_token falls back to VOCAB_SIZE - 1; acceptable
    # since this case is numerically near-impossible.
    final_token = VOCAB_SIZE - 1
    if tl.max(crossed.to(tl.int32), axis=0) > 0:
        block = tl.argmax(crossed.to(tl.int32), axis=0)
        cum_sum = tl.max(tl.where(blocks < block, block_cdf, 0.0), axis=0)
        v_offsets = block * BLOCK_V + tl.arange(0, BLOCK_V)
        val = _final_token_weights(
            TargetProbs,
            DraftProbs,
            ResidualScale,
            pid,
            num_accept,
            v_offsets,
            stride_tp_b,
            stride_tp_s,
            stride_tp_v,
            stride_dp_b,
            stride_dp_s,
            stride_dp_v,
            NUM_SLOTS,
            VOCAB_SIZE,
        )
        candidates_mask = (cum_sum + tl.cumsum(val, axis=0)) > target_u
        if tl.max(candidates_mask.to(tl.int32), axis=0) > 0:
            final_token = block * BLOCK_V + tl.argmax(
                candidates_mask.to(tl.int32), axis=0
            )
        else:
            # FP32 rounding put target_u at the end of this block: take its
            # last token with positive weight.
            final_token = tl.max(tl.where(val > 0.0, v_offsets, 0), axis=0)

    last_accepted_global_idx = tl.load(
        RetriveIndex + pid * stride_idx_b + num_accept * stride_idx_s
    )
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
    num_blocks = triton.cdiv(vocab_size, _FINAL_BLOCK_V)
    residual_scale = torch.empty(
        batch_size, dtype=torch.float32, device=target_probs.device
    )
    block_sums = torch.empty(
        (batch_size, num_blocks), dtype=torch.float32, device=target_probs.device
    )
    prob_strides = (*target_probs.stride(), *draft_probs.stride())

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        residual_scale,
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
        BLOCK_VERIFICATION=block_verification,
        BLOCK_REDUCE=block_reduce,
        num_warps=num_warps,
    )
    # Final token: the block sums run over all (request, block) pairs in
    # parallel, then one program per request searches a single block.
    _final_token_block_sums_kernel[(batch_size, num_blocks)](
        accept_token_num,
        residual_scale,
        target_probs,
        draft_probs,
        block_sums,
        *prob_strides,
        num_blocks,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=_FINAL_BLOCK_V,
        num_warps=8,
    )
    _final_token_kernel[grid](
        predicts,
        accept_token_num,
        retrive_index,
        uniform_samples_for_final_sampling,
        residual_scale,
        target_probs,
        draft_probs,
        block_sums,
        retrive_index.stride(0),
        retrive_index.stride(1),
        *prob_strides,
        num_blocks,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=_FINAL_BLOCK_V,
        NUM_BLOCKS_POW2=triton.next_power_of_2(num_blocks),
        num_warps=8,
    )
