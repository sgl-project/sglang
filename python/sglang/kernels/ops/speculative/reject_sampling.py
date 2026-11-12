# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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


# K31HT_SAMP_SPLIT kernels begin
# K31HT_SAMP_SPLIT (default off = the kernel above, unchanged): the non-block path
# of speculative_sampling_classic_kernel in three launches.
#   1. accept walk, grid (bs,): the same code; keeps (cur_prob_row, num_accept,
#      last accepted slot) per request in State.
#   2. grid (bs, ceil(V / BLOCK_V)): each vocab block's residual, built with the
#      same expressions as the original; stores tl.sum(val) and max(tl.cumsum(val)).
#   3. final, grid (bs,): norm_sum = the block sums added in block order (the
#      original's pass-1 order); the first block with fl(cum + max cumsum) > target_u
#      is the block where the original's has_match first fires (fl(cum + c) is
#      monotone in c); the cumsum inside that block gives the token.
# The original runs both vocab passes serially in one CTA per request; here they
# run on bs x ceil(V / BLOCK_V) CTAs. Same BLOCK_V and num_warps, so tokens are
# bitwise-equal to the original.
@triton.jit
def _k31ht_samp_split_accept_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TargetProbs,
    DraftProbs,
    State,
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
    NUM_SLOTS: tl.constexpr,
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
    step = 1
    continue_verifying = 1
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
    tl.store(State + pid * 4 + 0, cur_prob_row.to(tl.int64))
    tl.store(State + pid * 4 + 1, num_accept.to(tl.int64))
    tl.store(State + pid * 4 + 2, last_accepted_global_idx.to(tl.int64))
    tl.store(State + pid * 4 + 3, root_global_idx.to(tl.int64))


@triton.jit
def _k31ht_samp_split_block_kernel(
    State,
    TargetProbs,
    DraftProbs,
    BlkSum,
    BlkLast,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NBLK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    row = tl.load(State + pid * 4 + 0)
    n_acc = tl.load(State + pid * 4 + 1)
    blk_all_accepted = n_acc == NUM_SLOTS - 1
    residual_scale = 1.0
    blk_tp_ptr = TargetProbs + (pid * stride_tp_b) + (row * stride_tp_s)
    # Never dereferenced when all drafts are accepted (row == NUM_SLOTS - 1).
    blk_dp_ptr = DraftProbs + (pid * stride_dp_b) + (row * stride_dp_s)
    blk_offsets = blk * BLOCK_V + tl.arange(0, BLOCK_V)
    blk_mask = blk_offsets < VOCAB_SIZE
    p_val = tl.load(blk_tp_ptr + blk_offsets * stride_tp_v, mask=blk_mask, other=0.0)
    if blk_all_accepted:
        val = p_val
    else:
        q_val = tl.load(
            blk_dp_ptr + blk_offsets * stride_dp_v, mask=blk_mask, other=0.0
        )
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        diff = residual_scale * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    block_cumsum = tl.cumsum(val, axis=0)
    tl.store(BlkSum + pid * NBLK + blk, tl.sum(val))
    tl.store(BlkLast + pid * NBLK + blk, tl.max(block_cumsum, axis=0))


@triton.jit
def _k31ht_samp_split_final_kernel(
    State,
    UniformSamplesFinal,
    BlkSum,
    BlkLast,
    TargetProbs,
    DraftProbs,
    Predicts,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NBLK: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = tl.load(State + pid * 4 + 0)
    num_accept = tl.load(State + pid * 4 + 1)
    last_accepted_global_idx = tl.load(State + pid * 4 + 2)
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    residual_scale = 1.0
    norm_sum = 0.0
    for i in range(0, NBLK):
        norm_sum += tl.load(BlkSum + pid * NBLK + i)
    coin_final = tl.load(UniformSamplesFinal + pid)
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    found = 0
    hit = 0
    hit_cum = 0.0
    for j in range(0, NBLK):
        if found == 0:
            blk_max_cumsum = tl.load(BlkLast + pid * NBLK + j)
            if (cum_sum + blk_max_cumsum) > target_u:
                hit = j
                hit_cum = cum_sum
                found = 1
            cum_sum += tl.load(BlkSum + pid * NBLK + j)
    final_token = VOCAB_SIZE - 1
    if found == 1:
        tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
        dp_base_ptr_safe = (
            DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)
        )
        v_start = hit * BLOCK_V
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
        if all_drafts_accepted:
            val = p_val
        else:
            q_val = tl.load(
                dp_base_ptr_safe + v_offsets * stride_dp_v, mask=mask, other=0.0
            )
            q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
            diff = residual_scale * p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)
        block_cumsum = tl.cumsum(val, axis=0)
        total_cumsum = hit_cum + block_cumsum
        candidates_mask = total_cumsum > target_u
        has_match = tl.max(candidates_mask, axis=0)
        if has_match:
            match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
            final_token = v_start + match_idx
    tl.store(Predicts + last_accepted_global_idx, final_token)


# K31HT_SAMP_SPLIT kernels end


# K31HT_SAMP_SPARSEQ kernels begin (v2)
# K31HT_SAMP_SPARSEQ (default off; needs K31HT_SAMP_SPLIT): the caller no longer scatters the selector's q rows into the dense FP32
# [bs, gamma, V] buffer (and no longer resets it): the buffer stays all-zero between calls. The accept walk reads q at each drafted token from
# the sparse (candidate ids, q rows) [bs, gamma, K] (unique ids per row), and, when it rejects, writes the K q values of the rejection row
# into the dense buffer; the unchanged SAMP_SPLIT block kernel reads that row as today; the final kernel reads it, then zeroes those K
# entries again. Every value the dense kernels read is the one the served scatter would have put there -> bitwise to the served path.
# (Rebuilding the residual row from the sparse form inside the block kernels was correct but slower.)
@triton.jit
def _k31ht_sq_lookup(CandIds, QRows, base_c, base_q, sck, sqk, token, K: tl.constexpr):
    q = 0.0
    for k in tl.static_range(K):
        cid = tl.load(CandIds + base_c + k * sck)
        qk = tl.load(QRows + base_q + k * sqk).to(tl.float32)
        q = tl.where(cid == token, qk, q)
    return q


@triton.jit
def _k31ht_sq_put_row(CandIds, QRows, DraftProbs, base_c, base_q, base_d, sck, sqk, stride_dp_v, ZERO: tl.constexpr, K: tl.constexpr):
    for k in tl.static_range(K):
        cid = tl.load(CandIds + base_c + k * sck)
        if ZERO:
            tl.store(DraftProbs + base_d + cid * stride_dp_v, 0.0)
        else:
            qk = tl.load(QRows + base_q + k * sqk).to(tl.float32)
            tl.store(DraftProbs + base_d + cid * stride_dp_v, qk)


@triton.jit
def _k31ht_sq_accept_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TargetProbs,
    DraftProbs,
    CandIds,
    QRows,
    State,
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
    scb,
    scs,
    sck,
    sqb,
    sqs,
    sqk,
    NUM_SLOTS: tl.constexpr,
    K: tl.constexpr,
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
    step = 1
    continue_verifying = 1
    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        p = tl.load(
            TargetProbs
            + pid * stride_tp_b
            + cur_prob_row * stride_tp_s
            + draft_token * stride_tp_v
        )
        q = _k31ht_sq_lookup(CandIds, QRows, pid * scb + cur_prob_row * scs, pid * sqb + cur_prob_row * sqs, sck, sqk, draft_token, K)
        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
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
    if num_accept < NUM_SLOTS - 1:
        _k31ht_sq_put_row(CandIds, QRows, DraftProbs, pid * scb + cur_prob_row * scs, pid * sqb + cur_prob_row * sqs,
                          pid * stride_dp_b + cur_prob_row * stride_dp_s, sck, sqk, stride_dp_v, False, K)
    tl.store(AcceptTokenNum + pid, num_accept)
    tl.store(State + pid * 4 + 0, cur_prob_row.to(tl.int64))
    tl.store(State + pid * 4 + 1, num_accept.to(tl.int64))
    tl.store(State + pid * 4 + 2, last_accepted_global_idx.to(tl.int64))
    tl.store(State + pid * 4 + 3, root_global_idx.to(tl.int64))


@triton.jit
def _k31ht_sq_final_kernel(
    State,
    UniformSamplesFinal,
    BlkSum,
    BlkLast,
    TargetProbs,
    DraftProbs,
    Predicts,
    CandIds,
    QRows,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    scb,
    scs,
    sck,
    sqb,
    sqs,
    sqk,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NBLK: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = tl.load(State + pid * 4 + 0)
    num_accept = tl.load(State + pid * 4 + 1)
    last_accepted_global_idx = tl.load(State + pid * 4 + 2)
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    residual_scale = 1.0
    norm_sum = 0.0
    for i in range(0, NBLK):
        norm_sum += tl.load(BlkSum + pid * NBLK + i)
    coin_final = tl.load(UniformSamplesFinal + pid)
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    found = 0
    hit = 0
    hit_cum = 0.0
    for j in range(0, NBLK):
        if found == 0:
            blk_max_cumsum = tl.load(BlkLast + pid * NBLK + j)
            if (cum_sum + blk_max_cumsum) > target_u:
                hit = j
                hit_cum = cum_sum
                found = 1
            cum_sum += tl.load(BlkSum + pid * NBLK + j)
    final_token = VOCAB_SIZE - 1
    if found == 1:
        tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
        dp_base_ptr_safe = (
            DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)
        )
        v_start = hit * BLOCK_V
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
        if all_drafts_accepted:
            val = p_val
        else:
            q_val = tl.load(
                dp_base_ptr_safe + v_offsets * stride_dp_v, mask=mask, other=0.0
            )
            q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
            diff = residual_scale * p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)
        block_cumsum = tl.cumsum(val, axis=0)
        total_cumsum = hit_cum + block_cumsum
        candidates_mask = total_cumsum > target_u
        has_match = tl.max(candidates_mask, axis=0)
        if has_match:
            match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
            final_token = v_start + match_idx
    tl.store(Predicts + last_accepted_global_idx, final_token)
    # Restore the all-zero dense buffer: every lane's read of this row is above; the barrier orders them before the zero stores.
    tl.debug_barrier()
    if not all_drafts_accepted:
        _k31ht_sq_put_row(CandIds, QRows, DraftProbs, pid * scb + cur_prob_row * scs, pid * sqb + cur_prob_row * sqs,
                          pid * stride_dp_b + cur_prob_row * stride_dp_s, sck, sqk, stride_dp_v, True, K)


# K31HT_SAMP_SPARSEQ kernels end

from sglang.kernels.ops.speculative.dspark.k31ht_softmax_split import k31ht_smx_lazy_take  # noqa: E402

_K31HT_SAMP_SPLIT_BLOCK_V = 4096
_K31HT_SAMP_SPLIT_LOGGED = set()


def k31ht_samp_split_requested() -> bool:
    """K31HT_SAMP_SPLIT: unset / '0' = the stock kernel, '1' = the split form."""
    raw = os.environ.get("K31HT_SAMP_SPLIT", "0").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"K31HT_SAMP_SPLIT={raw!r}: expected 0 or 1")


def _k31ht_samp_split_log_once(key, msg, *args):
    if key not in _K31HT_SAMP_SPLIT_LOGGED:
        _K31HT_SAMP_SPLIT_LOGGED.add(key)
        logger.info(msg, *args)


def _k31ht_samp_split(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    lazy=None,
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    block_v = _K31HT_SAMP_SPLIT_BLOCK_V
    nblk = triton.cdiv(vocab_size, block_v)
    device = candidates.device
    state = torch.empty((batch_size, 4), dtype=torch.int64, device=device)
    blk_sum = torch.empty((batch_size, nblk), dtype=torch.float32, device=device)
    blk_last = torch.empty((batch_size, nblk), dtype=torch.float32, device=device)
    tp_strides = (target_probs.stride(0), target_probs.stride(1), target_probs.stride(2))
    dp_strides = (draft_probs.stride(0), draft_probs.stride(1), draft_probs.stride(2))
    if lazy is not None:
        lazy.fill_draft(candidates)  # K31HT_SMX_LAZY: the entries the accept walk reads
    _k31ht_samp_split_accept_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        target_probs,
        draft_probs,
        state,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        *tp_strides,
        *dp_strides,
        NUM_SLOTS=num_slots,
        num_warps=4,
    )
    if lazy is not None:
        lazy.fill_rows(state)  # K31HT_SMX_LAZY: each request's accept row, read by the block and final kernels
    _k31ht_samp_split_block_kernel[(batch_size, nblk)](
        state,
        target_probs,
        draft_probs,
        blk_sum,
        blk_last,
        *tp_strides,
        *dp_strides,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=block_v,
        NBLK=nblk,
        num_warps=4,
    )
    _k31ht_samp_split_final_kernel[(batch_size,)](
        state,
        uniform_samples_for_final_sampling,
        blk_sum,
        blk_last,
        target_probs,
        draft_probs,
        predicts,
        *tp_strides,
        *dp_strides,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=block_v,
        NBLK=nblk,
        num_warps=4,
    )
    _k31ht_samp_split_log_once(
        ("engaged", num_slots, vocab_size),
        "K31HT_SAMP_SPLIT engaged (slots %d, vocab %d, blocks %d)",
        num_slots,
        vocab_size,
        nblk,
    )


_K31HT_SPARSE_Q = None  # (candidate_ids, q_rows) while a k31ht_sparse_q_context is open
_K31HT_SAMP_SPARSEQ_LOGGED = False


def k31ht_samp_sparseq_requested() -> bool:
    """K31HT_SAMP_SPARSEQ: unset / '' / '0' off, '1' on, anything else raises (fail closed)."""
    raw = os.environ.get("K31HT_SAMP_SPARSEQ", "")
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"K31HT_SAMP_SPARSEQ={raw!r}: expected unset, '', '0' or '1'")


class k31ht_sparse_q_context:
    """While open, chain_speculative_sampling_triton takes the draft distribution from (candidate_ids, q_rows) ([bs, gamma, K] each, unique
    ids per row) and the dense draft_probs argument must be the caller's ALL-ZERO buffer (the kernels write the rejection row into it and
    zero it again). Only the SAMP_SPLIT path supports it; any other path raises rather than read the empty dense buffer."""

    def __init__(self, candidate_ids, q_rows):
        self._v = (candidate_ids, q_rows)

    def __enter__(self):
        global _K31HT_SPARSE_Q
        if _K31HT_SPARSE_Q is not None:
            raise RuntimeError("K31HT_SAMP_SPARSEQ: nested sparse-q context")
        _K31HT_SPARSE_Q = self._v
        return self

    def __exit__(self, *exc):
        global _K31HT_SPARSE_Q
        _K31HT_SPARSE_Q = None
        return False


def _k31ht_samp_split_sparse(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    cand_ids,
    q_rows,
    lazy=None,
):
    global _K31HT_SAMP_SPARSEQ_LOGGED
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    K = int(cand_ids.shape[-1])
    if tuple(cand_ids.shape) != tuple(q_rows.shape) or cand_ids.shape[0] != batch_size or cand_ids.shape[1] < num_slots - 1:
        raise RuntimeError(
            f"K31HT_SAMP_SPARSEQ: candidate ids {tuple(cand_ids.shape)} / q rows {tuple(q_rows.shape)} do not match bs {batch_size}, slots {num_slots}"
        )
    block_v = _K31HT_SAMP_SPLIT_BLOCK_V
    nblk = triton.cdiv(vocab_size, block_v)
    device = candidates.device
    state = torch.empty((batch_size, 4), dtype=torch.int64, device=device)
    blk_sum = torch.empty((batch_size, nblk), dtype=torch.float32, device=device)
    blk_last = torch.empty((batch_size, nblk), dtype=torch.float32, device=device)
    tp_strides = (target_probs.stride(0), target_probs.stride(1), target_probs.stride(2))
    dp_strides = (draft_probs.stride(0), draft_probs.stride(1), draft_probs.stride(2))
    c_strides = (cand_ids.stride(0), cand_ids.stride(1), cand_ids.stride(2))
    q_strides = (q_rows.stride(0), q_rows.stride(1), q_rows.stride(2))
    if lazy is not None:
        lazy.fill_draft(candidates)  # K31HT_SMX_LAZY: the entries the accept walk reads
    _k31ht_sq_accept_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        target_probs,
        draft_probs,
        cand_ids,
        q_rows,
        state,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        *tp_strides,
        *dp_strides,
        *c_strides,
        *q_strides,
        NUM_SLOTS=num_slots,
        K=K,
        num_warps=4,
    )
    if lazy is not None:
        lazy.fill_rows(state)  # K31HT_SMX_LAZY: each request's accept row, read by the block and final kernels
    _k31ht_samp_split_block_kernel[(batch_size, nblk)](
        state,
        target_probs,
        draft_probs,
        blk_sum,
        blk_last,
        *tp_strides,
        *dp_strides,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=block_v,
        NBLK=nblk,
        num_warps=4,
    )
    _k31ht_sq_final_kernel[(batch_size,)](
        state,
        uniform_samples_for_final_sampling,
        blk_sum,
        blk_last,
        target_probs,
        draft_probs,
        predicts,
        cand_ids,
        q_rows,
        *tp_strides,
        *dp_strides,
        *c_strides,
        *q_strides,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=block_v,
        NBLK=nblk,
        K=K,
        num_warps=4,
    )
    if not _K31HT_SAMP_SPARSEQ_LOGGED:
        _K31HT_SAMP_SPARSEQ_LOGGED = True
        logger.info(
            "K31HT_SAMP_SPARSEQ ENGAGED: SAMP_SPLIT takes the selector's sparse q (K %d); no dense scatter / reset",
            K,
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
    if _K31HT_SPARSE_Q is not None or k31ht_samp_split_requested():
        lazy = None  # no lazy softmax producer on this tree
        if _K31HT_SPARSE_Q is not None:
            # The caller skipped the dense q scatter: only the SAMP_SPLIT path can take the sparse form.
            if not k31ht_samp_split_requested() or block_verification or target_probs.dtype != torch.float32 or draft_probs.dtype != torch.float32:
                raise RuntimeError(
                    "K31HT_SAMP_SPARSEQ: sparse q given but the SAMP_SPLIT path cannot run "
                    f"(SAMP_SPLIT {k31ht_samp_split_requested()}, block_verification {block_verification}, dtypes {target_probs.dtype} / {draft_probs.dtype})"
                )
            cand_ids, q_rows = _K31HT_SPARSE_Q
            _k31ht_samp_split_sparse(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrive_index,
                uniform_samples,
                uniform_samples_for_final_sampling,
                target_probs,
                draft_probs,
                cand_ids,
                q_rows,
                lazy=lazy,
            )
            return
        if k31ht_samp_split_requested():
            # Only the non-block, FP32 path is ported; anything else runs the stock kernel.
            if block_verification:
                reason = "block_verification"
            elif target_probs.dtype != torch.float32 or draft_probs.dtype != torch.float32:
                reason = f"dtype {target_probs.dtype} / {draft_probs.dtype}"
            else:
                reason = None
            if reason is None:
                _k31ht_samp_split(
                    predicts,
                    accept_index,
                    accept_token_num,
                    candidates,
                    retrive_index,
                    uniform_samples,
                    uniform_samples_for_final_sampling,
                    target_probs,
                    draft_probs,
                    lazy=lazy,
                )
                return
            _k31ht_samp_split_log_once(
                ("not_engaged", reason), "K31HT_SAMP_SPLIT NOT ENGAGED: %s", reason
            )
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
