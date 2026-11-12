# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Chain speculative sampling over a sparse selector q, split across the vocabulary.

Computes what ``_selector_sampling_accept`` computes (scatter the selector's top-k q
into a dense [bs, gamma, vocab] q, ``speculative_sampling_classic_kernel`` token-wise,
then gather the bonus) without the dense q and with the residual pass split over
vocabulary blocks. The per-block sums, the block cumsums and the sequential
block-order sums are the classic kernel's own (same 4096-wide blocks and warps), so
the results match it bit for bit for the same uniforms.
"""

import torch
import triton
import triton.language as tl

_BLOCK_V = 4096


@triton.jit
def _walk_chain(
    pid,
    cand_ptr,
    tp_ptr,
    cand_ids_ptr,
    q_ptr,
    uni_ptr,
    stride_cand_b,
    stride_tp_b,
    stride_tp_s,
    stride_qb,
    stride_qs,
    stride_uni_b,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
):
    # speculative_sampling_classic_kernel's token-wise loop: slot s is tested on row
    # s - 1 while every earlier slot was accepted, so all slots' (p, q, coin) load
    # up front and the first failing slot ends the chain.
    s = tl.arange(0, SLOTS_P2)
    valid = (s >= 1) & (s < NUM_SLOTS)
    draft = tl.load(cand_ptr + pid * stride_cand_b + s, mask=valid, other=0)
    p = tl.load(
        tp_ptr + pid * stride_tp_b + (s - 1) * stride_tp_s + draft, mask=valid, other=0.0
    )
    coin = tl.load(uni_ptr + pid * stride_uni_b + (s - 1), mask=valid, other=1.0)
    k = tl.arange(0, TOPK_P2)
    offs = pid * stride_qb + (s[:, None] - 1) * stride_qs + k[None, :]
    lane = valid[:, None] & (k[None, :] < TOPK)
    cid = tl.load(cand_ids_ptr + offs, mask=lane, other=-1)
    qv = tl.load(q_ptr + offs, mask=lane, other=0.0)
    # Selector top-k ids are distinct: the masked sum is the matching q exactly, or 0.
    q = tl.sum(tl.where(cid == draft[:, None], qv, 0.0), axis=1)
    q_is_prob = (q > 0.0) & (q <= 1.0)
    ok = q_is_prob & (coin * q < p)
    first_fail = tl.min(tl.where(valid & ~ok, s, NUM_SLOTS), axis=0)
    num_accept = first_fail - 1
    return num_accept, num_accept


@triton.jit
def _residual_block(
    pid,
    v_start,
    cur_prob_row,
    all_drafts_accepted,
    tp_ptr,
    cand_ids_ptr,
    q_ptr,
    stride_tp_b,
    stride_tp_s,
    stride_qb,
    stride_qs,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    TOPK: tl.constexpr,
):
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE
    p_val = tl.load(
        tp_ptr + pid * stride_tp_b + cur_prob_row * stride_tp_s + v_offsets,
        mask=mask,
        other=0.0,
    )
    if all_drafts_accepted:
        val = p_val
    else:
        base = pid * stride_qb + cur_prob_row * stride_qs
        q_val = tl.zeros([BLOCK_V], dtype=tl.float32)
        for j in tl.static_range(TOPK):
            cid = tl.load(cand_ids_ptr + base + j)
            qj = tl.load(q_ptr + base + j)
            q_val = tl.where(v_offsets == cid, qj, q_val)
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        residual_scale = 1.0
        diff = residual_scale * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    return val


@triton.jit
def _residual_partials_kernel(
    cand_ptr,
    tp_ptr,
    cand_ids_ptr,
    q_ptr,
    uni_ptr,
    part_sum_ptr,
    part_max_ptr,
    stride_cand_b,
    stride_tp_b,
    stride_tp_s,
    stride_qb,
    stride_qs,
    stride_uni_b,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    num_accept, cur_prob_row = _walk_chain(
        pid, cand_ptr, tp_ptr, cand_ids_ptr, q_ptr, uni_ptr,
        stride_cand_b, stride_tp_b, stride_tp_s, stride_qb, stride_qs, stride_uni_b,
        NUM_SLOTS, SLOTS_P2, TOPK, TOPK_P2,
    )
    val = _residual_block(
        pid, blk * BLOCK_V, cur_prob_row, num_accept == NUM_SLOTS - 1,
        tp_ptr, cand_ids_ptr, q_ptr, stride_tp_b, stride_tp_s, stride_qb, stride_qs,
        VOCAB_SIZE, BLOCK_V, TOPK,
    )
    tl.store(part_sum_ptr + pid * NUM_BLOCKS + blk, tl.sum(val))
    tl.store(part_max_ptr + pid * NUM_BLOCKS + blk, tl.max(tl.cumsum(val, axis=0), axis=0))


@triton.jit
def _residual_select_kernel(
    cand_ptr,
    tp_ptr,
    cand_ids_ptr,
    q_ptr,
    uni_ptr,
    uni_final_ptr,
    part_sum_ptr,
    part_max_ptr,
    out_ptr,
    stride_cand_b,
    stride_tp_b,
    stride_tp_s,
    stride_qb,
    stride_qs,
    stride_uni_b,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_accept, cur_prob_row = _walk_chain(
        pid, cand_ptr, tp_ptr, cand_ids_ptr, q_ptr, uni_ptr,
        stride_cand_b, stride_tp_b, stride_tp_s, stride_qb, stride_qs, stride_uni_b,
        NUM_SLOTS, SLOTS_P2, TOPK, TOPK_P2,
    )
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    # Block-order sequential sums, as the classic kernel's pass 1 and pass 2; unrolled
    # so the partials load together.
    norm_sum = 0.0
    for b in tl.static_range(NUM_BLOCKS):
        norm_sum += tl.load(part_sum_ptr + pid * NUM_BLOCKS + b)
    target_u = tl.load(uni_final_ptr + pid) * norm_sum
    # Rounding is monotone, so some cum_sum + cumsum[j] exceeds target_u exactly
    # when cum_sum + max_j cumsum[j] does.
    cum_sum = 0.0
    hit_cum = 0.0
    hit_block = -1
    for b in tl.static_range(NUM_BLOCKS):
        hit = (hit_block < 0) & (cum_sum + tl.load(part_max_ptr + pid * NUM_BLOCKS + b) > target_u)
        hit_cum = tl.where(hit, cum_sum, hit_cum)
        hit_block = tl.where(hit, b, hit_block)
        cum_sum = tl.where(
            hit_block < 0, cum_sum + tl.load(part_sum_ptr + pid * NUM_BLOCKS + b), cum_sum
        )
    final_token = VOCAB_SIZE - 1
    if hit_block >= 0:
        val = _residual_block(
            pid, hit_block * BLOCK_V, cur_prob_row, all_drafts_accepted,
            tp_ptr, cand_ids_ptr, q_ptr, stride_tp_b, stride_tp_s, stride_qb, stride_qs,
            VOCAB_SIZE, BLOCK_V, TOPK,
        )
        total_cumsum = hit_cum + tl.cumsum(val, axis=0)
        candidates_mask = total_cumsum > target_u
        match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
        final_token = hit_block * BLOCK_V + match_idx
    tl.store(out_ptr + pid * 2, num_accept.to(tl.int64))
    tl.store(out_ptr + pid * 2 + 1, final_token.to(tl.int64))


def selector_sampled_accept(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
) -> torch.Tensor:
    """Token-wise chain verify of one selector path per row.

    candidates [bs, gamma + 1] int64 (slot 0 = anchor), target_probs [bs, gamma + 1,
    vocab] fp32 (last dim contiguous), candidate_ids / q_rows [bs, gamma, top_k]
    (int64 / fp32, distinct ids per row), uniform_samples [bs, gamma] and
    uniform_samples_final [bs] fp32. Returns int64 [bs, 2]: (num_correct_drafts,
    bonus_token) per row.
    """
    bs, num_slots = candidates.shape
    vocab = target_probs.shape[-1]
    top_k = candidate_ids.shape[-1]
    assert target_probs.stride(-1) == 1 and candidates.stride(-1) == 1
    assert candidate_ids.is_contiguous() and q_rows.is_contiguous()
    assert q_rows.dtype == torch.float32 and uniform_samples.stride(-1) == 1
    assert candidate_ids.stride() == q_rows.stride()
    num_blocks = triton.cdiv(vocab, _BLOCK_V)
    device = candidates.device
    part_sum = torch.empty((bs, num_blocks), dtype=torch.float32, device=device)
    part_max = torch.empty((bs, num_blocks), dtype=torch.float32, device=device)
    out = torch.empty((bs, 2), dtype=torch.int64, device=device)
    strides = (
        candidates.stride(0),
        target_probs.stride(0),
        target_probs.stride(1),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        uniform_samples.stride(0),
    )
    common = dict(
        NUM_SLOTS=num_slots,
        SLOTS_P2=triton.next_power_of_2(num_slots),
        TOPK=top_k,
        TOPK_P2=triton.next_power_of_2(top_k),
        VOCAB_SIZE=vocab,
        BLOCK_V=_BLOCK_V,
        NUM_BLOCKS=num_blocks,
    )
    # num_warps=4 matches chain_speculative_sampling_triton's token-wise launch,
    # which fixes the per-block reduction order.
    _residual_partials_kernel[(bs, num_blocks)](
        candidates, target_probs, candidate_ids, q_rows, uniform_samples,
        part_sum, part_max, *strides, num_warps=4, **common,
    )
    _residual_select_kernel[(bs,)](
        candidates, target_probs, candidate_ids, q_rows, uniform_samples,
        uniform_samples_final, part_sum, part_max, out, *strides,
        num_warps=4, **common,
    )
    return out


@triton.jit
def _commit_block_kernel(
    cand_ptr,
    res_ptr,
    prefix_ptr,
    num_correct_out,
    commit_lens_out,
    bonus_out,
    out_tokens_out,
    new_seq_lens_out,
    BLOCK: tl.constexpr,
    BLOCK_P2: tl.constexpr,
):
    # _commit_accept: drafts shifted left, the bonus at the accept boundary.
    pid = tl.program_id(0)
    num_correct = tl.load(res_ptr + pid * 2)
    bonus = tl.load(res_ptr + pid * 2 + 1)
    cols = tl.arange(0, BLOCK_P2)
    shifted = tl.load(cand_ptr + pid * BLOCK + cols + 1, mask=cols < BLOCK - 1, other=0)
    tokens = tl.where(cols == num_correct, bonus, shifted)
    tl.store(out_tokens_out + pid * BLOCK + cols, tokens, mask=cols < BLOCK)
    tl.store(num_correct_out + pid, num_correct.to(tl.int32))
    tl.store(commit_lens_out + pid, (num_correct + 1).to(tl.int32))
    tl.store(bonus_out + pid, bonus)
    prefix = tl.load(prefix_ptr + pid)
    tl.store(new_seq_lens_out + pid, prefix + (num_correct + 1).to(prefix.dtype))


def commit_selector_block(
    *,
    candidates: torch.Tensor,
    accept_result: torch.Tensor,
    prefix_lens: torch.Tensor,
):
    """(num_correct_drafts int32, commit_lens int32, bonus_tokens int64, out_tokens
    int64 [bs, block], new_seq_lens) from selector_sampled_accept's [bs, 2] result."""
    bs, block = candidates.shape
    assert candidates.is_contiguous() and accept_result.is_contiguous()
    device = candidates.device
    num_correct = torch.empty((bs,), dtype=torch.int32, device=device)
    commit_lens = torch.empty((bs,), dtype=torch.int32, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    out_tokens = torch.empty((bs, block), dtype=torch.int64, device=device)
    new_seq_lens = torch.empty_like(prefix_lens)
    _commit_block_kernel[(bs,)](
        candidates, accept_result, prefix_lens, num_correct, commit_lens, bonus,
        out_tokens, new_seq_lens, BLOCK=block, BLOCK_P2=triton.next_power_of_2(block),
    )
    return num_correct, commit_lens, bonus, out_tokens, new_seq_lens


# SGLANG_OPT_DFLASH_SELECTOR_SYNC_FOLD: the accept result's TP sync (SpecTpSync's exact fp32 all-reduce of the root's
# values) without its three glue kernels (zeros, the root's int -> fp32 copy, the fp32 -> int copy back). The select
# kernel writes the padded fp32 all-reduce input itself (the root its values, the other ranks zeros) and the commit
# kernel reads the all-reduce output. Copies of the served kernels plus those stores / loads, so the served kernels
# stay byte-identical; every value is an integer below 2**24, so fp32 holds it exactly and the sum of one nonzero and
# zeros is exact.


@triton.jit
def _residual_select_sync_kernel(
    cand_ptr,
    tp_ptr,
    cand_ids_ptr,
    q_ptr,
    uni_ptr,
    uni_final_ptr,
    part_sum_ptr,
    part_max_ptr,
    sync_ptr,
    sync_n,
    sync_padded,
    stride_cand_b,
    stride_tp_b,
    stride_tp_s,
    stride_qb,
    stride_qs,
    stride_uni_b,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SYNC_VALUES: tl.constexpr,
):
    # _residual_select_kernel's body, unchanged up to its stores.
    pid = tl.program_id(0)
    num_accept, cur_prob_row = _walk_chain(
        pid, cand_ptr, tp_ptr, cand_ids_ptr, q_ptr, uni_ptr,
        stride_cand_b, stride_tp_b, stride_tp_s, stride_qb, stride_qs, stride_uni_b,
        NUM_SLOTS, SLOTS_P2, TOPK, TOPK_P2,
    )
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    norm_sum = 0.0
    for b in tl.static_range(NUM_BLOCKS):
        norm_sum += tl.load(part_sum_ptr + pid * NUM_BLOCKS + b)
    target_u = tl.load(uni_final_ptr + pid) * norm_sum
    cum_sum = 0.0
    hit_cum = 0.0
    hit_block = -1
    for b in tl.static_range(NUM_BLOCKS):
        hit = (hit_block < 0) & (cum_sum + tl.load(part_max_ptr + pid * NUM_BLOCKS + b) > target_u)
        hit_cum = tl.where(hit, cum_sum, hit_cum)
        hit_block = tl.where(hit, b, hit_block)
        cum_sum = tl.where(
            hit_block < 0, cum_sum + tl.load(part_sum_ptr + pid * NUM_BLOCKS + b), cum_sum
        )
    final_token = VOCAB_SIZE - 1
    if hit_block >= 0:
        val = _residual_block(
            pid, hit_block * BLOCK_V, cur_prob_row, all_drafts_accepted,
            tp_ptr, cand_ids_ptr, q_ptr, stride_tp_b, stride_tp_s, stride_qb, stride_qs,
            VOCAB_SIZE, BLOCK_V, TOPK,
        )
        total_cumsum = hit_cum + tl.cumsum(val, axis=0)
        candidates_mask = total_cumsum > target_u
        match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
        final_token = hit_block * BLOCK_V + match_idx
    # The all-reduce input: this rank's values (the root, or every rank when the sync is off) or zeros.
    if SYNC_VALUES:
        tl.store(sync_ptr + pid * 2, num_accept.to(tl.float32))
        tl.store(sync_ptr + pid * 2 + 1, final_token.to(tl.float32))
    else:
        tl.store(sync_ptr + pid * 2, 0.0)
        tl.store(sync_ptr + pid * 2 + 1, 0.0)
    if pid == 0:
        pad = sync_n + tl.arange(0, 4)
        tl.store(sync_ptr + pad, tl.zeros((4,), dtype=tl.float32), mask=pad < sync_padded)


@triton.jit
def _commit_block_f32_kernel(
    cand_ptr,
    res_ptr,
    prefix_ptr,
    num_correct_out,
    commit_lens_out,
    bonus_out,
    out_tokens_out,
    new_seq_lens_out,
    BLOCK: tl.constexpr,
    BLOCK_P2: tl.constexpr,
):
    # _commit_block_kernel reading the fp32 all-reduce output (exact integers).
    pid = tl.program_id(0)
    num_correct = tl.load(res_ptr + pid * 2).to(tl.int64)
    bonus = tl.load(res_ptr + pid * 2 + 1).to(tl.int64)
    cols = tl.arange(0, BLOCK_P2)
    shifted = tl.load(cand_ptr + pid * BLOCK + cols + 1, mask=cols < BLOCK - 1, other=0)
    tokens = tl.where(cols == num_correct, bonus, shifted)
    tl.store(out_tokens_out + pid * BLOCK + cols, tokens, mask=cols < BLOCK)
    tl.store(num_correct_out + pid, num_correct.to(tl.int32))
    tl.store(commit_lens_out + pid, (num_correct + 1).to(tl.int32))
    tl.store(bonus_out + pid, bonus)
    prefix = tl.load(prefix_ptr + pid)
    tl.store(new_seq_lens_out + pid, prefix + (num_correct + 1).to(prefix.dtype))


def selector_sampled_accept_synced(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    prefix_lens: torch.Tensor,
    all_reduce,
    write_values: bool,
):
    """selector_sampled_accept + SpecTpSync's all-reduce sync + commit_selector_block, with the sync's glue folded
    into the two kernels. ``all_reduce(buf)`` returns the group's sum of the fp32 [ceil(2 bs / 4) * 4] buffer (or
    ``buf`` itself when the sync is off); ``write_values``: this rank's values go in (the root, or every rank when the
    sync is off), else zeros. Returns commit_selector_block's tuple."""
    bs, num_slots = candidates.shape
    vocab = target_probs.shape[-1]
    top_k = candidate_ids.shape[-1]
    assert target_probs.stride(-1) == 1 and candidates.stride(-1) == 1 and candidates.is_contiguous()
    assert candidate_ids.is_contiguous() and q_rows.is_contiguous()
    assert q_rows.dtype == torch.float32 and uniform_samples.stride(-1) == 1
    assert candidate_ids.stride() == q_rows.stride()
    assert vocab < (1 << 24), "fp32 sync holds token ids exactly only below 2**24"
    num_blocks = triton.cdiv(vocab, _BLOCK_V)
    device = candidates.device
    part_sum = torch.empty((bs, num_blocks), dtype=torch.float32, device=device)
    part_max = torch.empty((bs, num_blocks), dtype=torch.float32, device=device)
    n = 2 * bs
    padded = (n + 3) // 4 * 4
    sync_in = torch.empty((padded,), dtype=torch.float32, device=device)
    strides = (
        candidates.stride(0),
        target_probs.stride(0),
        target_probs.stride(1),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        uniform_samples.stride(0),
    )
    common = dict(
        NUM_SLOTS=num_slots,
        SLOTS_P2=triton.next_power_of_2(num_slots),
        TOPK=top_k,
        TOPK_P2=triton.next_power_of_2(top_k),
        VOCAB_SIZE=vocab,
        BLOCK_V=_BLOCK_V,
        NUM_BLOCKS=num_blocks,
    )
    _residual_partials_kernel[(bs, num_blocks)](
        candidates, target_probs, candidate_ids, q_rows, uniform_samples,
        part_sum, part_max, *strides, num_warps=4, **common,
    )
    _residual_select_sync_kernel[(bs,)](
        candidates, target_probs, candidate_ids, q_rows, uniform_samples,
        uniform_samples_final, part_sum, part_max, sync_in, n, padded, *strides,
        num_warps=4, SYNC_VALUES=bool(write_values), **common,
    )
    summed = all_reduce(sync_in)
    block = num_slots
    num_correct = torch.empty((bs,), dtype=torch.int32, device=device)
    commit_lens = torch.empty((bs,), dtype=torch.int32, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    out_tokens = torch.empty((bs, block), dtype=torch.int64, device=device)
    new_seq_lens = torch.empty_like(prefix_lens)
    _commit_block_f32_kernel[(bs,)](
        candidates, summed, prefix_lens, num_correct, commit_lens, bonus,
        out_tokens, new_seq_lens, BLOCK=block, BLOCK_P2=triton.next_power_of_2(block),
    )
    return num_correct, commit_lens, bonus, out_tokens, new_seq_lens


# SGLANG_OPT_DFLASH_SELECTOR_LOGIT_ACCEPT: the accept reads the verify logits and verify_softmax_wide_stats's per-row
# (row_max, inv_denominator, inv_temp) instead of the [bs, gamma + 1, vocab] probabilities, and evaluates p only where
# it reads it (the drafts of the chain walk, the residual row). _prob_from_logit is verify_softmax_norm_kernel's p in
# the same PTX instructions, so every p, partial and result equals the probability path's bit for bit. Copies of the
# served kernels with each probability load replaced by that evaluation, so the served kernels stay byte-identical.


@triton.jit
def _prob_from_logit(logit, inv_temp, row_max, inv_denominator):
    # nvcc -use_fast_math's code for __expf(__fmul_rn(logit, inv_temp) - row_max) * inv_denominator. The .rn on the
    # sub and the multiplies only stops ptxas from contracting them into an FMA with neighbouring arithmetic; rn is
    # their rounding either way.
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .f32 t;
        mul.rn.ftz.f32 t, $1, $2;
        sub.rn.ftz.f32 t, t, $3;
        mul.rn.ftz.f32 t, t, 0f3FB8AA3B;
        ex2.approx.ftz.f32 t, t;
        mul.rn.ftz.f32 $0, $4, t;
        }""",
        constraints="=f,f,f,f,f",
        args=[logit, inv_temp, row_max, inv_denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _walk_chain_lg(
    pid,
    cand_ptr,
    lg_ptr,
    stats_ptr,
    cand_ids_ptr,
    q_ptr,
    uni_ptr,
    stride_cand_b,
    stride_lg_b,
    stride_lg_s,
    stride_qb,
    stride_qs,
    stride_uni_b,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
):
    # _walk_chain with p evaluated from the logit of row s - 1 at the draft.
    s = tl.arange(0, SLOTS_P2)
    valid = (s >= 1) & (s < NUM_SLOTS)
    draft = tl.load(cand_ptr + pid * stride_cand_b + s, mask=valid, other=0)
    logit = tl.load(
        lg_ptr + pid * stride_lg_b + (s - 1) * stride_lg_s + draft, mask=valid, other=0.0
    )
    row = pid * NUM_SLOTS + (s - 1)
    row_max = tl.load(stats_ptr + row * 4, mask=valid, other=0.0)
    inv_denominator = tl.load(stats_ptr + row * 4 + 1, mask=valid, other=0.0)
    inv_temp = tl.load(stats_ptr + row * 4 + 2, mask=valid, other=0.0)
    p = tl.where(valid, _prob_from_logit(logit, inv_temp, row_max, inv_denominator), 0.0)
    coin = tl.load(uni_ptr + pid * stride_uni_b + (s - 1), mask=valid, other=1.0)
    k = tl.arange(0, TOPK_P2)
    offs = pid * stride_qb + (s[:, None] - 1) * stride_qs + k[None, :]
    lane = valid[:, None] & (k[None, :] < TOPK)
    cid = tl.load(cand_ids_ptr + offs, mask=lane, other=-1)
    qv = tl.load(q_ptr + offs, mask=lane, other=0.0)
    # Selector top-k ids are distinct: the masked sum is the matching q exactly, or 0.
    q = tl.sum(tl.where(cid == draft[:, None], qv, 0.0), axis=1)
    q_is_prob = (q > 0.0) & (q <= 1.0)
    ok = q_is_prob & (coin * q < p)
    first_fail = tl.min(tl.where(valid & ~ok, s, NUM_SLOTS), axis=0)
    num_accept = first_fail - 1
    return num_accept, num_accept


@triton.jit
def _residual_block_lg(
    pid,
    v_start,
    cur_prob_row,
    all_drafts_accepted,
    lg_ptr,
    stats_ptr,
    cand_ids_ptr,
    q_ptr,
    stride_lg_b,
    stride_lg_s,
    stride_qb,
    stride_qs,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    TOPK: tl.constexpr,
):
    # _residual_block with p_val evaluated from row cur_prob_row's logits.
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE
    logit = tl.load(
        lg_ptr + pid * stride_lg_b + cur_prob_row * stride_lg_s + v_offsets,
        mask=mask,
        other=0.0,
    )
    row = pid * NUM_SLOTS + cur_prob_row
    row_max = tl.load(stats_ptr + row * 4)
    inv_denominator = tl.load(stats_ptr + row * 4 + 1)
    inv_temp = tl.load(stats_ptr + row * 4 + 2)
    p_val = tl.where(mask, _prob_from_logit(logit, inv_temp, row_max, inv_denominator), 0.0)
    if all_drafts_accepted:
        val = p_val
    else:
        base = pid * stride_qb + cur_prob_row * stride_qs
        q_val = tl.zeros([BLOCK_V], dtype=tl.float32)
        for j in tl.static_range(TOPK):
            cid = tl.load(cand_ids_ptr + base + j)
            qj = tl.load(q_ptr + base + j)
            q_val = tl.where(v_offsets == cid, qj, q_val)
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        residual_scale = 1.0
        diff = residual_scale * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    return val


@triton.jit
def _residual_partials_lg_kernel(
    cand_ptr,
    lg_ptr,
    stats_ptr,
    cand_ids_ptr,
    q_ptr,
    uni_ptr,
    part_sum_ptr,
    part_max_ptr,
    stride_cand_b,
    stride_lg_b,
    stride_lg_s,
    stride_qb,
    stride_qs,
    stride_uni_b,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    num_accept, cur_prob_row = _walk_chain_lg(
        pid, cand_ptr, lg_ptr, stats_ptr, cand_ids_ptr, q_ptr, uni_ptr,
        stride_cand_b, stride_lg_b, stride_lg_s, stride_qb, stride_qs, stride_uni_b,
        NUM_SLOTS, SLOTS_P2, TOPK, TOPK_P2,
    )
    val = _residual_block_lg(
        pid, blk * BLOCK_V, cur_prob_row, num_accept == NUM_SLOTS - 1,
        lg_ptr, stats_ptr, cand_ids_ptr, q_ptr, stride_lg_b, stride_lg_s, stride_qb, stride_qs,
        NUM_SLOTS, VOCAB_SIZE, BLOCK_V, TOPK,
    )
    tl.store(part_sum_ptr + pid * NUM_BLOCKS + blk, tl.sum(val))
    tl.store(part_max_ptr + pid * NUM_BLOCKS + blk, tl.max(tl.cumsum(val, axis=0), axis=0))


@triton.jit
def _residual_select_lg_kernel(
    cand_ptr,
    lg_ptr,
    stats_ptr,
    cand_ids_ptr,
    q_ptr,
    uni_ptr,
    uni_final_ptr,
    part_sum_ptr,
    part_max_ptr,
    out_ptr,
    sync_n,
    sync_padded,
    stride_cand_b,
    stride_lg_b,
    stride_lg_s,
    stride_qb,
    stride_qs,
    stride_uni_b,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SYNC: tl.constexpr,
    SYNC_VALUES: tl.constexpr,
):
    # _residual_select_kernel (SYNC False: out_ptr is the int64 [bs, 2] result) or _residual_select_sync_kernel
    # (SYNC True: out_ptr is the padded fp32 all-reduce input), with the _lg helpers.
    pid = tl.program_id(0)
    num_accept, cur_prob_row = _walk_chain_lg(
        pid, cand_ptr, lg_ptr, stats_ptr, cand_ids_ptr, q_ptr, uni_ptr,
        stride_cand_b, stride_lg_b, stride_lg_s, stride_qb, stride_qs, stride_uni_b,
        NUM_SLOTS, SLOTS_P2, TOPK, TOPK_P2,
    )
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    norm_sum = 0.0
    for b in tl.static_range(NUM_BLOCKS):
        norm_sum += tl.load(part_sum_ptr + pid * NUM_BLOCKS + b)
    target_u = tl.load(uni_final_ptr + pid) * norm_sum
    cum_sum = 0.0
    hit_cum = 0.0
    hit_block = -1
    for b in tl.static_range(NUM_BLOCKS):
        hit = (hit_block < 0) & (cum_sum + tl.load(part_max_ptr + pid * NUM_BLOCKS + b) > target_u)
        hit_cum = tl.where(hit, cum_sum, hit_cum)
        hit_block = tl.where(hit, b, hit_block)
        cum_sum = tl.where(
            hit_block < 0, cum_sum + tl.load(part_sum_ptr + pid * NUM_BLOCKS + b), cum_sum
        )
    final_token = VOCAB_SIZE - 1
    if hit_block >= 0:
        val = _residual_block_lg(
            pid, hit_block * BLOCK_V, cur_prob_row, all_drafts_accepted,
            lg_ptr, stats_ptr, cand_ids_ptr, q_ptr, stride_lg_b, stride_lg_s, stride_qb, stride_qs,
            NUM_SLOTS, VOCAB_SIZE, BLOCK_V, TOPK,
        )
        total_cumsum = hit_cum + tl.cumsum(val, axis=0)
        candidates_mask = total_cumsum > target_u
        match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
        final_token = hit_block * BLOCK_V + match_idx
    if SYNC:
        if SYNC_VALUES:
            tl.store(out_ptr + pid * 2, num_accept.to(tl.float32))
            tl.store(out_ptr + pid * 2 + 1, final_token.to(tl.float32))
        else:
            tl.store(out_ptr + pid * 2, 0.0)
            tl.store(out_ptr + pid * 2 + 1, 0.0)
        if pid == 0:
            pad = sync_n + tl.arange(0, 4)
            tl.store(out_ptr + pad, tl.zeros((4,), dtype=tl.float32), mask=pad < sync_padded)
    else:
        tl.store(out_ptr + pid * 2, num_accept.to(tl.int64))
        tl.store(out_ptr + pid * 2 + 1, final_token.to(tl.int64))


def _selector_logit_launch(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    logit_stats: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    out: torch.Tensor,
    sync_n: int,
    sync_padded: int,
    sync: bool,
    sync_values: bool,
) -> None:
    bs, num_slots = candidates.shape
    vocab = target_logits.shape[-1]
    top_k = candidate_ids.shape[-1]
    # Rows are request-major, num_slots per request, as verify_softmax_wide_stats wrote the stats.
    assert target_logits.dtype == torch.float32 and target_logits.is_contiguous()
    assert target_logits.shape == (bs * num_slots, vocab)
    assert logit_stats.dtype == torch.float32 and logit_stats.is_contiguous()
    assert logit_stats.shape == (bs * num_slots, 4)
    assert candidates.stride(-1) == 1
    assert candidate_ids.is_contiguous() and q_rows.is_contiguous()
    assert q_rows.dtype == torch.float32 and uniform_samples.stride(-1) == 1
    assert candidate_ids.stride() == q_rows.stride()
    num_blocks = triton.cdiv(vocab, _BLOCK_V)
    device = candidates.device
    part_sum = torch.empty((bs, num_blocks), dtype=torch.float32, device=device)
    part_max = torch.empty((bs, num_blocks), dtype=torch.float32, device=device)
    strides = (
        candidates.stride(0),
        num_slots * vocab,
        vocab,
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        uniform_samples.stride(0),
    )
    common = dict(
        NUM_SLOTS=num_slots,
        SLOTS_P2=triton.next_power_of_2(num_slots),
        TOPK=top_k,
        TOPK_P2=triton.next_power_of_2(top_k),
        VOCAB_SIZE=vocab,
        BLOCK_V=_BLOCK_V,
        NUM_BLOCKS=num_blocks,
    )
    # num_warps=4 as the served kernels: it fixes the per-block reduction order.
    _residual_partials_lg_kernel[(bs, num_blocks)](
        candidates, target_logits, logit_stats, candidate_ids, q_rows, uniform_samples,
        part_sum, part_max, *strides, num_warps=4, **common,
    )
    _residual_select_lg_kernel[(bs,)](
        candidates, target_logits, logit_stats, candidate_ids, q_rows, uniform_samples,
        uniform_samples_final, part_sum, part_max, out, sync_n, sync_padded, *strides,
        num_warps=4, SYNC=sync, SYNC_VALUES=sync_values, **common,
    )


def selector_logit_accept(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    logit_stats: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
) -> torch.Tensor:
    """selector_sampled_accept from the verify logits fp32 [bs * (gamma + 1), vocab] and
    verify_softmax_wide_stats's stats [bs * (gamma + 1), 4] instead of target_probs; the same
    int64 [bs, 2] result bit for bit."""
    bs = candidates.shape[0]
    out = torch.empty((bs, 2), dtype=torch.int64, device=candidates.device)
    _selector_logit_launch(
        candidates=candidates, target_logits=target_logits, logit_stats=logit_stats,
        candidate_ids=candidate_ids, q_rows=q_rows, uniform_samples=uniform_samples,
        uniform_samples_final=uniform_samples_final, out=out, sync_n=0, sync_padded=0,
        sync=False, sync_values=False,
    )
    return out


def selector_logit_accept_synced(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    logit_stats: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    prefix_lens: torch.Tensor,
    all_reduce,
    write_values: bool,
):
    """selector_sampled_accept_synced from the verify logits and stats (selector_logit_accept's
    inputs); returns commit_selector_block's tuple, bit for bit."""
    bs, block = candidates.shape
    assert candidates.is_contiguous()
    assert target_logits.shape[-1] < (1 << 24), "fp32 sync holds token ids exactly only below 2**24"
    device = candidates.device
    n = 2 * bs
    padded = (n + 3) // 4 * 4
    sync_in = torch.empty((padded,), dtype=torch.float32, device=device)
    _selector_logit_launch(
        candidates=candidates, target_logits=target_logits, logit_stats=logit_stats,
        candidate_ids=candidate_ids, q_rows=q_rows, uniform_samples=uniform_samples,
        uniform_samples_final=uniform_samples_final, out=sync_in, sync_n=n, sync_padded=padded,
        sync=True, sync_values=bool(write_values),
    )
    summed = all_reduce(sync_in)
    num_correct = torch.empty((bs,), dtype=torch.int32, device=device)
    commit_lens = torch.empty((bs,), dtype=torch.int32, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    out_tokens = torch.empty((bs, block), dtype=torch.int64, device=device)
    new_seq_lens = torch.empty_like(prefix_lens)
    _commit_block_f32_kernel[(bs,)](
        candidates, summed, prefix_lens, num_correct, commit_lens, bonus,
        out_tokens, new_seq_lens, BLOCK=block, BLOCK_P2=triton.next_power_of_2(block),
    )
    return num_correct, commit_lens, bonus, out_tokens, new_seq_lens
