"""Chain speculative sampling with the vocabulary split over CTAs.

``chain_speculative_sampling_triton`` runs one CTA a request that walks the whole
vocabulary (202k for Muse Spark) twice for the final sample, after a one-CTA-a-row
softmax: 340 us a step at batch 1 on MI355X, the same latency at batch 8. Here the
same decision is made in four small launches, each spread over ``SPLITS`` CTAs a row
or request: the rows' softmax statistics by vocabulary chunk, the sequential accept
walk (scalar loads, p from the statistics), the residual mass by chunk, and the
inverse-CDF sample in the chunk that holds the target. The target probabilities are
never materialized. Chain verification only (``block_verification`` keeps the
original kernel); the accept semantics and the residual fallbacks follow
``reject_sampling.speculative_sampling_classic_kernel``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SPLITS = 64  # vocabulary chunks a row: CTAs a row / request in the split launches


@triton.jit
def _row_stats_kernel(
    logits_ptr, temp_ptr, stats_ptr, vocab, rows_per_request, logits_row_stride,
    SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """stats[row, s] = (max, sum exp(x - max)) of chunk ``s`` of row ``row``'s
    logits / temperature."""
    row = tl.program_id(0)
    s = tl.program_id(1)
    temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_row_stride
    chunk = tl.cdiv(vocab, SPLITS)
    start = s * chunk
    end = tl.minimum(start + chunk, vocab)
    m = -float("inf")
    z = 0.0
    for v0 in range(start, end, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < end
        x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32) / temp
        m_new = tl.maximum(m, tl.max(x, axis=0))
        e = tl.where(mask, tl.exp(x - m_new), 0.0)
        z = z * tl.exp(m - m_new) + tl.sum(e, axis=0)
        m = m_new
    tl.store(stats_ptr + (row * SPLITS + s) * 2, m)
    tl.store(stats_ptr + (row * SPLITS + s) * 2 + 1, z)


@triton.jit
def _row_norm(stats_ptr, row, SPLITS: tl.constexpr):
    """(M, Z) of a row from its chunks' (max, sum): softmax = exp(x - M) / Z."""
    offs = tl.arange(0, SPLITS)
    ms = tl.load(stats_ptr + (row * SPLITS + offs) * 2)
    zs = tl.load(stats_ptr + (row * SPLITS + offs) * 2 + 1)
    M = tl.max(ms, axis=0)
    Z = tl.sum(zs * tl.exp(ms - M), axis=0)
    return M, Z


@triton.jit
def _accept_kernel(
    predicts_ptr, accept_index_ptr, accept_num_ptr, sel_row_ptr, last_idx_ptr,
    cand_ptr, idx_ptr, uni_ptr, logits_ptr, temp_ptr, stats_ptr, draft_ptr,
    stride_cand_b, stride_cand_s, stride_idx_b, stride_idx_s, stride_uni_b, stride_uni_s,
    logits_row_stride, stride_dp_b, stride_dp_s, stride_dp_v,
    NUM_SLOTS: tl.constexpr, SPLITS: tl.constexpr,
):  # fmt: skip
    """The chain accept walk of request ``pid``: slot ``step`` is accepted while coin
    q < p (p = softmax of target row ``step - 1``, from the row statistics); the
    accepted tokens to ``predicts`` at the retrieve indices, the accept index row,
    the count, the row the final sample draws from and its retrieve index."""
    pid = tl.program_id(0)
    temp = tl.load(temp_ptr + pid).to(tl.float32)
    root = tl.load(idx_ptr + pid * stride_idx_b)
    tl.store(accept_index_ptr + pid * stride_idx_b, root)
    last = root
    num_accept = 0
    cur_row = 0
    step = 1
    cont = 1
    while (step < NUM_SLOTS) and (cont == 1):
        tok = tl.load(cand_ptr + pid * stride_cand_b + step * stride_cand_s)
        row = pid * NUM_SLOTS + cur_row
        M, Z = _row_norm(stats_ptr, row, SPLITS)
        x = (
            tl.load(logits_ptr + row.to(tl.int64) * logits_row_stride + tok).to(
                tl.float32
            )
            / temp
        )
        p = tl.exp(x - M) / Z
        q = tl.load(
            draft_ptr + pid * stride_dp_b + cur_row * stride_dp_s + tok * stride_dp_v
        )
        coin = tl.load(uni_ptr + pid * stride_uni_b + (step - 1) * stride_uni_s)
        q_is_prob = (q > 0.0) & (q <= 1.0)
        if q_is_prob & (coin * q < p):
            num_accept += 1
            cur_row = step
            tl.store(predicts_ptr + last, tok)
            last = tl.load(idx_ptr + pid * stride_idx_b + step * stride_idx_s)
            tl.store(
                accept_index_ptr + pid * stride_idx_b + num_accept * stride_idx_s, last
            )
            step += 1
        else:
            cont = 0
    tl.store(accept_num_ptr + pid, num_accept)
    tl.store(sel_row_ptr + pid, cur_row)
    tl.store(last_idx_ptr + pid, last)


@triton.jit
def _residual_vals(
    logits_ptr, draft_ptr, offs, mask, row, drow, pid, temp, M, Z, all_accepted,
    logits_row_stride, stride_dp_b, stride_dp_s, stride_dp_v,
):  # fmt: skip
    """The final sample's unnormalized weights on ``offs``: p when every draft was
    accepted, else max(p - q, 0) with a non-probability q read as 0."""
    x = (
        tl.load(
            logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        / temp
    )
    p = tl.where(mask, tl.exp(x - M) / Z, 0.0)
    q = tl.load(
        draft_ptr + pid * stride_dp_b + drow * stride_dp_s + offs * stride_dp_v,
        mask=mask & (all_accepted == 0),
        other=0.0,
    )
    q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
    diff = p - q
    return tl.where(all_accepted == 1, p, tl.where(diff > 0.0, diff, 0.0))


@triton.jit
def _residual_part_kernel(
    part_ptr, sel_row_ptr, logits_ptr, temp_ptr, stats_ptr, draft_ptr, vocab,
    logits_row_stride, stride_dp_b, stride_dp_s, stride_dp_v,
    NUM_SLOTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """part[pid, s] = the residual mass of request ``pid``'s final-sample row on
    vocabulary chunk ``s``."""
    pid = tl.program_id(0)
    s = tl.program_id(1)
    cur_row = tl.load(sel_row_ptr + pid)
    all_accepted = (cur_row == NUM_SLOTS - 1).to(tl.int32)
    drow = tl.minimum(
        cur_row, NUM_SLOTS - 2
    )  # draft rows: NUM_SLOTS - 1 (unused when all accepted)
    row = pid * NUM_SLOTS + cur_row
    temp = tl.load(temp_ptr + pid).to(tl.float32)
    M, Z = _row_norm(stats_ptr, row, SPLITS)
    chunk = tl.cdiv(vocab, SPLITS)
    start = s * chunk
    end = tl.minimum(start + chunk, vocab)
    acc = 0.0
    for v0 in range(start, end, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        mask = offs < end
        val = _residual_vals(logits_ptr, draft_ptr, offs, mask, row, drow, pid, temp, M, Z, all_accepted, logits_row_stride, stride_dp_b, stride_dp_s, stride_dp_v)  # fmt: skip
        acc += tl.sum(val, axis=0)
    tl.store(part_ptr + pid * SPLITS + s, acc)


@triton.jit
def _residual_sample_kernel(
    predicts_ptr, last_idx_ptr, part_ptr, sel_row_ptr, uni_final_ptr, logits_ptr, temp_ptr, stats_ptr, draft_ptr, vocab,
    logits_row_stride, stride_dp_b, stride_dp_s, stride_dp_v,
    NUM_SLOTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """The chunk holding coin x total mass finds the first token whose cumulative
    mass exceeds it and writes it as request ``pid``'s final token; a target at or
    past the total (an empty residual) falls back to vocab - 1 like the original."""
    pid = tl.program_id(0)
    s = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS)
    parts = tl.load(part_ptr + pid * SPLITS + offs_s)
    total = tl.sum(parts, axis=0)
    prefix = tl.sum(tl.where(offs_s < s, parts, 0.0), axis=0)
    mine = tl.load(part_ptr + pid * SPLITS + s)
    target = tl.load(uni_final_ptr + pid) * total
    last = tl.load(last_idx_ptr + pid)
    hit = (prefix <= target) & (target < prefix + mine)
    if hit:
        cur_row = tl.load(sel_row_ptr + pid)
        all_accepted = (cur_row == NUM_SLOTS - 1).to(tl.int32)
        drow = tl.minimum(cur_row, NUM_SLOTS - 2)
        row = pid * NUM_SLOTS + cur_row
        temp = tl.load(temp_ptr + pid).to(tl.float32)
        M, Z = _row_norm(stats_ptr, row, SPLITS)
        chunk = tl.cdiv(vocab, SPLITS)
        start = s * chunk
        end = tl.minimum(start + chunk, vocab)
        cum = prefix
        token = vocab - 1
        found = 0
        for v0 in range(start, end, BLOCK):
            if found == 0:
                offs = v0 + tl.arange(0, BLOCK)
                mask = offs < end
                val = _residual_vals(logits_ptr, draft_ptr, offs, mask, row, drow, pid, temp, M, Z, all_accepted, logits_row_stride, stride_dp_b, stride_dp_s, stride_dp_v)  # fmt: skip
                above = (cum + tl.cumsum(val, axis=0)) > target
                if tl.max(above.to(tl.int32), axis=0) > 0:
                    token = v0 + tl.argmax(above.to(tl.int32), axis=0)
                    found = 1
                cum += tl.sum(val, axis=0)
        tl.store(predicts_ptr + last, token)
    if (s == SPLITS - 1) & (target >= total):
        tl.store(predicts_ptr + last, vocab - 1)


def chain_speculative_sampling_split(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_logits: torch.Tensor,
    temperatures: torch.Tensor,
    draft_probs: torch.Tensor,
) -> None:
    """``chain_speculative_sampling_triton`` from the target logits and the
    requests' temperatures (the softmax folded in), the vocabulary split over
    SPLITS CTAs. ``target_logits`` [bs x num_slots, vocab]; ``draft_probs``
    [bs, num_slots - 1, vocab]."""
    bs, num_slots = candidates.shape
    vocab = target_logits.shape[-1]
    device = candidates.device
    rows = bs * num_slots
    assert target_logits.shape[0] == rows and target_logits.stride(1) == 1
    temps = temperatures.reshape(-1)[:bs].to(torch.float32).contiguous()
    stats = torch.empty((rows, SPLITS, 2), dtype=torch.float32, device=device)
    part = torch.empty((bs, SPLITS), dtype=torch.float32, device=device)
    sel_row = torch.empty((bs,), dtype=torch.int32, device=device)
    last_idx = torch.empty((bs,), dtype=torch.int64, device=device)
    block = min(4096, triton.next_power_of_2(triton.cdiv(vocab, SPLITS)))
    _row_stats_kernel[(rows, SPLITS)](
        target_logits, temps, stats, vocab, num_slots, target_logits.stride(0),
        SPLITS=SPLITS, BLOCK=block, num_warps=4,
    )  # fmt: skip
    _accept_kernel[(bs,)](
        predicts, accept_index, accept_token_num, sel_row, last_idx,
        candidates, retrive_index, uniform_samples, target_logits, temps, stats, draft_probs,
        candidates.stride(0), candidates.stride(1), retrive_index.stride(0), retrive_index.stride(1),
        uniform_samples.stride(0), uniform_samples.stride(1), target_logits.stride(0),
        draft_probs.stride(0), draft_probs.stride(1), draft_probs.stride(2),
        NUM_SLOTS=num_slots, SPLITS=SPLITS, num_warps=1,
    )  # fmt: skip
    _residual_part_kernel[(bs, SPLITS)](
        part, sel_row, target_logits, temps, stats, draft_probs, vocab,
        target_logits.stride(0), draft_probs.stride(0), draft_probs.stride(1), draft_probs.stride(2),
        NUM_SLOTS=num_slots, SPLITS=SPLITS, BLOCK=block, num_warps=4,
    )  # fmt: skip
    _residual_sample_kernel[(bs, SPLITS)](
        predicts, last_idx, part, sel_row, uniform_samples_for_final_sampling, target_logits, temps, stats, draft_probs, vocab,
        target_logits.stride(0), draft_probs.stride(0), draft_probs.stride(1), draft_probs.stride(2),
        NUM_SLOTS=num_slots, SPLITS=SPLITS, BLOCK=block, num_warps=4,
    )  # fmt: skip
