from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0, tie_break_left=True)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0, tie_break_left=True)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


@triton.jit
def _target_verify_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    candidates,
    retrieve_index,
    retrieve_next_token,
    seq_lens,
    predict,
    num_correct_drafts,
    accept_lens,
    accept_index,
    bonus_tokens,
    new_seq_lens,
    select_index,
    draft_input_ids,
    num_splits: tl.constexpr,
    NUM_DRAFT_TOKENS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    request_base = batch_idx * NUM_DRAFT_TOKENS
    rows = tl.arange(0, ROW_BLOCK)
    splits = tl.arange(0, SPLIT_BLOCK)

    # Finish every row reduction for this request in parallel. The accepted
    # chain then selects from these register-resident token ids.
    partial_offsets = (request_base + rows[:, None]) * num_splits + splits[None, :]
    partial_mask = (rows[:, None] < NUM_DRAFT_TOKENS) & (splits[None, :] < num_splits)
    vals = tl.load(
        partial_vals + partial_offsets,
        mask=partial_mask,
        other=-float("inf"),
    )
    best_splits = tl.argmax(vals, axis=1, tie_break_left=True)
    target_ids = tl.load(
        partial_indices + (request_base + rows) * num_splits + best_splits,
        mask=rows < NUM_DRAFT_TOKENS,
        other=0,
    )

    predict_values = tl.zeros((ROW_BLOCK,), tl.int32)
    accept_values = tl.full((ROW_BLOCK,), -1, tl.int32)

    last_global = tl.load(retrieve_index + request_base)
    current_node = tl.cast(0, last_global.dtype)
    accept_values = tl.where(rows == 0, last_global, accept_values)
    accepted = 0
    active = True

    # topk=1 produces a chain, so there are no siblings to search. Keeping an
    # active predicate preserves the early-rejection behavior without a
    # tensor-dependent loop exit.
    for _ in range(1, NUM_DRAFT_TOKENS):
        last_local = last_global - request_base
        target_id = tl.sum(tl.where(rows == last_local, target_ids, 0), axis=0)

        safe_current = tl.where(active, current_node, 0)
        next_node = tl.load(
            retrieve_next_token + request_base + safe_current,
            mask=active,
            other=-1,
        )
        valid = active & (next_node >= 0) & (next_node < NUM_DRAFT_TOKENS)
        safe_next = tl.where(valid, next_node, 0)
        candidate = tl.load(candidates + request_base + safe_next)
        next_global = tl.load(retrieve_index + request_base + safe_next)
        matched = valid & (candidate == target_id)

        predict_values = tl.where(
            matched & (rows == last_local), target_id, predict_values
        )
        next_accepted = accepted + 1
        accept_values = tl.where(
            matched & (rows == next_accepted), next_global, accept_values
        )

        accepted = tl.where(matched, next_accepted, accepted)
        current_node = tl.where(matched, safe_next, current_node)
        last_global = tl.where(matched, next_global, last_global)
        active = matched

    # The target token following the last accepted draft is the bonus token.
    last_local = last_global - request_base
    bonus_token = tl.sum(tl.where(rows == last_local, target_ids, 0), axis=0)
    predict_values = tl.where(rows == last_local, bonus_token, predict_values)

    row_mask = rows < NUM_DRAFT_TOKENS
    tl.store(predict + request_base + rows, predict_values, mask=row_mask)
    tl.store(draft_input_ids + request_base + rows, predict_values, mask=row_mask)
    tl.store(accept_index + request_base + rows, accept_values, mask=row_mask)

    accept_len = accepted + 1
    tl.store(num_correct_drafts + batch_idx, accepted)
    tl.store(accept_lens + batch_idx, accept_len)
    tl.store(bonus_tokens + batch_idx, bonus_token)
    tl.store(new_seq_lens + batch_idx, tl.load(seq_lens + batch_idx) + accept_len)
    tl.store(select_index + batch_idx, request_base + accepted)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operand for the disabled draft-token slot: the pointer must be
    # valid even though the kernel never dereferences it (gated off by
    # WRITE_DRAFT_TOKEN).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


def target_verify_topk1_postprocess(
    next_token_logits: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """Reduce target logits and finalize greedy topk=1 verification.

    This reuses the split argmax reduction used by draft decoding, then folds
    chain verification and the small tensors consumed by draft-extend into one
    finalizer launch. The caller is responsible for selecting this CUDA-only
    fast path only after penalties, grammar masks, and NaN sanitization.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.device.type == "cuda"
    assert next_token_logits.stride(1) == 1
    assert next_token_logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert candidates.ndim == 2
    assert candidates.dtype == torch.long
    assert candidates.is_contiguous()
    assert retrieve_index.shape == candidates.shape
    assert retrieve_index.dtype == torch.long
    assert retrieve_index.is_contiguous()
    assert retrieve_next_token.shape == candidates.shape
    assert retrieve_next_token.dtype == torch.long
    assert retrieve_next_token.is_contiguous()
    assert seq_lens.ndim == 1
    assert seq_lens.is_contiguous()
    assert seq_lens.dtype in (torch.int32, torch.int64)

    batch_size, num_draft_tokens = candidates.shape
    total_rows, vocab_size = next_token_logits.shape
    assert num_draft_tokens > 0
    assert vocab_size > 0
    assert total_rows == batch_size * num_draft_tokens
    assert seq_lens.shape[0] == batch_size
    assert all(
        tensor.device == next_token_logits.device
        for tensor in (candidates, retrieve_index, retrieve_next_token, seq_lens)
    )

    device = next_token_logits.device
    predict = torch.empty((total_rows,), dtype=torch.int32, device=device)
    num_correct_drafts = torch.empty((batch_size,), dtype=torch.int32, device=device)
    accept_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    accept_index = torch.empty(
        (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )
    bonus_tokens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    new_seq_lens = torch.empty_like(seq_lens)
    select_index = torch.empty((batch_size,), dtype=torch.int64, device=device)
    draft_input_ids = torch.empty((total_rows,), dtype=torch.int64, device=device)
    if batch_size == 0:
        return (
            predict,
            num_correct_drafts,
            accept_lens,
            accept_index,
            bonus_tokens,
            new_seq_lens,
            select_index,
            draft_input_ids,
        )

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (total_rows, num_splits), dtype=torch.float32, device=device
    )
    partial_indices = torch.empty(
        (total_rows, num_splits), dtype=torch.int32, device=device
    )
    _draft_topk1_partial_argmax_kernel[(total_rows, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    _target_verify_topk1_finalize_kernel[(batch_size,)](
        partial_vals,
        partial_indices,
        candidates,
        retrieve_index,
        retrieve_next_token,
        seq_lens,
        predict,
        num_correct_drafts,
        accept_lens,
        accept_index,
        bonus_tokens,
        new_seq_lens,
        select_index,
        draft_input_ids,
        num_splits,
        NUM_DRAFT_TOKENS=num_draft_tokens,
        ROW_BLOCK=triton.next_power_of_2(num_draft_tokens),
        SPLIT_BLOCK=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return (
        predict,
        num_correct_drafts,
        accept_lens,
        accept_index,
        bonus_tokens,
        new_seq_lens,
        select_index,
        draft_input_ids,
    )
