from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192
_DRAFT_TOPK1_GATHER_BLOCK = 1024


class TargetVerifyTopk1Output(NamedTuple):
    predict: torch.Tensor
    num_correct_drafts: torch.Tensor
    accept_lens: torch.Tensor
    accept_index: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    select_index: torch.Tensor
    draft_input_ids: torch.Tensor


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    row_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    USE_ROW_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    out_row = tl.program_id(0).to(tl.int64)
    src_row = (
        tl.load(row_indices + out_row).to(tl.int64) if USE_ROW_INDICES else out_row
    )
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + src_row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_val, local_index = tl.max(vals, axis=0, return_indices=True)
    out_offset = out_row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_finalize(
    partial_vals,
    partial_indices,
    row,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )
    split = tl.argmax(vals, axis=0)
    return tl.load(partial_indices + row * num_splits + split).to(tl.int64)


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
    index = _draft_topk1_finalize(
        partial_vals,
        partial_indices,
        row,
        num_splits,
        BLOCK,
    )
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


@triton.jit
def _draft_extend_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    row_indices,
    hidden_states,
    selected_hidden_states,
    hidden_states_row_stride,
    dsa_topk_indices,
    selected_dsa_topk_indices,
    dsa_topk_indices_row_stride,
    num_splits: tl.constexpr,
    hidden_size: tl.constexpr,
    dsa_topk_size: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    HAS_DSA: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    GATHER_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    task = tl.program_id(1)

    if task == 0:
        index = _draft_topk1_finalize(
            partial_vals,
            partial_indices,
            row,
            num_splits,
            REDUCE_BLOCK,
        )
        tl.store(topk_index + row, index)
        tl.store(topk_p + row, 1.0)
    else:
        src_row = tl.load(row_indices + row).to(tl.int64)
        offsets = (task - 1) * GATHER_BLOCK + tl.arange(0, GATHER_BLOCK)

        if HAS_HIDDEN:
            hidden_mask = offsets < hidden_size
            hidden = tl.load(
                hidden_states + src_row * hidden_states_row_stride + offsets,
                mask=hidden_mask,
                other=0.0,
            )
            tl.store(
                selected_hidden_states + row * hidden_size + offsets,
                hidden,
                mask=hidden_mask,
            )

        if HAS_DSA:
            dsa_mask = offsets < dsa_topk_size
            dsa_topk = tl.load(
                dsa_topk_indices + src_row * dsa_topk_indices_row_stride + offsets,
                mask=dsa_mask,
                other=0,
            )
            tl.store(
                selected_dsa_topk_indices + row * dsa_topk_size + offsets,
                dsa_topk,
                mask=dsa_mask,
            )


def _validate_logits(logits: torch.Tensor) -> None:
    assert logits.ndim == 2
    assert logits.stride(1) == 1
    assert logits.shape[1] > 0


def _validate_row_indices(
    row_indices: torch.Tensor,
    *,
    logits: torch.Tensor,
) -> None:
    assert row_indices.ndim == 1
    assert row_indices.dtype == torch.long
    assert row_indices.is_contiguous()
    assert row_indices.device == logits.device


def _validate_row_buffer(
    buffer: torch.Tensor,
    *,
    name: str,
    logits: torch.Tensor,
) -> None:
    assert buffer.ndim == 2, f"{name} must be two-dimensional"
    assert buffer.stride(1) == 1, f"{name} rows must be contiguous"
    assert buffer.shape[0] >= logits.shape[0]
    assert buffer.device == logits.device


def _empty_selected_rows(buffer: torch.Tensor | None, bs: int) -> torch.Tensor | None:
    if buffer is None:
        return None
    return buffer.new_empty((bs, buffer.shape[1]))


def _launch_draft_topk1_partials(
    next_token_logits: torch.Tensor,
    row_indices: torch.Tensor | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    use_row_indices = row_indices is not None
    bs = row_indices.shape[0] if use_row_indices else next_token_logits.shape[0]
    vocab_size = next_token_logits.shape[1]
    num_splits = triton.cdiv(vocab_size, _DRAFT_TOPK1_BLOCK)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )
    # Triton's launcher still needs a valid pointer for the disabled index mode.
    row_indices_arg = row_indices if use_row_indices else next_token_logits
    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        row_indices_arg,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        USE_ROW_INDICES=use_row_indices,
        BLOCK=_DRAFT_TOPK1_BLOCK,
        num_warps=8,
    )
    return num_splits, partial_vals, partial_indices


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
    _validate_logits(next_token_logits)
    bs = next_token_logits.shape[0]
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == bs
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == bs
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    num_splits, partial_vals, partial_indices = _launch_draft_topk1_partials(
        next_token_logits
    )

    # The pointer must be valid even when WRITE_DRAFT_TOKEN is false, although
    # the compile-time-disabled branch never dereferences it.
    draft_tokens_arg = draft_tokens if write_draft_token else topk_index
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        draft_tokens_arg,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


def target_verify_topk1_is_supported(
    next_token_logits: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    """Return whether the tensors satisfy the fused verifier's contract."""
    if next_token_logits.ndim != 2 or candidates.ndim != 2:
        return False

    batch_size, num_draft_tokens = candidates.shape
    expected_shape = candidates.shape
    index_tensors = (candidates, retrieve_index, retrieve_next_token)
    return (
        next_token_logits.device.type == "cuda"
        and next_token_logits.stride(1) == 1
        and next_token_logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and num_draft_tokens > 0
        and next_token_logits.shape[1] > 0
        and next_token_logits.shape[0] == batch_size * num_draft_tokens
        and all(
            tensor.shape == expected_shape
            and tensor.dtype == torch.long
            and tensor.is_contiguous()
            and tensor.device == next_token_logits.device
            for tensor in index_tensors
        )
        and seq_lens.shape == (batch_size,)
        and seq_lens.dtype in (torch.int32, torch.int64)
        and seq_lens.is_contiguous()
        and seq_lens.device == next_token_logits.device
    )


def target_verify_topk1_postprocess(
    next_token_logits: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    seq_lens: torch.Tensor,
) -> TargetVerifyTopk1Output:
    """Reduce target logits and finalize greedy topk=1 verification.

    This reuses the split argmax reduction used by draft decoding, then folds
    chain verification and the small tensors consumed by draft-extend into one
    finalizer launch. The caller is responsible for selecting this CUDA-only
    fast path only after penalties, grammar masks, and NaN sanitization.
    """
    assert target_verify_topk1_is_supported(
        next_token_logits,
        candidates,
        retrieve_index,
        retrieve_next_token,
        seq_lens,
    ), "Unsupported tensor shape, layout, dtype, or device"

    batch_size, num_draft_tokens = candidates.shape
    total_rows = next_token_logits.shape[0]

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
        return TargetVerifyTopk1Output(
            predict=predict,
            num_correct_drafts=num_correct_drafts,
            accept_lens=accept_lens,
            accept_index=accept_index,
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            select_index=select_index,
            draft_input_ids=draft_input_ids,
        )

    num_splits, partial_vals, partial_indices = _launch_draft_topk1_partials(
        next_token_logits
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
    return TargetVerifyTopk1Output(
        predict=predict,
        num_correct_drafts=num_correct_drafts,
        accept_lens=accept_lens,
        accept_index=accept_index,
        bonus_tokens=bonus_tokens,
        new_seq_lens=new_seq_lens,
        select_index=select_index,
        draft_input_ids=draft_input_ids,
    )


def draft_extend_topk1_postprocess(
    next_token_logits: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    hidden_states: torch.Tensor | None,
    dsa_topk_indices: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Select and postprocess the final draft-extend row for each request.

    The indexed logits rows are reduced directly, without materializing a
    selected-logits tensor. The final reduction launch also gathers the matching
    hidden-state and optional DSA rows. ``row_indices`` must be contiguous int64
    and contain valid, nonnegative logits row indices.
    """
    _validate_logits(next_token_logits)
    _validate_row_indices(row_indices, logits=next_token_logits)
    if hidden_states is not None:
        _validate_row_buffer(
            hidden_states, name="hidden_states", logits=next_token_logits
        )
    if dsa_topk_indices is not None:
        _validate_row_buffer(
            dsa_topk_indices, name="dsa_topk_indices", logits=next_token_logits
        )

    bs = row_indices.shape[0]
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    selected_hidden_states = _empty_selected_rows(hidden_states, bs)
    selected_dsa_topk_indices = _empty_selected_rows(dsa_topk_indices, bs)
    if bs == 0:
        return (
            topk_p,
            topk_index,
            selected_hidden_states,
            selected_dsa_topk_indices,
        )

    num_splits, partial_vals, partial_indices = _launch_draft_topk1_partials(
        next_token_logits, row_indices
    )
    hidden_size = hidden_states.shape[1] if hidden_states is not None else 0
    dsa_topk_size = dsa_topk_indices.shape[1] if dsa_topk_indices is not None else 0
    num_gather_blocks = max(
        triton.cdiv(hidden_size, _DRAFT_TOPK1_GATHER_BLOCK),
        triton.cdiv(dsa_topk_size, _DRAFT_TOPK1_GATHER_BLOCK),
    )

    # Disabled slots still need valid pointers for Triton's launcher.
    hidden_states_arg = hidden_states if hidden_size else next_token_logits
    selected_hidden_states_arg = selected_hidden_states if hidden_size else topk_p
    dsa_topk_indices_arg = dsa_topk_indices if dsa_topk_size else partial_indices
    selected_dsa_topk_indices_arg = (
        selected_dsa_topk_indices if dsa_topk_size else partial_indices
    )
    _draft_extend_topk1_finalize_kernel[(bs, 1 + num_gather_blocks)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        row_indices,
        hidden_states_arg,
        selected_hidden_states_arg,
        hidden_states.stride(0) if hidden_size else 0,
        dsa_topk_indices_arg,
        selected_dsa_topk_indices_arg,
        dsa_topk_indices.stride(0) if dsa_topk_size else 0,
        num_splits,
        hidden_size,
        dsa_topk_size,
        HAS_HIDDEN=hidden_size > 0,
        HAS_DSA=dsa_topk_size > 0,
        REDUCE_BLOCK=triton.next_power_of_2(num_splits),
        GATHER_BLOCK=_DRAFT_TOPK1_GATHER_BLOCK,
        num_warps=4 if num_gather_blocks else 1,
    )
    return (
        topk_p,
        topk_index,
        selected_hidden_states,
        selected_dsa_topk_indices,
    )
