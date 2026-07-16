from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192
_DRAFT_TOPK1_GATHER_BLOCK = 1024


@triton.jit
def _draft_topk1_store_partial_argmax(
    logits,
    partial_vals,
    partial_indices,
    out_row,
    src_row,
    split,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + src_row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0, tie_break_left=True)
    out_offset = out_row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


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
    out_row = tl.program_id(0).to(tl.int64)
    src_row = out_row
    split = tl.program_id(1)
    _draft_topk1_store_partial_argmax(
        logits,
        partial_vals,
        partial_indices,
        out_row,
        src_row,
        split,
        logits_row_stride,
        vocab_size,
        num_splits,
        BLOCK,
    )


@triton.jit
def _draft_extend_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    row_indices,
    logits_row_stride,
    row_indices_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_row = tl.program_id(0).to(tl.int64)
    src_row = tl.load(row_indices + out_row * row_indices_stride).to(tl.int64)
    split = tl.program_id(1)
    _draft_topk1_store_partial_argmax(
        logits,
        partial_vals,
        partial_indices,
        out_row,
        src_row,
        split,
        logits_row_stride,
        vocab_size,
        num_splits,
        BLOCK,
    )


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
    split = tl.argmax(vals, axis=0, tie_break_left=True)
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
def _draft_extend_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    row_indices,
    row_indices_stride,
    hidden_states,
    selected_hidden_states,
    hidden_states_row_stride,
    dsa_topk_indices,
    selected_dsa_topk_indices,
    dsa_topk_indices_row_stride,
    num_splits: tl.constexpr,
    num_hidden_blocks: tl.constexpr,
    num_dsa_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    dsa_topk_size: tl.constexpr,
    GATHER_HIDDEN_STATES: tl.constexpr,
    GATHER_DSA_TOPK: tl.constexpr,
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
        src_row = tl.load(row_indices + row * row_indices_stride).to(tl.int64)
        hidden_block = task - 1
        if GATHER_HIDDEN_STATES:
            if hidden_block < num_hidden_blocks:
                hidden_offsets = hidden_block * GATHER_BLOCK + tl.arange(
                    0, GATHER_BLOCK
                )
                hidden_mask = hidden_offsets < hidden_size
                hidden = tl.load(
                    hidden_states + src_row * hidden_states_row_stride + hidden_offsets,
                    mask=hidden_mask,
                )
                tl.store(
                    selected_hidden_states + row * hidden_size + hidden_offsets,
                    hidden,
                    mask=hidden_mask,
                )

        dsa_block = task - 1 - num_hidden_blocks
        if GATHER_DSA_TOPK:
            if dsa_block >= 0:
                if dsa_block < num_dsa_blocks:
                    dsa_offsets = dsa_block * GATHER_BLOCK + tl.arange(0, GATHER_BLOCK)
                    dsa_mask = dsa_offsets < dsa_topk_size
                    dsa_topk = tl.load(
                        dsa_topk_indices
                        + src_row * dsa_topk_indices_row_stride
                        + dsa_offsets,
                        mask=dsa_mask,
                    )
                    tl.store(
                        selected_dsa_topk_indices + row * dsa_topk_size + dsa_offsets,
                        dsa_topk,
                        mask=dsa_mask,
                    )


def _validate_row_indices(
    row_indices: torch.Tensor,
    *,
    logits: torch.Tensor,
) -> None:
    assert row_indices.ndim == 1
    assert row_indices.dtype in (torch.int32, torch.int64)
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


def _empty_selected_rows(buffer: torch.Tensor | None, bs: int):
    if buffer is None:
        return None
    return torch.empty((bs, buffer.shape[1]), dtype=buffer.dtype, device=buffer.device)


def _launch_draft_topk1_partials(
    next_token_logits: torch.Tensor,
    row_indices: torch.Tensor | None = None,
):
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert next_token_logits.shape[1] > 0

    use_row_indices = row_indices is not None
    if use_row_indices:
        _validate_row_indices(row_indices, logits=next_token_logits)
        bs = row_indices.shape[0]
    else:
        bs = next_token_logits.shape[0]

    vocab_size = next_token_logits.shape[1]
    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )
    if bs == 0:
        return bs, num_splits, partial_vals, partial_indices

    if use_row_indices:
        _draft_extend_topk1_partial_argmax_kernel[(bs, num_splits)](
            next_token_logits,
            partial_vals,
            partial_indices,
            row_indices,
            next_token_logits.stride(0),
            row_indices.stride(0),
            vocab_size,
            num_splits,
            BLOCK=block,
            num_warps=8,
        )
    else:
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
    return bs, num_splits, partial_vals, partial_indices


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
    bs, num_splits, partial_vals, partial_indices = _launch_draft_topk1_partials(
        next_token_logits
    )
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


def draft_extend_topk1_postprocess(
    next_token_logits: torch.Tensor,
    row_indices: torch.Tensor,
    hidden_states: torch.Tensor | None,
    dsa_topk_indices: torch.Tensor | None = None,
):
    """Select and postprocess the final draft-extend row for each request.

    The indexed logits rows are reduced directly, without materializing a
    selected-logits tensor. The final reduction launch also gathers the matching
    hidden-state and optional DSA rows.
    """
    bs, num_splits, partial_vals, partial_indices = _launch_draft_topk1_partials(
        next_token_logits, row_indices
    )
    if hidden_states is not None:
        _validate_row_buffer(
            hidden_states, name="hidden_states", logits=next_token_logits
        )
    if dsa_topk_indices is not None:
        _validate_row_buffer(
            dsa_topk_indices, name="dsa_topk_indices", logits=next_token_logits
        )

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

    gather_hidden_states = hidden_states is not None and hidden_states.shape[1] > 0
    gather_dsa_topk = dsa_topk_indices is not None and dsa_topk_indices.shape[1] > 0
    num_hidden_blocks = (
        triton.cdiv(hidden_states.shape[1], _DRAFT_TOPK1_GATHER_BLOCK)
        if gather_hidden_states
        else 0
    )
    num_dsa_blocks = (
        triton.cdiv(dsa_topk_indices.shape[1], _DRAFT_TOPK1_GATHER_BLOCK)
        if gather_dsa_topk
        else 0
    )
    # Disabled slots still need valid pointers for Triton's launcher.
    hidden_states_arg = hidden_states if gather_hidden_states else next_token_logits
    selected_hidden_states_arg = (
        selected_hidden_states if gather_hidden_states else topk_p
    )
    dsa_topk_indices_arg = dsa_topk_indices if gather_dsa_topk else partial_indices
    selected_dsa_topk_indices_arg = (
        selected_dsa_topk_indices if gather_dsa_topk else partial_indices
    )
    _draft_extend_topk1_finalize_kernel[(bs, 1 + num_hidden_blocks + num_dsa_blocks)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        row_indices,
        row_indices.stride(0),
        hidden_states_arg,
        selected_hidden_states_arg,
        hidden_states.stride(0) if gather_hidden_states else 0,
        dsa_topk_indices_arg,
        selected_dsa_topk_indices_arg,
        dsa_topk_indices.stride(0) if gather_dsa_topk else 0,
        num_splits,
        num_hidden_blocks,
        num_dsa_blocks,
        hidden_states.shape[1] if gather_hidden_states else 0,
        dsa_topk_indices.shape[1] if gather_dsa_topk else 0,
        GATHER_HIDDEN_STATES=gather_hidden_states,
        GATHER_DSA_TOPK=gather_dsa_topk,
        REDUCE_BLOCK=triton.next_power_of_2(num_splits),
        GATHER_BLOCK=_DRAFT_TOPK1_GATHER_BLOCK,
        num_warps=4 if gather_hidden_states or gather_dsa_topk else 1,
    )
    return (
        topk_p,
        topk_index,
        selected_hidden_states,
        selected_dsa_topk_indices,
    )
