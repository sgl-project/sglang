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
    partial_sumexp,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    COMPUTE_PROB: tl.constexpr,
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
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)
    if COMPUTE_PROB:
        sumexp = tl.sum(tl.exp(vals - max_val), axis=0)
        tl.store(
            partial_sumexp + out_offset, tl.where(max_val > -float("inf"), sumexp, 0.0)
        )


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    partial_sumexp,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    draft_probs,
    draft_probs_stride,
    draft_prob_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    COMPUTE_PROB: tl.constexpr,
    WRITE_DRAFT_PROB: tl.constexpr,
    ADVANCE_POSITIONS: tl.constexpr,
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

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    if COMPUTE_PROB:
        sumexps = tl.load(
            partial_sumexp + row * num_splits + offsets, mask=mask, other=0.0
        )
        max_val = tl.max(vals, axis=0)
        scale = tl.where(vals > -float("inf"), tl.exp(vals - max_val), 0.0)
        total = tl.sum(tl.where(mask, sumexps * scale, 0.0), axis=0)
        prob = tl.where(total > 0.0, 1.0 / total, 0.0)
        tl.store(topk_p + row, prob)
        if WRITE_DRAFT_PROB:
            tl.store(draft_probs + row * draft_probs_stride + draft_prob_column, prob)
    else:
        tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    if ADVANCE_POSITIONS:
        position = tl.load(positions + row)
        tl.store(positions + row, position + 1)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor | None,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
    *,
    compute_prob: bool = False,
    draft_probs: torch.Tensor | None = None,
    draft_prob_column: int = 0,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is constant 1.0 unless ``compute_prob`` is set,
    in which case it is the softmax probability of the argmax token, also stored
    into ``draft_probs[:, draft_prob_column]`` when ``draft_probs`` is given.
    ``positions`` may be None to skip advancing them.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    advance_positions = positions is not None
    if advance_positions:
        assert positions.ndim == 1
        assert positions.is_contiguous()
        assert positions.shape[0] == next_token_logits.shape[0]
        assert positions.device == next_token_logits.device
    write_draft_prob = draft_probs is not None
    if write_draft_prob:
        assert compute_prob
        assert draft_probs.ndim == 2
        assert draft_probs.dtype == torch.float32
        assert draft_probs.shape[0] >= next_token_logits.shape[0]
        assert draft_probs.stride(1) == 1
        assert 0 <= draft_prob_column < draft_probs.shape[1]
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
    partial_sumexp = (
        torch.empty(
            (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
        )
        if compute_prob
        else partial_vals
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        partial_sumexp,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        COMPUTE_PROB=compute_prob,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operands for the disabled slots: the pointers must be valid even
    # though the kernel never dereferences them (gated off by the constexprs).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        partial_sumexp,
        topk_p,
        topk_index,
        positions if advance_positions else topk_index,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        draft_probs if write_draft_prob else topk_p,
        draft_probs.stride(0) if write_draft_prob else 0,
        draft_prob_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        COMPUTE_PROB=compute_prob,
        WRITE_DRAFT_PROB=write_draft_prob,
        ADVANCE_POSITIONS=advance_positions,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


def draft_topk1_argmax_only(
    next_token_logits: torch.Tensor, *, compute_prob: bool = False
):
    """Select top-k=1 without mutating caller-owned positions."""
    return draft_topk1_postprocess(next_token_logits, None, compute_prob=compute_prob)
