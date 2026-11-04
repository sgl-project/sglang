from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192
INT64_MIN = tl.constexpr(-(2**63))


@triton.jit
def _draft_topk1_kernel(
    logits,
    best,
    arrived,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    first_tokens,
    first_tokens_stride,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    WRITE_FIRST_TOKEN: tl.constexpr,
    ADVANCE_POSITIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One launch: each CTA reduces a vocabulary split of a row and folds its best
    # (value, lowest index) into the row's 64-bit key with an atomic max; the last
    # CTA of the row to arrive reads the key, writes the token and resets the state.
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
    max_val = tl.where(max_val == 0.0, 0.0, max_val)  # -0.0 as +0.0: equal values tie
    index = split * BLOCK + tl.argmax(vals, axis=0)
    # the value's bits ordered as signed integers, then the index reversed, so a
    # larger key is a larger value or, on a tie, a lower index
    bits = max_val.to(tl.int32, bitcast=True)
    bits = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = (bits.to(tl.int64) << 32) | (0x7FFFFFFF - index).to(tl.int64)
    tl.atomic_max(best + row, key, sem="relaxed")
    if tl.atomic_add(arrived + row, 1, sem="acq_rel") == num_splits - 1:
        key = tl.atomic_xchg(best + row, INT64_MIN, sem="acquire")
        tl.atomic_xchg(arrived + row, 0, sem="relaxed")
        token = 0x7FFFFFFF - (key & 0x7FFFFFFF)
        tl.store(topk_index + row, token)
        tl.store(topk_p + row, 1.0)
        if WRITE_DRAFT_TOKEN:
            tl.store(
                draft_tokens + row * draft_tokens_stride + draft_token_column, token
            )
        if WRITE_FIRST_TOKEN:
            first = tl.load(first_tokens + row * first_tokens_stride)
            tl.store(draft_tokens + row * draft_tokens_stride, first)
        if ADVANCE_POSITIONS:
            position = tl.load(positions + row)
            tl.store(positions + row, position + 1)


# per device: each row's best key and arrival count, left reset by every launch (the
# launches of a device run in stream order: the draft's top-1s share one stream)
_STATE: dict = {}
# outgrown states stay allocated: captured graphs keep their addresses
_OUTGROWN: list = []


def _row_state(device: torch.device, rows: int):
    state = _STATE.get(device)
    if state is None or state[0].shape[0] < rows:
        if state is not None:
            _OUTGROWN.append(state)
        n = max(rows, 1024)
        state = (
            torch.full((n,), -(2**63), dtype=torch.int64, device=device),
            torch.zeros((n,), dtype=torch.int32, device=device),
        )
        _STATE[device] = state
    return state


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor | None,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
    first_tokens: torch.Tensor | None = None,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, and the last CTA of a row finalizes its
    token in the same launch. Ties resolve to the lowest index, NaN reads as
    -1e30.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place; with ``first_tokens`` ([bs, 1], the tokens the step's rows
    took) it also stores those into ``draft_tokens[:, 0]``, in place of a copy before
    the first step. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream. With
    ``positions`` None nothing is advanced.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    advance = positions is not None
    if advance:
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
    write_first = first_tokens is not None
    if write_first:
        assert write_draft_token and draft_token_column > 0
        assert first_tokens.shape[0] == next_token_logits.shape[0]
        assert first_tokens.dtype == torch.long
        assert first_tokens.device == next_token_logits.device

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    best, arrived = _row_state(next_token_logits.device, bs)
    # Dummy operands for the disabled slots: the pointers must be valid even
    # though the kernel never dereferences them (gated off by the flags).
    _draft_topk1_kernel[(bs, num_splits)](
        next_token_logits,
        best,
        arrived,
        topk_p,
        topk_index,
        positions if advance else topk_index,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        first_tokens if write_first else topk_index,
        first_tokens.stride(0) if write_first else 0,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        WRITE_FIRST_TOKEN=write_first,
        ADVANCE_POSITIONS=advance,
        BLOCK=block,
        num_warps=8,
    )
    return topk_p, topk_index


def draft_topk1_argmax_only(next_token_logits: torch.Tensor):
    """Select top-k=1 without advancing positions (draft extend has consumed its)."""
    return draft_topk1_postprocess(next_token_logits, None)
