"""Triton kernels for sampling penalties in chain speculative decoding.

A verify block of a request has ``W`` rows: row ``j`` predicts the token after
the anchor and the drafts ``d_1 .. d_j``, so its penalties also count those
drafts. A step keeps the block-start tables (``additive``, ``scaling``:
``[bs, V]``) and, per request, a few override tokens ``o_t`` with per-row values
(``[bs, W, T]``). Row ``j`` of request ``b`` uses

    add(b, j, v)   = override_add[b, j, t]    if v = o_bt, else additive[b, v]
    scale(b, j, v) = override_scale[b, j, t]  if v = o_bt, else scaling[b, v]

and transforms a logit ``x`` as ``SamplingBatchInfo`` does for one row:
``y = round(x + add)``, then ``y * scale`` for ``y < 0`` and ``y / scale``
otherwise, where ``round`` rounds to the logits dtype.

The kernels round like the penalizers' ops that non-speculative decoding runs,
so both give the same logits: the frequency table is advanced by sequential fp32
additions that keep subnormals, like ``index_put_(accumulate=True)`` in
``BatchedFrequencyPenalizer``, and the division is Triton's ``div.full.f32``,
which torch.compile emits for ``apply_scaling_penalties``. They launch with FP
contraction disabled so a logit bias is added after the scaled value is rounded.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_V = 2048


def _check_rows(t: torch.Tensor, bs: int, width: Optional[int] = None) -> None:
    """The kernels read row ``b`` of ``t`` at ``b * t.stride(0)``, with its
    elements adjacent."""
    assert (
        t.dim() == 2
        and t.shape[0] == bs
        and t.stride(1) == 1
        and (width is None or t.shape[1] == width)
    ), (tuple(t.shape), t.stride(), bs, width)


def _check_per_request(t: torch.Tensor, bs: int) -> None:
    """The kernels read request ``b``'s value at element ``b``."""
    assert t.numel() == bs and t.is_contiguous(), (tuple(t.shape), t.stride(), bs)


@triton.jit
def _cumulate_padded_tokens_kernel(
    table_ptr,
    table_stride,
    coef_ptr,
    coef_stride,
    tokens_ptr,
    tokens_stride,
    lens_ptr,
    width,
    vocab_size,
    ACCUMULATE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.minimum(tl.load(lens_ptr + row), width)
    scalar_coef = tl.load(coef_ptr + row * coef_stride)
    table_row = table_ptr + row.to(tl.int64) * table_stride
    if ACCUMULATE:
        # One token after another, with plain fp32 additions: atomics would flush
        # a subnormal coefficient to zero, where index_put_ keeps it.
        for j in range(0, width):
            token = tl.load(tokens_ptr + row * tokens_stride + j).to(tl.int64)
            hit = (j < n) & (token >= 0) & (token < vocab_size)
            ptr = table_row + tl.where(hit, token, 0)
            count = tl.load(ptr, mask=hit, other=0.0)
            tl.store(ptr, count + scalar_coef, mask=hit)
    else:
        cols = tl.arange(0, BLOCK_W)
        tokens = tl.load(
            tokens_ptr + row * tokens_stride + cols, mask=cols < n, other=-1
        ).to(tl.int64)
        valid = (cols < n) & (tokens >= 0) & (tokens < vocab_size)
        coef = tl.full([BLOCK_W], scalar_coef, tl.float32)
        tl.store(table_row + tokens, coef, mask=valid)


def cumulate_padded_tokens(
    table: torch.Tensor,
    coefficients: torch.Tensor,
    tokens: torch.Tensor,
    lens: torch.Tensor,
    *,
    accumulate: bool,
) -> None:
    """Count ``tokens[b, :lens[b]]`` in the fp32 ``[bs, V]`` ``table``.

    Each valid token adds (``accumulate``) or writes ``coefficients[b]`` at its
    column; tokens outside ``[0, V)`` are skipped. Equivalent to one
    ``index_put_`` (accumulating, or not) per column, in column order.
    """
    bs, width = tokens.shape
    _check_rows(tokens, bs)
    _check_rows(table, bs)
    _check_per_request(lens, bs)
    assert coefficients.shape[0] == bs, (tuple(coefficients.shape), bs)
    if bs == 0 or width == 0:
        return
    _cumulate_padded_tokens_kernel[(bs,)](
        table,
        table.stride(0),
        coefficients,
        coefficients.stride(0),
        tokens,
        tokens.stride(0),
        lens,
        width,
        table.shape[1],
        ACCUMULATE=accumulate,
        BLOCK_W=triton.next_power_of_2(width),
    )


# Arguments that change with a batch's draft and stop-id counts are not
# specialized, so serving does not compile a variant per count.
@triton.jit(
    do_not_specialize=[
        "drafts_stride",
        "stop_ids_stride",
        "freq_stride",
        "pres_stride",
        "rep_stride",
        "stop_pen_stride",
        "max_drafts",
        "num_slots",
    ]
)
def _chain_overrides_kernel(
    drafts_ptr,
    drafts_stride,
    num_drafts_ptr,
    stop_ids_ptr,
    stop_ids_stride,
    freq_coef_ptr,
    freq_table_ptr,
    freq_stride,
    pres_coef_ptr,
    pres_table_ptr,
    pres_stride,
    rep_coef_ptr,
    rep_table_ptr,
    rep_stride,
    min_new_ptr,
    out_len_ptr,
    stop_pen_ptr,
    stop_pen_stride,
    out_tokens_ptr,
    out_add_ptr,
    out_scale_ptr,
    max_drafts,
    num_slots,
    vocab_size,
    HAS_NUM_DRAFTS: tl.constexpr,
    HAS_FREQ: tl.constexpr,
    HAS_PRES: tl.constexpr,
    HAS_REP: tl.constexpr,
    HAS_MIN_NEW: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    in_slots = t < num_slots
    num_drafts = max_drafts
    if HAS_NUM_DRAFTS:
        num_drafts = tl.minimum(tl.load(num_drafts_ptr + b), max_drafts)
    # Slots 0..K-1 hold the drafts, K..K+S-1 the stop tokens; a token in two
    # slots gets the same values in both.
    tokens = tl.load(
        drafts_ptr + b * drafts_stride + t, mask=t < num_drafts, other=-1
    ).to(tl.int64)
    if HAS_MIN_NEW:
        stop_slot = (t >= max_drafts) & in_slots
        stop_tokens = tl.load(
            stop_ids_ptr + b * stop_ids_stride + (t - max_drafts),
            mask=stop_slot,
            other=-1,
        ).to(tl.int64)
        tokens = tl.where(stop_slot, stop_tokens, tokens)
    valid = in_slots & (tokens >= 0) & (tokens < vocab_size)
    safe = tl.where(valid, tokens, 0)

    if HAS_FREQ:
        freq_coef = tl.load(freq_coef_ptr + b)
        freq = tl.load(freq_table_ptr + b * freq_stride + safe)
    if HAS_PRES:
        pres_coef = tl.load(pres_coef_ptr + b)
        presence = tl.load(pres_table_ptr + b * pres_stride + safe)
    if HAS_REP:
        rep_coef = tl.load(rep_coef_ptr + b)
        rep = tl.load(rep_table_ptr + b * rep_stride + safe)
    if HAS_MIN_NEW:
        min_new = tl.load(min_new_ptr + b)
        out_len = tl.load(out_len_ptr + b)
        is_stop = tl.load(stop_pen_ptr + b * stop_pen_stride + safe) == float("-inf")

    seen = t < 0
    for j in range(0, max_drafts + 1):
        if j > 0:
            draft = tl.load(drafts_ptr + b * drafts_stride + (j - 1)).to(tl.int64)
            counted = (j - 1 < num_drafts) & (draft >= 0) & (draft < vocab_size)
            hit = valid & (tokens == draft) & counted
            if HAS_FREQ:
                freq = tl.where(hit, freq + freq_coef, freq)
            seen = seen | hit
        # The same ops, in the same order, as update_penalties' accumulation.
        add = tl.zeros([BLOCK_T], dtype=tl.float32)
        if HAS_FREQ:
            add = add - freq
        if HAS_PRES:
            add = add - tl.where(seen, pres_coef, presence)
        if HAS_MIN_NEW:
            add = add + tl.where(is_stop & (out_len + j < min_new), float("-inf"), 0.0)
        out = (b * (max_drafts + 1) + j) * num_slots + t
        tl.store(out_add_ptr + out, add, mask=in_slots)
        if HAS_REP:
            tl.store(out_scale_ptr + out, tl.where(seen, rep_coef, rep), mask=in_slots)
    tl.store(
        out_tokens_ptr + b * num_slots + t,
        tl.where(valid, tokens, -1).to(tl.int32),
        mask=in_slots,
    )


@dataclasses.dataclass(frozen=True)
class TokenPenalty:
    """A frequency, presence or repetition penalizer's state."""

    coefficients: torch.Tensor  # [bs, 1]
    table: torch.Tensor  # [bs, V]


@dataclasses.dataclass(frozen=True)
class MinNewTokensState:
    """The min_new_tokens penalizer's state."""

    min_new_tokens: torch.Tensor  # [bs, 1]
    output_lengths: torch.Tensor  # [bs, 1]
    stop_penalties: torch.Tensor  # [bs, V], -inf at the stop tokens
    stop_token_ids: torch.Tensor  # [bs, S], padded with -1


def chain_penalty_overrides(
    draft_tokens: torch.Tensor,
    num_drafts: Optional[torch.Tensor],
    *,
    vocab_size: int,
    frequency: Optional[TokenPenalty],
    presence: Optional[TokenPenalty],
    repetition: Optional[TokenPenalty],
    min_new_tokens: Optional[MinNewTokensState],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The override tokens ``[bs, T]`` (int32, -1 for an empty slot) and their
    per-row additive and scaling values ``[bs, K + 1, T]`` (fp32) of a block with
    ``K`` drafts per request (``num_drafts[b]`` of them valid)."""
    bs, max_drafts = draft_tokens.shape
    _check_rows(draft_tokens, bs)
    if num_drafts is not None:
        _check_per_request(num_drafts, bs)
    for penalty in (frequency, presence, repetition):
        if penalty is not None:
            _check_per_request(penalty.coefficients, bs)
            _check_rows(penalty.table, bs, vocab_size)
    if min_new_tokens is not None:
        _check_per_request(min_new_tokens.min_new_tokens, bs)
        _check_per_request(min_new_tokens.output_lengths, bs)
        _check_rows(min_new_tokens.stop_penalties, bs, vocab_size)
        _check_rows(min_new_tokens.stop_token_ids, bs)
    num_stop_ids = (
        0 if min_new_tokens is None else min_new_tokens.stop_token_ids.shape[1]
    )
    num_slots = max_drafts + num_stop_ids
    device = draft_tokens.device
    tokens = torch.empty((bs, num_slots), dtype=torch.int32, device=device)
    additive = torch.empty(
        (bs, max_drafts + 1, num_slots), dtype=torch.float32, device=device
    )
    scaling = None if repetition is None else torch.empty_like(additive)
    if bs == 0 or num_slots == 0:
        return tokens, additive, scaling

    dummy = additive

    def coefficients_and_table(penalty: Optional[TokenPenalty]):
        return (
            (dummy, dummy) if penalty is None else (penalty.coefficients, penalty.table)
        )

    freq_coef, freq_table = coefficients_and_table(frequency)
    pres_coef, pres_table = coefficients_and_table(presence)
    rep_coef, rep_table = coefficients_and_table(repetition)
    min_new = out_len = stop_penalties = stop_ids = dummy
    if min_new_tokens is not None:
        min_new = min_new_tokens.min_new_tokens
        out_len = min_new_tokens.output_lengths
        stop_penalties = min_new_tokens.stop_penalties
        stop_ids = min_new_tokens.stop_token_ids
    _chain_overrides_kernel[(bs,)](
        draft_tokens,
        draft_tokens.stride(0),
        num_drafts if num_drafts is not None else dummy,
        stop_ids,
        stop_ids.stride(0),
        freq_coef,
        freq_table,
        freq_table.stride(0),
        pres_coef,
        pres_table,
        pres_table.stride(0),
        rep_coef,
        rep_table,
        rep_table.stride(0),
        min_new,
        out_len,
        stop_penalties,
        stop_penalties.stride(0),
        tokens,
        additive,
        scaling if scaling is not None else dummy,
        max_drafts,
        num_slots,
        vocab_size,
        HAS_NUM_DRAFTS=num_drafts is not None,
        HAS_FREQ=frequency is not None,
        HAS_PRES=presence is not None,
        HAS_REP=repetition is not None,
        HAS_MIN_NEW=min_new_tokens is not None,
        BLOCK_T=triton.next_power_of_2(num_slots),
    )
    return tokens, additive, scaling


@triton.jit(do_not_specialize=["width", "num_slots"])
def _apply_chain_penalties_kernel(
    logits_ptr,
    logits_stride,
    add_ptr,
    add_stride,
    scale_ptr,
    scale_stride,
    bias_ptr,
    bias_stride,
    ovr_tokens_ptr,
    ovr_add_ptr,
    ovr_scale_ptr,
    width,
    num_slots,
    vocab_size,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    v0 = tl.program_id(1) * BLOCK_V
    b = row // width
    cols = v0 + tl.arange(0, BLOCK_V)
    mask = cols < vocab_size
    logit_ptrs = logits_ptr + row.to(tl.int64) * logits_stride + cols
    x = tl.load(logit_ptrs, mask=mask, other=0.0)
    add = tl.load(add_ptr + b.to(tl.int64) * add_stride + cols, mask=mask, other=0.0)
    if HAS_SCALE:
        scale = tl.load(
            scale_ptr + b.to(tl.int64) * scale_stride + cols, mask=mask, other=1.0
        )

    t = tl.arange(0, BLOCK_T)
    slot_tokens = tl.load(
        ovr_tokens_ptr + b * num_slots + t, mask=t < num_slots, other=-1
    )
    in_tile = (slot_tokens >= v0) & (slot_tokens < v0 + BLOCK_V)
    if tl.max(in_tile.to(tl.int32), axis=0) > 0:
        for k in range(0, num_slots):
            hit = cols == tl.load(ovr_tokens_ptr + b * num_slots + k)
            add = tl.where(hit, tl.load(ovr_add_ptr + row * num_slots + k), add)
            if HAS_SCALE:
                scale = tl.where(
                    hit, tl.load(ovr_scale_ptr + row * num_slots + k), scale
                )

    y = (x.to(tl.float32) + add).to(x.dtype).to(tl.float32)
    if HAS_SCALE:
        y = tl.where(y < 0, y * scale, y / scale)
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + b.to(tl.int64) * bias_stride + cols, mask=mask, other=0.0
        )
        y = y.to(x.dtype).to(tl.float32) + bias
    tl.store(logit_ptrs, y.to(x.dtype), mask=mask)


def apply_chain_penalties_(
    logits: torch.Tensor,
    additive: torch.Tensor,
    scaling: Optional[torch.Tensor],
    override_tokens: torch.Tensor,
    override_additive: torch.Tensor,
    override_scaling: Optional[torch.Tensor],
    logit_bias: Optional[torch.Tensor] = None,
) -> None:
    """Penalize the ``[bs * W, V]`` verify logits in place (see the module
    docstring), then add ``logit_bias[b]`` when given."""
    rows, vocab_size = logits.shape
    bs, width, num_slots = override_additive.shape
    _check_rows(logits, bs * width)
    for table in (additive, scaling, logit_bias):
        if table is not None:
            _check_rows(table, bs, vocab_size)
    assert (scaling is None) == (override_scaling is None)
    # The kernel indexes the overrides as dense arrays.
    assert override_tokens.shape == (bs, num_slots) and override_tokens.is_contiguous()
    for values in (override_additive, override_scaling):
        if values is not None:
            assert values.shape == (bs, width, num_slots) and values.is_contiguous()
    if rows == 0:
        return
    dummy = additive
    _apply_chain_penalties_kernel[(rows, triton.cdiv(vocab_size, _BLOCK_V))](
        logits,
        logits.stride(0),
        additive,
        additive.stride(0),
        scaling if scaling is not None else dummy,
        scaling.stride(0) if scaling is not None else 0,
        logit_bias if logit_bias is not None else dummy,
        logit_bias.stride(0) if logit_bias is not None else 0,
        override_tokens,
        override_additive,
        override_scaling if override_scaling is not None else dummy,
        width,
        num_slots,
        vocab_size,
        HAS_SCALE=scaling is not None,
        HAS_BIAS=logit_bias is not None,
        BLOCK_V=_BLOCK_V,
        BLOCK_T=max(triton.next_power_of_2(num_slots), 2),
        num_warps=8,
        enable_fp_fusion=False,
    )
