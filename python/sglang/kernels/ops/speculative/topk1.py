from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

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
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
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

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


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


def draft_topk1_argmax_only(next_token_logits: torch.Tensor):
    """Select top-k=1 without mutating caller-owned positions."""
    # The upstream finalizer always advances positions. Draft-extend has already
    # consumed its positions, so absorb that side effect in a discarded buffer.
    scratch_positions = torch.zeros(
        next_token_logits.shape[0],
        dtype=torch.long,
        device=next_token_logits.device,
    )
    return draft_topk1_postprocess(next_token_logits, scratch_positions)


# Vocab-parallel top-1 for topk=1 drafts: each TP rank reduces its LM-head shard
# to one candidate per (rank, split), a sum all-reduce stands in for the
# all-gather of the candidates, and a finalizer picks the global argmax.
#
# A candidate is 4 fp32 lanes: value bits >> 16, value bits & 0xffff, global
# index, first NaN index (or -1). Every lane is an integer below 2**24 and each
# candidate slot has exactly one nonzero contributor, so any fp32 sum all-reduce
# (Lamport sentinels, FTZ, reduction order) returns it bit for bit.
DRAFT_TOPK1_PARTIAL_LANES = 4
_DRAFT_TOPK1_SHARD_BLOCK = 8192


@triton.jit
def _draft_topk1_shard_partials_kernel(
    logits,
    partials,
    logits_row_stride,
    partials_row_stride,
    local_width,
    col_base,
    vocab_size,
    tp_rank,
    num_splits,
    TP_SIZE: tl.constexpr,
    LANES: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    local = split * BLOCK + tl.arange(0, BLOCK)
    col = col_base + local
    # Columns past vocab_size are LM-head padding, which the gathered path slices off.
    mask = (local < local_width) & (col < vocab_size)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()  # the logits are the LM-head GEMM's output
    vals = tl.load(
        logits + row * logits_row_stride + local,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    if USE_PDL:
        # The dependent all-reduce still waits for this grid before reading.
        tl.extra.cuda.gdc_launch_dependents()
    is_nan = vals != vals
    first_nan = tl.min(tl.where(is_nan, col, 2147483647), axis=0)
    # Same NaN mapping as _draft_topk1_partial_argmax_kernel.
    vals = tl.where(is_nan, -1e30, vals)
    max_val = tl.max(vals, axis=0)
    arg = tl.argmax(vals, axis=0)

    bits = max_val.to(tl.uint32, bitcast=True)
    hi = (bits >> 16).to(tl.float32)
    lo = (bits & 0xFFFF).to(tl.float32)
    index = (col_base + split * BLOCK + arg).to(tl.float32)
    nan_index = tl.where(first_nan < 2147483647, first_nan, -1).to(tl.float32)

    lane = tl.arange(0, LANES)
    owner = lane // 4
    kind = lane % 4
    payload = tl.where(
        kind == 0,
        hi,
        tl.where(kind == 1, lo, tl.where(kind == 2, index, nan_index)),
    )
    # Other ranks' slots are written as +0 so the all-reduce sum is the owner's lanes.
    payload = tl.where(owner == tp_rank, payload, 0.0)
    tl.store(
        partials + row * partials_row_stride + (owner * num_splits + split) * 4 + kind,
        payload,
        mask=owner < TP_SIZE,
    )


@triton.jit
def _draft_topk1_partials_finalize_kernel(
    partials,
    partials_row_stride,
    num_candidates,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    NAN_FIRST: tl.constexpr,
    ADVANCE_POSITIONS: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cand = tl.arange(0, BLOCK)
    mask = cand < num_candidates
    base = partials + row * partials_row_stride + cand * 4
    if USE_PDL:
        # Waits on the all-reduce; no early trigger, so the next step's readers
        # of positions and draft_tokens keep plain completion order.
        tl.extra.cuda.gdc_wait()
    hi = tl.load(base, mask=mask, other=0.0).to(tl.uint32)
    lo = tl.load(base + 1, mask=mask, other=0.0).to(tl.uint32)
    vals = ((hi << 16) | lo).to(tl.float32, bitcast=True)
    vals = tl.where(mask, vals, -float("inf"))
    # Candidates are ordered by vocab range, so the leftmost max is the lowest index.
    best = tl.argmax(vals, axis=0)
    index = tl.load(partials + row * partials_row_stride + best * 4 + 2).to(tl.int64)
    if NAN_FIRST:
        # torch.argmax order: the lowest-index NaN wins over every number.
        nan_index = tl.load(base + 3, mask=mask, other=-1.0)
        nan_index = tl.where(nan_index >= 0, nan_index, float("inf"))
        first_nan = tl.min(nan_index, axis=0)
        index = tl.where(first_nan < float("inf"), first_nan.to(tl.int64), index)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)
    if ADVANCE_POSITIONS:
        position = tl.load(positions + row)
        tl.store(positions + row, position + 1)


def draft_topk1_shard_partials_width(*, local_width: int, tp_size: int) -> int:
    """Columns of ``draft_topk1_shard_partials`` for a ``local_width`` shard."""
    num_splits = triton.cdiv(local_width, _DRAFT_TOPK1_SHARD_BLOCK)
    return tp_size * num_splits * DRAFT_TOPK1_PARTIAL_LANES


def draft_topk1_shard_partials(
    logits: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    vocab_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-1 candidates of this rank's vocab shard, laid out for a sum all-reduce.

    ``logits`` is ``[rows, local_width]``, the LM-head output of rank ``tp_rank``
    whose columns are global vocab ``[tp_rank * local_width, ...)``, the layout
    the TP all-gather along the last dim produces. Returns (or fills ``out``
    with) an fp32 ``[rows, draft_topk1_shard_partials_width(...)]`` tensor
    holding this rank's candidates and zeros in every other rank's slots; after
    a sum all-reduce over the TP group, ``draft_topk1_postprocess_from_partials``
    / ``draft_topk1_argmax_from_partials`` select the same token as the existing
    finalizers on the gathered fp32 logits ``[:, :vocab_size]``.
    """
    assert logits.ndim == 2 and logits.stride(1) == 1
    assert 0 <= tp_rank < tp_size
    rows, local_width = logits.shape
    # Lanes carry indices as exact fp32 integers.
    assert tp_size * local_width < (1 << 24)
    block = _DRAFT_TOPK1_SHARD_BLOCK
    num_splits = triton.cdiv(local_width, block)
    width = draft_topk1_shard_partials_width(local_width=local_width, tp_size=tp_size)
    if out is None:
        partials = torch.empty((rows, width), dtype=torch.float32, device=logits.device)
    else:
        assert out.shape == (rows, width) and out.dtype == torch.float32
        assert out.stride(1) == 1 and out.device == logits.device
        partials = out
    if rows == 0:
        return partials
    use_pdl = is_arch_support_pdl()
    _draft_topk1_shard_partials_kernel[(rows, num_splits)](
        logits,
        partials,
        logits.stride(0),
        partials.stride(0),
        local_width,
        tp_rank * local_width,
        vocab_size,
        tp_rank,
        num_splits,
        TP_SIZE=tp_size,
        LANES=triton.next_power_of_2(tp_size * DRAFT_TOPK1_PARTIAL_LANES),
        BLOCK=block,
        USE_PDL=use_pdl,
        num_warps=8,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return partials


def _launch_partials_finalize(
    partials: torch.Tensor,
    *,
    nan_first: bool,
    positions: torch.Tensor | None,
    draft_tokens: torch.Tensor | None,
    draft_token_column: int,
):
    assert partials.ndim == 2 and partials.dtype == torch.float32
    assert partials.stride(1) == 1
    assert partials.shape[1] % DRAFT_TOPK1_PARTIAL_LANES == 0
    bs = partials.shape[0]
    num_candidates = partials.shape[1] // DRAFT_TOPK1_PARTIAL_LANES
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=partials.device)
    topk_index = torch.empty((bs, 1), dtype=torch.int64, device=partials.device)
    if bs == 0:
        return topk_p, topk_index
    write_draft_token = draft_tokens is not None
    advance_positions = positions is not None
    use_pdl = is_arch_support_pdl()
    # Dummy operands for the disabled slots: valid pointers, never dereferenced.
    _draft_topk1_partials_finalize_kernel[(bs,)](
        partials,
        partials.stride(0),
        num_candidates,
        topk_p,
        topk_index,
        positions if advance_positions else topk_index,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        NAN_FIRST=nan_first,
        ADVANCE_POSITIONS=advance_positions,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_candidates),
        USE_PDL=use_pdl,
        num_warps=1,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return topk_p, topk_index


def draft_topk1_postprocess_from_partials(
    partials: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """``draft_topk1_postprocess`` on all-reduced ``draft_topk1_shard_partials``.

    Same token (NaN logits rank as -1e30, lowest index on ties), same writes:
    ``topk_index``, constant ``topk_p``, optional ``draft_tokens`` column and
    ``positions += 1``.
    """
    assert positions.ndim == 1 and positions.is_contiguous()
    assert positions.shape[0] == partials.shape[0]
    assert positions.device == partials.device
    if draft_tokens is not None:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == partials.device
        assert draft_tokens.shape[0] == partials.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]
    return _launch_partials_finalize(
        partials,
        nan_first=False,
        positions=positions,
        draft_tokens=draft_tokens,
        draft_token_column=draft_token_column,
    )


def draft_topk1_argmax_from_partials(partials: torch.Tensor):
    """``torch.argmax(logits, -1, keepdim=True)`` and ones from all-reduced partials.

    torch.argmax order: the lowest-index NaN first, else the lowest index of the
    maximum. Positions are not touched.
    """
    return _launch_partials_finalize(
        partials,
        nan_first=True,
        positions=None,
        draft_tokens=None,
        draft_token_column=0,
    )
