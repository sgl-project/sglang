"""Fused EAGLE draft proposal: q = softmax(logits / T), X ~ q, plus the greedy arm.

A flat two-stage split over the vocabulary: one partial pass carries the
online-softmax statistics and both candidate indices, a tiny combine pass picks
the per-row winner, and a last pass materializes ``probs``. Only ``probs`` is
written at vocabulary width.

Both arms rank on ``logits / T`` rather than on the softmax probabilities:
softmax is monotonic, and the Gumbel-max score ``probs / q`` is ranked by
``logits / T - log q`` because the row max and the normalizer are row
constants. Ties resolve to the lowest index, matching ``row_argmax`` and
``ArgMaxOps``.
"""

import torch
import triton
import triton.language as tl

# Finite floor for the running max so the online-softmax rescale never has to
# evaluate exp(-inf - -inf) on a fully masked tile.
_NEG_FLOOR = tl.constexpr(-1e30)


@triton.jit
def _draft_proposal_partial_kernel(
    LOGITS,
    TEMP,
    Q,
    PART_MAX,
    PART_SUM,
    PART_GREEDY_VAL,
    PART_GREEDY_IDX,
    PART_SAMPLE_VAL,
    PART_SAMPLE_IDX,
    PART_SAMPLE_X,
    N,
    stride_logits,
    stride_q,
    q_tiny,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    temp = tl.load(TEMP + row).to(tl.float32)
    per = tl.cdiv(N, SPLITS)
    start = part * per
    end = tl.minimum(start + per, N)

    run_max = _NEG_FLOOR
    run_sum = 0.0
    greedy_val = float("-inf")
    greedy_idx = N
    sample_val = float("-inf")
    sample_idx = N
    sample_x = float("-inf")

    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = tl.load(LOGITS + row * stride_logits + idx, mask, float("-inf")).to(
            tl.float32
        )
        x = x / temp

        # Online softmax over the split's tiles.
        tile_max = tl.max(x, 0)
        new_max = tl.maximum(run_max, tile_max)
        run_sum = run_sum * tl.exp(run_max - new_max) + tl.sum(
            tl.where(mask, tl.exp(x - new_max), 0.0), 0
        )
        run_max = new_max

        # Greedy arm: argmax(logits / T), lowest index on ties.
        tile_idx = tl.min(tl.where(x == tile_max, idx, N), 0)
        take = (tile_max > greedy_val) | (
            (tile_max == greedy_val) & (tile_idx < greedy_idx)
        )
        greedy_idx = tl.where(take, tile_idx, greedy_idx)
        greedy_val = tl.where(take, tile_max, greedy_val)

        # Sample arm: argmax(probs / q) == argmax(logits / T - log q). q is
        # clamped off zero here so the eager clamp_min_ pass can be dropped.
        q = tl.load(Q + row * stride_q + idx, mask, 1.0).to(tl.float32)
        gumbel = x - tl.log(tl.maximum(q, q_tiny))
        gumbel = tl.where(mask, gumbel, float("-inf"))
        tile_gumbel = tl.max(gumbel, 0)
        tile_gumbel_idx = tl.min(tl.where(gumbel == tile_gumbel, idx, N), 0)
        tile_gumbel_x = tl.sum(tl.where(idx == tile_gumbel_idx, x, 0.0), 0)
        take_sample = (tile_gumbel > sample_val) | (
            (tile_gumbel == sample_val) & (tile_gumbel_idx < sample_idx)
        )
        sample_idx = tl.where(take_sample, tile_gumbel_idx, sample_idx)
        sample_x = tl.where(take_sample, tile_gumbel_x, sample_x)
        sample_val = tl.where(take_sample, tile_gumbel, sample_val)

    slot = row * SPLITS + part
    tl.store(PART_MAX + slot, run_max)
    tl.store(PART_SUM + slot, run_sum)
    tl.store(PART_GREEDY_VAL + slot, greedy_val)
    tl.store(PART_GREEDY_IDX + slot, greedy_idx)
    tl.store(PART_SAMPLE_VAL + slot, sample_val)
    tl.store(PART_SAMPLE_IDX + slot, sample_idx)
    tl.store(PART_SAMPLE_X + slot, sample_x)


@triton.jit
def _draft_proposal_combine_kernel(
    PART_MAX,
    PART_SUM,
    PART_GREEDY_VAL,
    PART_GREEDY_IDX,
    PART_SAMPLE_VAL,
    PART_SAMPLE_IDX,
    PART_SAMPLE_X,
    TOP_KS,
    ROW_MAX,
    ROW_SUM,
    OUT_IDX,
    OUT_P,
    HAS_TOP_KS: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < SPLITS
    base = row * SPLITS + off

    part_max = tl.load(PART_MAX + base, mask, _NEG_FLOOR)
    part_sum = tl.load(PART_SUM + base, mask, 0.0)
    row_max = tl.max(part_max, 0)
    row_sum = tl.sum(tl.where(mask, part_sum * tl.exp(part_max - row_max), 0.0), 0)

    greedy_val = tl.load(PART_GREEDY_VAL + base, mask, float("-inf"))
    greedy_idx = tl.load(PART_GREEDY_IDX + base, mask, 0x7FFFFFFF)
    best_greedy = tl.max(greedy_val, 0)
    win_greedy = tl.min(tl.where(greedy_val == best_greedy, greedy_idx, 0x7FFFFFFF), 0)

    sample_val = tl.load(PART_SAMPLE_VAL + base, mask, float("-inf"))
    sample_idx = tl.load(PART_SAMPLE_IDX + base, mask, 0x7FFFFFFF)
    sample_x = tl.load(PART_SAMPLE_X + base, mask, float("-inf"))
    best_sample = tl.max(sample_val, 0)
    win_sample = tl.min(tl.where(sample_val == best_sample, sample_idx, 0x7FFFFFFF), 0)
    # Splits cover disjoint index ranges, so at most one lane matches.
    win_sample_x = tl.sum(tl.where(sample_idx == win_sample, sample_x, 0.0), 0)

    if HAS_TOP_KS:
        # A greedy row (top_k == 1) proposes its argmax; see sample_draft_proposal.
        is_greedy = tl.load(TOP_KS + row) <= 1
        win_idx = tl.where(is_greedy, win_greedy, win_sample)
        win_x = tl.where(is_greedy, best_greedy, win_sample_x)
    else:
        win_idx = win_sample
        win_x = win_sample_x

    tl.store(ROW_MAX + row, row_max)
    tl.store(ROW_SUM + row, row_sum)
    tl.store(OUT_IDX + row, win_idx.to(tl.int64))
    tl.store(
        OUT_P + row,
        (tl.exp(win_x - row_max) / row_sum).to(OUT_P.dtype.element_ty),
    )


@triton.jit
def _draft_proposal_probs_kernel(
    LOGITS,
    TEMP,
    ROW_MAX,
    ROW_SUM,
    OUT,
    N,
    stride_logits,
    stride_out,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    temp = tl.load(TEMP + row).to(tl.float32)
    row_max = tl.load(ROW_MAX + row)
    row_sum = tl.load(ROW_SUM + row)
    per = tl.cdiv(N, SPLITS)
    start = part * per
    end = tl.minimum(start + per, N)

    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = tl.load(LOGITS + row * stride_logits + idx, mask, float("-inf")).to(
            tl.float32
        )
        p = tl.exp(x / temp - row_max) / row_sum
        tl.store(OUT + row * stride_out + idx, p.to(OUT.dtype.element_ty), mask=mask)


_SPLITS = 64
_BLOCK = 2048


def can_use_fused_draft_proposal(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
) -> bool:
    rows = logits.shape[0] if logits.dim() == 2 else 0
    if rows == 0 or logits.stride(1) != 1:
        return False
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if (
        not isinstance(temperatures, torch.Tensor)
        or temperatures.numel() != rows
        or not temperatures.is_contiguous()
        or temperatures.device != logits.device
    ):
        return False
    if top_ks is not None and (
        top_ks.numel() != rows
        or not top_ks.is_contiguous()
        or top_ks.device != logits.device
    ):
        return False
    return True


def fused_draft_proposal(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
):
    """Return ``(probs, probs(X), X)`` for the EAGLE draft proposal.

    ``probs`` is ``softmax(logits / temperatures)``; ``X`` is a Gumbel-max draw
    from it, except on rows whose ``top_k <= 1``, which propose their argmax.
    Consumes the same per-element Exp(1) draw as ``sample_draft_proposal``.
    """
    rows, n = logits.shape
    device = logits.device
    q = torch.empty(logits.shape, dtype=torch.float32, device=device).exponential_(1.0)

    part_shape = (rows, _SPLITS)
    part_max = torch.empty(part_shape, dtype=torch.float32, device=device)
    part_sum = torch.empty(part_shape, dtype=torch.float32, device=device)
    part_greedy_val = torch.empty(part_shape, dtype=torch.float32, device=device)
    part_greedy_idx = torch.empty(part_shape, dtype=torch.int32, device=device)
    part_sample_val = torch.empty(part_shape, dtype=torch.float32, device=device)
    part_sample_idx = torch.empty(part_shape, dtype=torch.int32, device=device)
    part_sample_x = torch.empty(part_shape, dtype=torch.float32, device=device)

    _draft_proposal_partial_kernel[(rows, _SPLITS)](
        logits,
        temperatures,
        q,
        part_max,
        part_sum,
        part_greedy_val,
        part_greedy_idx,
        part_sample_val,
        part_sample_idx,
        part_sample_x,
        n,
        logits.stride(0),
        q.stride(0),
        torch.finfo(torch.float32).tiny,
        SPLITS=_SPLITS,
        BLOCK=_BLOCK,
        num_warps=8,
    )

    row_max = torch.empty((rows,), dtype=torch.float32, device=device)
    row_sum = torch.empty((rows,), dtype=torch.float32, device=device)
    out_idx = torch.empty((rows, 1), dtype=torch.int64, device=device)
    out_p = torch.empty((rows, 1), dtype=logits.dtype, device=device)

    _draft_proposal_combine_kernel[(rows,)](
        part_max,
        part_sum,
        part_greedy_val,
        part_greedy_idx,
        part_sample_val,
        part_sample_idx,
        part_sample_x,
        top_ks if top_ks is not None else part_max,
        row_max,
        row_sum,
        out_idx,
        out_p,
        HAS_TOP_KS=top_ks is not None,
        SPLITS=_SPLITS,
        BLOCK=triton.next_power_of_2(_SPLITS),
        num_warps=2,
    )

    probs = torch.empty_like(logits)
    _draft_proposal_probs_kernel[(rows, _SPLITS)](
        logits,
        temperatures,
        row_max,
        row_sum,
        probs,
        n,
        logits.stride(0),
        probs.stride(0),
        SPLITS=_SPLITS,
        BLOCK=_BLOCK,
        num_warps=8,
    )
    return probs, out_p, out_idx
