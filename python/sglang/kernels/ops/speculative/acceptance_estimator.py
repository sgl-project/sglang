"""Kernels of the online draft acceptance estimator.

``PredictAcceptance`` scores each draft row with logit(q_max), the log-odds of
the most likely token of its temperature-scaled draft distribution, and the
logistic prediction sigmoid(slope * feature + intercepts[position]).
``AccumulateAcceptanceCounts`` bins verified drafts by (position, feature) and
``RefitAcceptance`` folds a round of bins into a decayed history and solves the
penalized maximum likelihood on it. All three read their state from device
tensors, so none synchronizes with the host.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda

MAX_LOG_ODDS = 40.0
BIN_LOW, BIN_HIGH, NUM_BINS = -8.0, 24.0, 128
BIN_WIDTH = (BIN_HIGH - BIN_LOW) / NUM_BINS
STEP_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0)
assert STEP_SCALES == tuple(2.0**-i for i in range(len(STEP_SCALES) - 1)) + (0.0,)


class PredictAcceptance:
    """Writes the [bs, num_positions] features and acceptance predictions of
    [bs, num_positions, vocab] draft logits. Rows with a non-positive
    temperature are greedy and scored at temperature 1."""

    @classmethod
    def execute(cls, **kwargs) -> None:
        if inputs_on_cuda(**kwargs):
            return predict_acceptance(**kwargs)
        return predict_acceptance_torch(**kwargs)


class AccumulateAcceptanceCounts:
    """Adds one verified block to the [num_positions, NUM_BINS, (graded,
    correct)] counts. Draft k of a request is graded when k <= num_correct
    and k < num_graded, and correct when k < num_correct."""

    @classmethod
    def execute(cls, **kwargs) -> None:
        if inputs_on_cuda(**kwargs):
            return accumulate_acceptance_counts(**kwargs)
        return accumulate_acceptance_counts_torch(**kwargs)


class RefitAcceptance:
    """Decays ``history_counts`` and adds ``round_counts`` (then clears them),
    and refits ``slope`` and ``intercepts`` on the history in place."""

    @classmethod
    def execute(cls, **kwargs) -> None:
        if inputs_on_cuda(**kwargs):
            return refit_acceptance(**kwargs)
        return refit_acceptance_torch(**kwargs)


def bin_centers(device) -> torch.Tensor:
    return BIN_LOW + (torch.arange(NUM_BINS, device=device) + 0.5) * BIN_WIDTH


def predict_acceptance_torch(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    features_out: torch.Tensor,
    predictions_out: torch.Tensor,
) -> None:
    temperatures = temperatures.reshape(-1, 1, 1).to(torch.float32)
    scaled = logits.to(torch.float32) / torch.where(temperatures > 0, temperatures, 1.0)
    top = scaled.argmax(dim=-1, keepdim=True)
    rest = torch.exp(scaled - scaled.gather(-1, top)).scatter(-1, top, 0.0).sum(-1)
    features = torch.where(rest > 0, -torch.log(rest), MAX_LOG_ODDS)
    features = features.clamp(-MAX_LOG_ODDS, MAX_LOG_ODDS)
    features_out.copy_(features)
    predictions_out.copy_(torch.sigmoid(slope * features + intercepts))


@triton.jit
def _predict_partial_kernel(
    logits_ptr,
    stride_req,
    stride_pos,
    temperatures_ptr,
    partial_ptr,
    num_positions,
    vocab,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    num_splits = tl.num_programs(1)
    req = row // num_positions
    pos = row % num_positions
    temperature = tl.load(temperatures_ptr + req).to(tl.float32)
    temperature = tl.where(temperature > 0.0, temperature, 1.0)
    base = logits_ptr + req.to(tl.int64) * stride_req + pos.to(tl.int64) * stride_pos
    offs = split * BLOCK_V + tl.arange(0, BLOCK_V)
    z = tl.load(base + offs, mask=offs < vocab, other=float("-inf")).to(tl.float32)
    z = z / temperature
    blk_max = tl.max(z, axis=0)
    blk_arg = tl.min(tl.where(z == blk_max, offs, vocab), axis=0)
    e = tl.where((z > float("-inf")) & (offs != blk_arg), tl.exp(z - blk_max), 0.0)
    out = partial_ptr + (row.to(tl.int64) * num_splits + split) * 2
    tl.store(out, blk_max)
    tl.store(out + 1, tl.sum(e, axis=0))


@triton.jit
def _predict_finish_kernel(
    partial_ptr,
    slope_ptr,
    intercepts_ptr,
    features_ptr,
    features_stride,
    predictions_ptr,
    predictions_stride,
    num_positions,
    num_splits,
    MAX_ODDS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    req = row // num_positions
    pos = row % num_positions
    i = tl.arange(0, BLOCK_S)
    mask = i < num_splits
    base = partial_ptr + row.to(tl.int64) * num_splits * 2
    split_max = tl.load(base + i * 2, mask=mask, other=float("-inf"))
    split_rest = tl.load(base + i * 2 + 1, mask=mask, other=0.0)
    row_max = tl.max(split_max, axis=0)
    top = tl.min(tl.where(split_max == row_max, i, BLOCK_S), axis=0)
    scale = tl.where(split_max > float("-inf"), tl.exp(split_max - row_max), 0.0)
    rest = tl.sum(split_rest * scale + tl.where(i != top, scale, 0.0), axis=0)
    feature = tl.where(rest > 0.0, -tl.log(rest), MAX_ODDS)
    feature = tl.minimum(tl.maximum(feature, -MAX_ODDS), MAX_ODDS)
    prediction = tl.sigmoid(
        tl.load(slope_ptr) * feature + tl.load(intercepts_ptr + pos)
    )
    tl.store(features_ptr + req * features_stride + pos, feature)
    tl.store(predictions_ptr + req * predictions_stride + pos, prediction)


def predict_acceptance(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    features_out: torch.Tensor,
    predictions_out: torch.Tensor,
) -> None:
    bs, num_positions, vocab = logits.shape
    assert logits.stride(2) == 1
    assert intercepts.shape == (num_positions,) and slope.numel() == 1
    assert temperatures.shape == (bs,) and temperatures.stride(0) == 1
    for out in (features_out, predictions_out):
        assert out.shape == (bs, num_positions) and out.stride(1) == 1
        assert out.dtype == torch.float32
    if bs == 0:
        return
    num_rows = bs * num_positions
    block_v = min(triton.next_power_of_2(vocab), 4096)
    num_splits = triton.cdiv(vocab, block_v)
    partial = torch.empty(
        (num_rows, num_splits, 2), dtype=torch.float32, device=logits.device
    )
    _predict_partial_kernel[(num_rows, num_splits)](
        logits,
        logits.stride(0),
        logits.stride(1),
        temperatures,
        partial,
        num_positions,
        vocab,
        BLOCK_V=block_v,
    )
    _predict_finish_kernel[(num_rows,)](
        partial,
        slope,
        intercepts,
        features_out,
        features_out.stride(0),
        predictions_out,
        predictions_out.stride(0),
        num_positions,
        num_splits,
        MAX_ODDS=MAX_LOG_ODDS,
        BLOCK_S=triton.next_power_of_2(num_splits),
    )


def _graded_and_correct(
    *, num_positions: int, num_correct_drafts, num_graded_drafts, device
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(num_positions, device=device).view(1, -1)
    num_correct = num_correct_drafts.view(-1, 1).to(positions.dtype)
    graded = (positions <= num_correct) & (
        positions < num_graded_drafts.view(-1, 1).to(positions.dtype)
    )
    return graded, graded & (positions < num_correct)


def accumulate_acceptance_counts_torch(
    *,
    counts: torch.Tensor,
    features: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    num_graded_drafts: torch.Tensor,
) -> None:
    num_positions = counts.shape[0]
    graded, correct = _graded_and_correct(
        num_positions=num_positions,
        num_correct_drafts=num_correct_drafts,
        num_graded_drafts=num_graded_drafts,
        device=features.device,
    )
    bins = ((features - BIN_LOW) / BIN_WIDTH).floor().clamp(0, NUM_BINS - 1).long()
    positions = torch.arange(num_positions, device=features.device).view(1, -1)
    slots = (positions * NUM_BINS + bins).view(-1)
    flat = counts.view(-1, 2)
    flat[:, 0].index_add_(0, slots, graded.view(-1).to(counts.dtype))
    flat[:, 1].index_add_(0, slots, correct.view(-1).to(counts.dtype))


@triton.jit
def _accumulate_acceptance_counts_kernel(
    counts_ptr,
    features_ptr,
    features_stride,
    num_correct_ptr,
    num_graded_ptr,
    num_positions,
    BIN_LOW_: tl.constexpr,
    BIN_WIDTH_: tl.constexpr,
    NUM_BINS_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    num_correct = tl.load(num_correct_ptr + req)
    graded = (
        (k < num_positions) & (k <= num_correct) & (k < tl.load(num_graded_ptr + req))
    )
    correct = graded & (k < num_correct)
    feature = tl.load(features_ptr + req * features_stride + k, mask=graded, other=0.0)
    bins = tl.floor((feature - BIN_LOW_) / BIN_WIDTH_)
    bins = tl.minimum(tl.maximum(bins, 0.0), NUM_BINS_ - 1.0).to(tl.int64)
    slot = counts_ptr + (k * NUM_BINS_ + bins) * 2
    tl.atomic_add(slot, 1.0, mask=graded)
    tl.atomic_add(slot + 1, 1.0, mask=correct)


def accumulate_acceptance_counts(
    *,
    counts: torch.Tensor,
    features: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    num_graded_drafts: torch.Tensor,
) -> None:
    num_positions = counts.shape[0]
    bs = features.shape[0]
    assert counts.shape == (num_positions, NUM_BINS, 2) and counts.is_contiguous()
    assert features.shape == (bs, num_positions) and features.stride(1) == 1
    if bs == 0:
        return
    _accumulate_acceptance_counts_kernel[(bs,)](
        counts,
        features,
        features.stride(0),
        num_correct_drafts.to(torch.int32).contiguous(),
        num_graded_drafts.to(torch.int32).contiguous(),
        num_positions,
        BIN_LOW_=BIN_LOW,
        BIN_WIDTH_=BIN_WIDTH,
        NUM_BINS_=NUM_BINS,
        BLOCK_K=triton.next_power_of_2(num_positions),
    )


def refit_acceptance_torch(
    *,
    round_counts: torch.Tensor,
    history_counts: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    decay: float,
    prior_weight: float,
    num_iters: int,
) -> None:
    """Penalized maximum likelihood on the binned labels. Each Newton
    iteration solves the arrowhead system (the slope couples to every
    position, each intercept only to itself) in closed form, then backtracks
    along the step. The Gaussian prior pulls toward the previous coefficients,
    so positions without labels stay where they were."""
    history_counts.mul_(decay).add_(round_counts)
    round_counts.zero_()
    graded, correct = history_counts[..., 0], history_counts[..., 1]
    f = bin_centers(history_counts.device).view(1, -1)
    prior_slope, prior_intercepts = slope.view(()).clone(), intercepts.clone()
    s, b = prior_slope.clone(), prior_intercepts.clone()
    scales = torch.tensor(STEP_SCALES, device=f.device)
    for _ in range(num_iters):
        p = torch.sigmoid(s * f + b.view(-1, 1))
        weight = graded * p * (1.0 - p)
        residual = correct - graded * p
        b_k = (weight * f).sum(dim=1)
        c_k = weight.sum(dim=1) + prior_weight
        h_k = residual.sum(dim=1) - prior_weight * (b - prior_intercepts)
        a = (weight * f * f).sum() + prior_weight
        g = (residual * f).sum() - prior_weight * (s - prior_slope)
        d_s = (g - (b_k * h_k / c_k).sum()) / (a - (b_k * b_k / c_k).sum())
        d_s = torch.where(torch.isfinite(d_s), d_s, 0.0)
        d_b = (h_k - b_k * d_s) / c_k
        d_b = torch.where(torch.isfinite(d_b), d_b, 0.0)
        cand_s = torch.clamp(s + scales * d_s, min=0.0)
        cand_b = b.view(1, -1) + scales.view(-1, 1) * d_b
        z = cand_s.view(-1, 1, 1) * f.view(1, 1, -1) + cand_b.unsqueeze(-1)
        nll = (graded * F.softplus(z) - correct * z).sum(dim=(1, 2))
        nll = nll + 0.5 * prior_weight * (
            (cand_s - prior_slope) ** 2 + ((cand_b - prior_intercepts) ** 2).sum(dim=1)
        )
        nll = torch.where(torch.isfinite(nll), nll, float("inf"))
        best = torch.argmin(nll)
        best = torch.where(torch.isfinite(nll[best]), best, len(STEP_SCALES) - 1)
        s, b = cand_s[best], cand_b[best]
    slope.copy_(s.view_as(slope))
    intercepts.copy_(b)


@triton.jit
def _softplus(z):
    return tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(z)))


@triton.jit
def _refit_acceptance_kernel(
    round_ptr,
    history_ptr,
    slope_ptr,
    intercepts_ptr,
    num_positions,
    decay,
    prior_weight,
    num_iters,
    BIN_LOW_: tl.constexpr,
    BIN_WIDTH_: tl.constexpr,
    NUM_BINS_: tl.constexpr,
    NUM_SCALES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    j = tl.arange(0, NUM_BINS_)
    k_mask = k < num_positions
    mask = k_mask[:, None] & (j[None, :] < NUM_BINS_)
    offs = (k[:, None] * NUM_BINS_ + j[None, :]) * 2
    graded = tl.load(history_ptr + offs, mask=mask, other=0.0) * decay + tl.load(
        round_ptr + offs, mask=mask, other=0.0
    )
    correct = tl.load(history_ptr + offs + 1, mask=mask, other=0.0) * decay + tl.load(
        round_ptr + offs + 1, mask=mask, other=0.0
    )
    tl.store(history_ptr + offs, graded, mask=mask)
    tl.store(history_ptr + offs + 1, correct, mask=mask)
    tl.store(round_ptr + offs, tl.zeros_like(graded), mask=mask)
    tl.store(round_ptr + offs + 1, tl.zeros_like(graded), mask=mask)

    f = (BIN_LOW_ + (j.to(tl.float32) + 0.5) * BIN_WIDTH_)[None, :]
    prior_s = tl.load(slope_ptr)
    prior_b = tl.load(intercepts_ptr + k, mask=k_mask, other=0.0)
    s = prior_s
    b = prior_b
    for _ in range(num_iters):
        p = tl.sigmoid(s * f + b[:, None])
        weight = graded * p * (1.0 - p)
        residual = correct - graded * p
        b_k = tl.sum(weight * f, axis=1)
        c_k = tl.where(k_mask, tl.sum(weight, axis=1) + prior_weight, 1.0)
        h_k = tl.where(
            k_mask, tl.sum(residual, axis=1) - prior_weight * (b - prior_b), 0.0
        )
        a = tl.sum(tl.sum(weight * f * f, axis=1), axis=0) + prior_weight
        g = tl.sum(tl.sum(residual * f, axis=1), axis=0) - prior_weight * (s - prior_s)
        d_s = (g - tl.sum(b_k * h_k / c_k, axis=0)) / (
            a - tl.sum(b_k * b_k / c_k, axis=0)
        )
        d_s = tl.where(tl.abs(d_s) < float("inf"), d_s, 0.0)
        d_b = (h_k - b_k * d_s) / c_k
        d_b = tl.where(tl.abs(d_b) < float("inf"), d_b, 0.0)
        best_nll = float("inf")
        best_scale = 0.0
        for i in range(NUM_SCALES):
            step = tl.where(i < NUM_SCALES - 1, tl.exp2(-1.0 * i), 0.0)
            cand_s = tl.maximum(s + step * d_s, 0.0)
            cand_b = b + step * d_b
            z = cand_s * f + cand_b[:, None]
            nll = tl.sum(tl.sum(graded * _softplus(z) - correct * z, axis=1), axis=0)
            diff_b = tl.where(k_mask, cand_b - prior_b, 0.0)
            nll += (
                0.5
                * prior_weight
                * (
                    (cand_s - prior_s) * (cand_s - prior_s)
                    + tl.sum(diff_b * diff_b, axis=0)
                )
            )
            better = nll < best_nll
            best_scale = tl.where(better, step, best_scale)
            best_nll = tl.where(better, nll, best_nll)
        s = tl.maximum(s + best_scale * d_s, 0.0)
        b = b + best_scale * d_b
    tl.store(slope_ptr, s)
    tl.store(intercepts_ptr + k, b, mask=k_mask)


def refit_acceptance(
    *,
    round_counts: torch.Tensor,
    history_counts: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    decay: float,
    prior_weight: float,
    num_iters: int,
) -> None:
    num_positions = intercepts.shape[0]
    for counts in (round_counts, history_counts):
        assert counts.shape == (num_positions, NUM_BINS, 2) and counts.is_contiguous()
    assert slope.numel() == 1 and slope.dtype == intercepts.dtype == torch.float32
    _refit_acceptance_kernel[(1,)](
        round_counts,
        history_counts,
        slope,
        intercepts,
        num_positions,
        decay,
        prior_weight,
        num_iters,
        BIN_LOW_=BIN_LOW,
        BIN_WIDTH_=BIN_WIDTH,
        NUM_BINS_=NUM_BINS,
        NUM_SCALES=len(STEP_SCALES),
        BLOCK_K=triton.next_power_of_2(num_positions),
        num_warps=4,
    )
