from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda

# A row carrying all its mass on one token sends the log-odds to infinity.
MAX_LOG_ODDS = 40.0


class DraftTopLogOdds:
    """logit(q_max) of each draft row's temperature-scaled distribution."""

    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return draft_top_log_odds(*args, **kwargs)
        return draft_top_log_odds_torch(*args, **kwargs)


class PredictAcceptance:
    """[rows] features logit(q_max) and acceptance predictions
    sigmoid(slope * feature + intercepts[row % rows_per_request])."""

    @classmethod
    def execute(cls, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(*args, **kwargs):
            return predict_acceptance(*args, **kwargs)
        return predict_acceptance_torch(*args, **kwargs)


class AccumulateBinCounts:
    """Adds one graded block into [gamma, num_bins, (graded, accepted)] counts."""

    @classmethod
    def execute(cls, *args, **kwargs) -> None:
        if inputs_on_cuda(*args, **kwargs):
            return accumulate_bin_counts(*args, **kwargs)
        return accumulate_bin_counts_torch(*args, **kwargs)


def predict_acceptance_torch(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = draft_top_log_odds_torch(
        logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
    )
    position_intercepts = intercepts.repeat(logits.shape[0] // rows_per_request)
    return features, torch.sigmoid(slope * features + position_intercepts)


def accumulate_bin_counts_torch(
    *,
    counts: torch.Tensor,
    features: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    num_graded_drafts: torch.Tensor,
    bin_low: float,
    bin_width: float,
) -> None:
    """``features``: [bs, gamma]. Draft k of a request is graded when
    k <= num_correct and k < num_graded, and accepted when k < num_correct."""
    gamma, num_bins, _ = counts.shape
    positions = torch.arange(gamma, device=features.device).view(1, -1)
    num_correct = num_correct_drafts.view(-1, 1).to(positions.dtype)
    num_graded = num_graded_drafts.view(-1, 1).to(positions.dtype)
    graded = ((positions <= num_correct) & (positions < num_graded)).to(counts.dtype)
    accepted = graded * (positions < num_correct).to(counts.dtype)
    bins = torch.clamp(((features - bin_low) / bin_width).floor(), 0, num_bins - 1)
    flat = (positions * num_bins + bins.long()).view(-1)
    flat_counts = counts.view(-1, 2)
    flat_counts[:, 0].index_add_(0, flat, graded.view(-1))
    flat_counts[:, 1].index_add_(0, flat, accepted.view(-1))


def draft_top_log_odds_torch(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    """``logits``: [bs * rows_per_request, vocab]; ``temperatures``: [bs]."""
    num_rows = logits.shape[0]
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(-1).to(torch.float32), rows_per_request, dim=0
    )
    assert temp_per_row.numel() == num_rows
    scaled = logits.to(torch.float32) / temp_per_row[:, None]
    row_max = scaled.max(dim=-1, keepdim=True).values
    sum_exp = torch.exp(scaled - row_max).sum(dim=-1)
    # The top token contributes exactly e^0 = 1, so the rest sum to sum_exp - 1
    # and logit(q_max) = log(1 / (sum_exp - 1)).
    complement = torch.clamp(sum_exp - 1.0, min=0.0)
    return torch.clamp(-torch.log(complement), min=-MAX_LOG_ODDS, max=MAX_LOG_ODDS)


@triton.jit
def _draft_top_log_odds_kernel(
    logits_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    rows_per_request,
    logits_row_stride,
    slope_ptr,
    intercepts_ptr,
    pred_ptr,
    MAX_ODDS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    PREDICT: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_row_stride

    row_max = -float("inf")
    sum_exp = 0.0
    for v0 in range(0, vocab, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < vocab
        x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
        x = x / temp
        new_max = tl.maximum(row_max, tl.max(x, axis=0))
        # Masked (-inf) logits may fill whole blocks; keep exp() off -inf - -inf.
        shift = tl.where(new_max == -float("inf"), 0.0, new_max)
        e = tl.where(vmask, tl.exp(x - shift), 0.0)
        sum_exp = sum_exp * tl.exp(row_max - shift) + tl.sum(e, axis=0)
        row_max = new_max

    complement = tl.maximum(sum_exp - 1.0, 0.0)
    feature = -tl.log(complement)
    feature = tl.minimum(tl.maximum(feature, -MAX_ODDS), MAX_ODDS)
    tl.store(out_ptr + row, feature)
    if PREDICT:
        slope = tl.load(slope_ptr)
        intercept = tl.load(intercepts_ptr + row % rows_per_request)
        tl.store(pred_ptr + row, tl.sigmoid(slope * feature + intercept))


def _launch_top_log_odds(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
    slope: torch.Tensor | None,
    intercepts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_rows, vocab = logits.shape
    bs = num_rows // rows_per_request
    assert bs * rows_per_request == num_rows, (
        f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    )
    assert logits.stride(-1) == 1
    temperatures = temperatures.reshape(bs).to(torch.float32).contiguous()
    features = torch.empty((num_rows,), dtype=torch.float32, device=logits.device)
    predict = slope is not None
    predictions = torch.empty_like(features) if predict else None
    if num_rows == 0:
        return features, predictions
    _draft_top_log_odds_kernel[(num_rows,)](
        logits,
        temperatures,
        features,
        vocab,
        rows_per_request,
        logits.stride(0),
        slope if predict else features,
        intercepts if predict else features,
        predictions if predict else features,
        MAX_ODDS=MAX_LOG_ODDS,
        BLOCK_V=4096,
        PREDICT=predict,
    )
    return features, predictions


def draft_top_log_odds(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    features, _ = _launch_top_log_odds(
        logits=logits,
        temperatures=temperatures,
        rows_per_request=rows_per_request,
        slope=None,
        intercepts=None,
    )
    return features


def predict_acceptance(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert intercepts.numel() == rows_per_request
    return _launch_top_log_odds(
        logits=logits,
        temperatures=temperatures,
        rows_per_request=rows_per_request,
        slope=slope.to(torch.float32).contiguous(),
        intercepts=intercepts.to(torch.float32).contiguous(),
    )


@triton.jit
def _accumulate_bin_counts_kernel(
    counts_ptr,
    features_ptr,
    num_correct_ptr,
    num_graded_ptr,
    gamma,
    num_bins,
    bin_low,
    bin_width,
    BLOCK_K: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    in_block = k < gamma
    num_correct = tl.load(num_correct_ptr + req)
    num_graded = tl.load(num_graded_ptr + req)
    graded = in_block & (k <= num_correct) & (k < num_graded)
    accepted = graded & (k < num_correct)
    feature = tl.load(features_ptr + req * gamma + k, mask=in_block, other=0.0)
    bins = tl.floor((feature - bin_low) / bin_width)
    bins = tl.minimum(tl.maximum(bins, 0.0), num_bins - 1.0).to(tl.int64)
    slot = counts_ptr + (k * num_bins + bins) * 2
    # Counts are small integers held exactly in fp32, so add order does not matter.
    tl.atomic_add(slot, tl.where(graded, 1.0, 0.0), mask=graded)
    tl.atomic_add(slot + 1, tl.where(accepted, 1.0, 0.0), mask=accepted)


def accumulate_bin_counts(
    *,
    counts: torch.Tensor,
    features: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    num_graded_drafts: torch.Tensor,
    bin_low: float,
    bin_width: float,
) -> None:
    gamma, num_bins, _ = counts.shape
    bs = features.shape[0]
    assert counts.is_contiguous() and features.is_contiguous()
    if bs == 0:
        return
    _accumulate_bin_counts_kernel[(bs,)](
        counts,
        features,
        num_correct_drafts.to(torch.int32).contiguous(),
        num_graded_drafts.to(torch.int32).contiguous(),
        gamma,
        num_bins,
        bin_low,
        bin_width,
        BLOCK_K=triton.next_power_of_2(gamma),
    )
