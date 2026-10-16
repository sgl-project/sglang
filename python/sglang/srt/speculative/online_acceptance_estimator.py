"""Online-fitted acceptance estimator for confidence-scheduled verification.

Predicts each draft position's acceptance probability from the draft
distribution alone, ``p = sigmoid(slope * logit(q_max) + intercept[k])``, and
refits the shared slope and per-position intercepts on the target's
accept/reject outcomes while serving. The feature reads only ``q_k``, which
depends on the drafts before position ``k`` and never on the drafted token
itself, so trimming verification on it stays lossless. Scoring the drawn
token's probability instead would select on the proposal and bias the output
toward the drafter's confident modes.

Labels are binned by (position, feature) and kept with a decayed history; each
refit runs Newton iterations with a line search to the penalized maximum
likelihood. A single Newton step per round from the current coefficients is
cheaper but unstable: a confidently wrong prediction carries a full residual and
almost no curvature, so one such label can fling an intercept, and when
acceptance saturates below one for confident drafts (a sampled target keeps its
own entropy) the slope update stalls and survival is over-predicted.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.acceptance_estimator import (
    AccumulateBinCounts,
    PredictAcceptance,
)

logger = logging.getLogger(__name__)

# Cold-start coefficients: the median fit vLLM reports across five model pairs
# (vllm-project/vllm#52228); the first refit replaces them.
_INIT_SLOPE = 0.5
_INIT_INTERCEPT = -0.2

# logit(q_max) bins: [-8, 24) at width 0.25, outliers in the edge bins.
# Arbitrary; covers q_max in [3e-4, 1 - 4e-11].
_BIN_LOW, _BIN_HIGH, _NUM_BINS = -8.0, 24.0, 128
_BIN_WIDTH = (_BIN_HIGH - _BIN_LOW) / _NUM_BINS
# Backtracking ladder of each Newton iteration, evaluated in one batch.
_STEP_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0)


class OnlineAcceptanceEstimator:
    """Per step, ``predict`` scores the drafted block before verification and
    ``observe`` grades the same block once the target has verified it. Every
    TP rank sees identical inputs and fits identically, on device."""

    # Refit every this many observed steps.
    REFIT_INTERVAL = 100
    # Each refit keeps this fraction of the older counts (about four rounds).
    HISTORY_DECAY = 0.75
    # Gaussian prior on the change from the previous coefficients, in units of
    # Fisher information (one label at p = 1/2 carries 1/4). Keeps positions
    # with no labels where they were.
    PRIOR_WEIGHT = 1.0
    NEWTON_ITERS = 8

    def __init__(self, *, gamma: int, device: torch.device, log_rank: bool) -> None:
        self.gamma = int(gamma)
        self.slope = torch.full((1,), _INIT_SLOPE, dtype=torch.float32, device=device)
        self.intercepts = torch.full(
            (self.gamma,), _INIT_INTERCEPT, dtype=torch.float32, device=device
        )
        self.num_refits = 0
        self._log_rank = log_rank
        self._bin_centers = (
            _BIN_LOW + (torch.arange(_NUM_BINS, device=device) + 0.5) * _BIN_WIDTH
        )
        # [gamma, bins, (graded, accepted)] label counts of this round, and the
        # decayed history the fit solves on.
        self._round_counts = torch.zeros(
            (self.gamma, _NUM_BINS, 2), dtype=torch.float32, device=device
        )
        self._history_counts = torch.zeros_like(self._round_counts)
        self._steps_since_refit = 0
        # The last predict's [bs, gamma] features, graded by observe.
        self._features: torch.Tensor | None = None

    def predict(
        self,
        *,
        step_logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """[bs, gamma] acceptance probabilities from the [bs, gamma, vocab]
        pre-temperature draft logits."""
        bs, gamma, vocab = step_logits.shape
        assert gamma == self.gamma
        # Greedy rows draft by argmax; score their T = 1 distribution.
        feature_temperatures = torch.where(
            greedy_mask.view(-1), torch.ones_like(temperatures), temperatures
        )
        features, predictions = PredictAcceptance.execute(
            logits=step_logits.reshape(bs * gamma, vocab),
            temperatures=feature_temperatures,
            rows_per_request=gamma,
            slope=self.slope,
            intercepts=self.intercepts,
        )
        self._features = features.view(bs, gamma)
        return predictions.view(bs, gamma)

    def observe(
        self,
        *,
        num_correct_drafts: torch.Tensor,
        num_graded_drafts: torch.Tensor,
    ) -> None:
        """Grade the last predicted block. ``num_correct_drafts``: [bs] drafts the
        target accepted before its first rejection, uncapped by the verify cut;
        ``num_graded_drafts``: [bs] leading drafts checked against a real target
        row. Drafts up to and including the first rejection carry labels."""
        features = self._features
        assert features is not None, "observe() must follow predict()"
        self._features = None
        AccumulateBinCounts.execute(
            counts=self._round_counts,
            features=features,
            num_correct_drafts=num_correct_drafts,
            num_graded_drafts=num_graded_drafts,
            bin_low=_BIN_LOW,
            bin_width=_BIN_WIDTH,
        )
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.REFIT_INTERVAL:
            self._refit()

    def skip_observe(self) -> None:
        """Drop the last prediction ungraded (e.g. its outcome was simulated)."""
        self._features = None

    def _refit(self) -> None:
        if self._log_rank and _should_log_refit(self.num_refits + 1):
            self._log_round()
        self._history_counts.mul_(self.HISTORY_DECAY).add_(self._round_counts)
        self._round_counts.zero_()
        slope, intercepts = _fit_binned(
            counts=self._history_counts,
            centers=self._bin_centers,
            slope=self.slope,
            intercepts=self.intercepts,
            prior_weight=self.PRIOR_WEIGHT,
            num_iters=self.NEWTON_ITERS,
        )
        self.slope.copy_(slope)
        self.intercepts.copy_(intercepts)
        self._steps_since_refit = 0
        self.num_refits += 1

    def _log_round(self) -> None:
        graded, accepted = self._round_counts.unbind(dim=-1)
        predictions = torch.sigmoid(
            self.slope * self._bin_centers.view(1, -1) + self.intercepts.view(-1, 1)
        )
        num_graded = graded.sum(dim=1)
        denom = num_graded.clamp(min=1.0)
        logger.info(
            "Online acceptance estimator refit %d: slope=%.4f intercepts=%s "
            "graded=%s observed_accept=%s mean_pred=%s",
            self.num_refits + 1,
            float(self.slope.item()),
            _fmt(self.intercepts),
            num_graded.to(torch.int64).tolist(),
            _fmt(accepted.sum(dim=1) / denom),
            _fmt((graded * predictions).sum(dim=1) / denom),
        )


def _fit_binned(
    *,
    counts: torch.Tensor,
    centers: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    prior_weight: float,
    num_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalized maximum likelihood on binned labels. Each Newton iteration
    solves the arrowhead system (the slope row couples to every position, each
    intercept only to itself) in closed form, then backtracks along the step.
    At the optimum each position's mean prediction matches its observed
    acceptance up to the prior, which is what the verify budget sums."""
    graded, accepted = counts[..., 0], counts[..., 1]
    f = centers.view(1, -1)
    prior_slope, prior_intercepts = slope.clone(), intercepts.clone()
    scales = torch.tensor(_STEP_SCALES, device=centers.device)
    for _ in range(num_iters):
        p = torch.sigmoid(slope * f + intercepts.view(-1, 1))
        weight = graded * p * (1.0 - p)
        residual = accepted - graded * p
        b_k = (weight * f).sum(dim=1)
        c_k = weight.sum(dim=1) + prior_weight
        h_k = residual.sum(dim=1) - prior_weight * (intercepts - prior_intercepts)
        a = (weight * f * f).sum() + prior_weight
        g = (residual * f).sum() - prior_weight * (slope - prior_slope)
        d_slope = (g - (b_k * h_k / c_k).sum()) / (a - (b_k * b_k / c_k).sum())
        d_intercepts = (h_k - b_k * d_slope) / c_k
        cand_slope = torch.clamp(slope + scales * d_slope, min=0.0)
        cand_intercepts = intercepts.view(1, -1) + scales.view(-1, 1) * d_intercepts
        z = cand_slope.view(-1, 1, 1) * f.view(1, 1, -1) + cand_intercepts.unsqueeze(-1)
        nll = (graded * F.softplus(z) - accepted * z).sum(dim=(1, 2))
        nll = nll + 0.5 * prior_weight * (
            (cand_slope - prior_slope) ** 2
            + ((cand_intercepts - prior_intercepts) ** 2).sum(dim=1)
        )
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, float("inf")))
        best = torch.argmin(nll)
        slope = cand_slope[best].view(1)
        intercepts = cand_intercepts[best]
    return slope, intercepts


def _should_log_refit(refit: int) -> bool:
    return refit in (1, 10) or refit % 50 == 0


def _fmt(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values.cpu()) + "]"
