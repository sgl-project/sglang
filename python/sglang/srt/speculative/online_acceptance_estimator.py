"""Online-fitted acceptance estimator for confidence-scheduled verification.

Predicts each draft position's acceptance probability from the draft
distribution alone, ``p = sigmoid(slope * logit(q_max) + intercept[k])``, and
refits the shared slope and per-position intercepts on the target's
accept/reject outcomes while serving. The feature reads
only ``q_k``, which depends on the drafts before position ``k`` and never on the
drafted token itself, so trimming verification on it stays lossless. Scoring the
drawn token's probability instead would select on the proposal and bias the
output toward the drafter's confident modes.
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.ops.speculative.acceptance_estimator import DraftTopLogOdds

logger = logging.getLogger(__name__)

# Cold-start coefficients: the median fit vLLM reports across five model pairs
# (vllm-project/vllm#52228). Log-loss is flat near the optimum, so a real fit
# replaces them within a refit or two.
_INIT_SLOPE = 0.5
_INIT_INTERCEPT = -0.2

# Per-position columns of the round statistics.
_A, _B, _C, _G_SLOPE, _G_INTERCEPT, _NUM_GRADED, _NUM_ACCEPT, _PRED_SUM = range(8)

# Feature histogram of the binned fit: logit(q_max) in [-8, 24) at width 0.25;
# values outside land in the edge bins. Arbitrary; covers q_max in [3e-4, 1 - 4e-11].
_BIN_LOW, _BIN_HIGH, _NUM_BINS = -8.0, 24.0, 128
_BIN_WIDTH = (_BIN_HIGH - _BIN_LOW) / _NUM_BINS
# Backtracking ladder of the binned Newton iterations, evaluated in one batch.
_STEP_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0)


class OnlineAcceptanceEstimator:
    """Per step, ``predict`` scores the drafted block before verification and
    ``observe`` grades the same block once the target has verified it. Every
    TP rank sees identical inputs and fits identically."""

    # Refit every this many observed steps; statistics accumulate in between.
    REFIT_INTERVAL = 100
    # Each Newton step is damped by n / (n + DAMPING_OBSERVATIONS), so thinly
    # labeled deep positions move slowly instead of freezing behind a hard gate.
    DAMPING_OBSERVATIONS = 50.0
    # Ridge on the Newton solve.
    L2 = 1e-3
    # Binned fit: each refit keeps this fraction of the older counts.
    HISTORY_DECAY = 0.75
    # Binned fit: Gaussian prior on the change from the previous coefficients,
    # in units of Fisher information (one label at p = 1/2 carries 1/4).
    PRIOR_WEIGHT = 1.0
    NEWTON_ITERS = 8

    def __init__(
        self,
        *,
        gamma: int,
        device: torch.device,
        log_rank: bool,
        fit: str = "binned_irls",
    ) -> None:
        assert fit in ("binned_irls", "newton_step")
        self._fit = fit
        self.gamma = int(gamma)
        self.slope = torch.full((1,), _INIT_SLOPE, dtype=torch.float32, device=device)
        self.intercepts = torch.full(
            (self.gamma,), _INIT_INTERCEPT, dtype=torch.float32, device=device
        )
        self.num_refits = 0
        self._log_rank = log_rank
        self._positions = torch.arange(self.gamma, device=device).view(1, -1)
        # [gamma, 8] sufficient statistics of the current round, cleared on refit.
        self._stats = torch.zeros((self.gamma, 8), dtype=torch.float32, device=device)
        # [gamma, bins, (graded, accepted)] label counts of this round and the
        # decayed history the binned fit solves on.
        self._bin_centers = (
            _BIN_LOW + (torch.arange(_NUM_BINS, device=device) + 0.5) * _BIN_WIDTH
        )
        self._round_counts = torch.zeros(
            (self.gamma, _NUM_BINS, 2), dtype=torch.float32, device=device
        )
        self._history_counts = torch.zeros_like(self._round_counts)
        self._steps_since_refit = 0
        # The last predict's [bs, gamma] features and predictions, graded by observe.
        self._features: torch.Tensor | None = None
        self._predictions: torch.Tensor | None = None

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
        features = DraftTopLogOdds.execute(
            logits=step_logits.reshape(bs * gamma, vocab),
            temperatures=feature_temperatures,
            rows_per_request=gamma,
        ).view(bs, gamma)
        predictions = torch.sigmoid(self.slope * features + self.intercepts)
        self._features, self._predictions = features, predictions
        return predictions

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
        features, predictions = self._features, self._predictions
        assert features is not None, "observe() must follow predict()"
        self._features = self._predictions = None
        self._stats += _round_statistics(
            features=features,
            predictions=predictions,
            num_correct_drafts=num_correct_drafts,
            num_graded_drafts=num_graded_drafts,
            positions=self._positions,
        )
        if self._fit == "binned_irls":
            _accumulate_bin_counts(
                counts=self._round_counts,
                features=features,
                num_correct_drafts=num_correct_drafts,
                num_graded_drafts=num_graded_drafts,
                positions=self._positions,
            )
        self._steps_since_refit += 1
        if self._steps_since_refit >= self.REFIT_INTERVAL:
            self._refit()

    def skip_observe(self) -> None:
        """Drop the last prediction ungraded (e.g. its outcome was simulated)."""
        self._features = self._predictions = None

    def _refit(self) -> None:
        if self._log_rank and _should_log_refit(self.num_refits + 1):
            self._log_round()
        if self._fit == "binned_irls":
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
        else:
            step_slope, step_intercepts = _newton_step(
                stats=self._stats,
                slope=self.slope,
                l2=self.L2,
                damping=self.DAMPING_OBSERVATIONS,
            )
            self.slope += step_slope
            self.intercepts += step_intercepts
        self._stats.zero_()
        self._steps_since_refit = 0
        self.num_refits += 1

    def _log_round(self) -> None:
        stats = self._stats.cpu()
        graded = stats[:, _NUM_GRADED].clamp(min=1.0)
        logger.info(
            "Online acceptance estimator refit %d: slope=%.4f intercepts=%s "
            "graded=%s observed_accept=%s mean_pred=%s",
            self.num_refits + 1,
            float(self.slope.item()),
            _fmt(self.intercepts.cpu()),
            stats[:, _NUM_GRADED].to(torch.int64).tolist(),
            _fmt(stats[:, _NUM_ACCEPT] / graded),
            _fmt(stats[:, _PRED_SUM] / graded),
        )


def _round_statistics(
    *,
    features: torch.Tensor,
    predictions: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    num_graded_drafts: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """[gamma, 8] IRLS contributions of one graded block. For design row
    x = [f, e_k], weight w = p(1-p) and residual y - p: X^T W X has the arrowhead
    entries a_k = sum w f^2, b_k = sum w f, c_k = sum w, and X^T (y - p) has
    g_k = sum (y - p) f and h_k = sum (y - p)."""
    num_correct = num_correct_drafts.view(-1, 1).to(positions.dtype)
    num_graded = num_graded_drafts.view(-1, 1).to(positions.dtype)
    observed = (positions <= num_correct) & (positions < num_graded)
    labels = (positions < num_correct).to(torch.float32)
    weight = torch.where(observed, predictions * (1.0 - predictions), 0.0)
    residual = torch.where(observed, labels - predictions, 0.0)
    observed_f = observed.to(torch.float32)
    columns = (
        weight * features * features,
        weight * features,
        weight,
        residual * features,
        residual,
        observed_f,
        labels * observed_f,
        predictions * observed_f,
    )
    return torch.stack(columns, dim=-1).sum(dim=0)


def _newton_step(
    *,
    stats: torch.Tensor,
    slope: torch.Tensor,
    l2: float,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve (X^T W X + l2 I) d = X^T (y - p) for d = [d_slope, d_intercepts].

    The system is an arrowhead: the slope row couples to every position, each
    intercept only to itself, so the Schur complement gives d_slope in closed
    form and each d_intercept follows from it."""
    a_k, b_k, c_k = stats[:, _A], stats[:, _B], stats[:, _C] + l2
    g_k, h_k, n_k = stats[:, _G_SLOPE], stats[:, _G_INTERCEPT], stats[:, _NUM_GRADED]
    a = a_k.sum() + l2
    step_slope = (g_k.sum() - (b_k * h_k / c_k).sum()) / (a - (b_k * b_k / c_k).sum())
    total_graded = n_k.sum()
    step_slope = step_slope * total_graded / (total_graded + damping)
    # A NaN step or one that flips the slope negative is dropped.
    keep = torch.isfinite(step_slope) & (slope + step_slope >= 0.0)
    step_slope = torch.where(keep, step_slope, torch.zeros_like(step_slope))
    step_intercepts = (h_k - b_k * step_slope) / c_k
    step_intercepts = step_intercepts * n_k / (n_k + damping)
    step_intercepts = torch.where(
        torch.isfinite(step_intercepts),
        step_intercepts,
        torch.zeros_like(step_intercepts),
    )
    return step_slope.view(1), step_intercepts


def _accumulate_bin_counts(
    *,
    counts: torch.Tensor,
    features: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    num_graded_drafts: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    num_correct = num_correct_drafts.view(-1, 1).to(positions.dtype)
    num_graded = num_graded_drafts.view(-1, 1).to(positions.dtype)
    observed = ((positions <= num_correct) & (positions < num_graded)).to(torch.float32)
    accepted = observed * (positions < num_correct).to(torch.float32)
    bins = torch.clamp(
        ((features - _BIN_LOW) / _BIN_WIDTH).floor(), 0, _NUM_BINS - 1
    ).long()
    flat = (positions * _NUM_BINS + bins).view(-1)
    flat_counts = counts.view(-1, 2)
    flat_counts[:, 0].index_add_(0, flat, observed.view(-1))
    flat_counts[:, 1].index_add_(0, flat, accepted.view(-1))


def _fit_binned(
    *,
    counts: torch.Tensor,
    centers: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    prior_weight: float,
    num_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maximum a posteriori fit on the binned labels: Newton iterations with a
    batched backtracking line search, all on device. Unlike one Newton step per
    round, a confidently wrong prediction cannot fling a coefficient, and at the
    optimum each position's mean prediction matches its observed acceptance."""
    graded, accepted = counts[..., 0], counts[..., 1]
    f = centers.view(1, -1)
    prior_slope, prior_intercepts = slope.clone(), intercepts.clone()
    scales = torch.tensor(_STEP_SCALES, device=centers.device)
    for _ in range(num_iters):
        p = torch.sigmoid(slope * f + intercepts.view(-1, 1))
        weight = graded * p * (1.0 - p)
        residual = accepted - graded * p
        a_k = (weight * f * f).sum(dim=1)
        b_k = (weight * f).sum(dim=1)
        c_k = weight.sum(dim=1) + prior_weight
        g = (residual * f).sum() - prior_weight * (slope - prior_slope)
        h_k = residual.sum(dim=1) - prior_weight * (intercepts - prior_intercepts)
        a = a_k.sum() + prior_weight
        d_slope = (g - (b_k * h_k / c_k).sum()) / (a - (b_k * b_k / c_k).sum())
        d_intercepts = (h_k - b_k * d_slope) / c_k
        cand_slope = torch.clamp(slope + scales * d_slope, min=0.0)
        cand_intercepts = intercepts.view(1, -1) + scales.view(-1, 1) * d_intercepts
        z = cand_slope.view(-1, 1, 1) * f.view(1, 1, -1) + cand_intercepts.unsqueeze(-1)
        nll = (graded * torch.nn.functional.softplus(z) - accepted * z).sum(dim=(1, 2))
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
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"
