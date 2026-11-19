"""Online acceptance estimator for confidence-scheduled verification.

Predicts each draft position's acceptance probability, given that every earlier
draft was accepted, from the draft distribution alone:
``p_k = sigmoid(slope * logit(q_max_k) + intercepts[k])``, and refits the shared
slope and the per-position intercepts on the target's verify outcomes while
serving. It needs only the draft logits of each position, so any drafter can
use it in place of a trained confidence head.

The feature reads only ``q_k``, which depends on the drafts before position k
and never on the drafted token itself, so trimming verification on it stays
lossless. Scoring the drawn token's probability instead would select on the
proposal and bias the output toward the drafter's confident modes.

Labels are binned by (position, feature) and kept with a decayed history; each
refit runs Newton iterations with a line search to the penalized maximum
likelihood. A single Newton step per round is cheaper but unstable: a
confidently wrong prediction carries a full residual and almost no curvature,
and when acceptance saturates below one for confident drafts (a sampled target
keeps its own entropy) the slope stalls and survival is over-predicted.

Every TP rank sees identical inputs and fits identically, on device.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.speculative.acceptance_estimator import (
    NUM_BINS,
    AccumulateAcceptanceCounts,
    PredictAcceptance,
    RefitAcceptance,
    bin_centers,
)

logger = logging.getLogger(__name__)

_INIT_SLOPE = 0.5
_INIT_INTERCEPT = -0.2


class OnlineAcceptanceEstimator:
    """Per verify step the caller runs ``predict`` on the drafted block, then
    ``observe`` with the features it returned once the target has verified
    that block. ``predict`` only reads the coefficients, which refits update in
    place, so a CUDA graph that captured it stays valid."""

    REFIT_INTERVAL = 100
    HISTORY_DECAY = 0.75
    PRIOR_WEIGHT = 1.0
    NEWTON_ITERS = 8

    def __init__(
        self, *, num_positions: int, device: torch.device, log_rank: bool
    ) -> None:
        self.num_positions = int(num_positions)
        self.slope = torch.full((1,), _INIT_SLOPE, dtype=torch.float32, device=device)
        self.intercepts = torch.full(
            (self.num_positions,), _INIT_INTERCEPT, dtype=torch.float32, device=device
        )
        self.refit_ct = 0
        self._log_rank = log_rank
        self._round_counts = torch.zeros(
            (self.num_positions, NUM_BINS, 2), dtype=torch.float32, device=device
        )
        self._history_counts = torch.zeros_like(self._round_counts)
        self._observe_ct = 0
        self._warm_up(device)

    def _warm_up(self, device: torch.device) -> None:
        """Compiles the observe and refit kernels at startup, on scratch state;
        draft graph capture compiles predict for the real logits."""
        counts = torch.zeros_like(self._round_counts)
        AccumulateAcceptanceCounts.execute(
            counts=counts,
            features=torch.zeros((1, self.num_positions), device=device),
            num_correct_drafts=torch.zeros(1, dtype=torch.int32, device=device),
            num_graded_drafts=torch.zeros(1, dtype=torch.int32, device=device),
        )
        RefitAcceptance.execute(
            round_counts=counts,
            history_counts=torch.zeros_like(counts),
            slope=self.slope.clone(),
            intercepts=self.intercepts.clone(),
            decay=self.HISTORY_DECAY,
            prior_weight=self.PRIOR_WEIGHT,
            num_iters=self.NEWTON_ITERS,
        )

    def predict(
        self,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        features_out: Optional[torch.Tensor] = None,
        predictions_out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[bs, num_positions] features and acceptance predictions of the
        [bs, num_positions, vocab] draft logits each position sampled from.
        ``temperatures``: [bs], non-positive for greedy rows."""
        bs, num_positions, _ = logits.shape
        assert num_positions == self.num_positions
        if features_out is None:
            features_out = torch.empty(
                (bs, num_positions), dtype=torch.float32, device=logits.device
            )
        if predictions_out is None:
            predictions_out = torch.empty_like(features_out)
        PredictAcceptance.execute(
            logits=logits,
            temperatures=temperatures.reshape(-1).to(torch.float32).contiguous(),
            slope=self.slope,
            intercepts=self.intercepts,
            features_out=features_out,
            predictions_out=predictions_out,
        )
        return features_out, predictions_out

    def observe(
        self,
        *,
        features: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        num_graded_drafts: torch.Tensor,
    ) -> None:
        """Learn from one verified block. ``features``: what ``predict``
        returned for it; ``num_correct_drafts``: [bs] drafts the target accepted
        before its first rejection, uncapped by any verify cut;
        ``num_graded_drafts``: [bs] leading drafts checked against a real target
        row. Drafts up to and including the first rejection carry labels.

        Run it exactly once per real verify step, never for warmup, dummy or
        simulated-acceptance steps."""
        AccumulateAcceptanceCounts.execute(
            counts=self._round_counts,
            features=features,
            num_correct_drafts=num_correct_drafts,
            num_graded_drafts=num_graded_drafts,
        )
        self._observe_ct += 1
        if self._observe_ct % self.REFIT_INTERVAL == 0:
            self._refit()

    def _refit(self) -> None:
        log = self._log_rank and _should_log_refit(self.refit_ct + 1)
        if log:
            round_counts = self._round_counts.clone()
        RefitAcceptance.execute(
            round_counts=self._round_counts,
            history_counts=self._history_counts,
            slope=self.slope,
            intercepts=self.intercepts,
            decay=self.HISTORY_DECAY,
            prior_weight=self.PRIOR_WEIGHT,
            num_iters=self.NEWTON_ITERS,
        )
        self.refit_ct += 1
        if log:
            self._log_round(round_counts)

    def _log_round(self, round_counts: torch.Tensor) -> None:
        graded, correct = round_counts.unbind(dim=-1)
        predictions = torch.sigmoid(
            self.slope * bin_centers(graded.device).view(1, -1)
            + self.intercepts.view(-1, 1)
        )
        num_graded = graded.sum(dim=1)
        denom = num_graded.clamp(min=1.0)
        logger.info(
            "Online acceptance estimator refit %d: slope=%.4f intercepts=%s "
            "graded=%s observed_accept=%s refit_mean_pred=%s",
            self.refit_ct,
            float(self.slope.item()),
            _fmt(self.intercepts),
            num_graded.to(torch.int64).tolist(),
            _fmt(correct.sum(dim=1) / denom),
            _fmt((graded * predictions).sum(dim=1) / denom),
        )


def _should_log_refit(refit: int) -> bool:
    return refit in (1, 10) or refit % 50 == 0


def _fmt(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values.cpu()) + "]"
