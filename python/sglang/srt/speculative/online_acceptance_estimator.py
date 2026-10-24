"""Online per-position acceptance estimator for drafts without a confidence head.

Confidence-scheduled verification needs the probability that each draft token
is accepted given that every earlier one was. A draft model without a trained
confidence head gets it from a logistic model calibrated while serving:

    p_k = sigmoid(slope * x_k + bias_k)

where x_k = logit(q_max) is the log-odds of the most likely token of the draft
distribution at position k, slope is shared by all positions and bias_k is
per position. The feature scores the draft distribution, not the sampled
token, so trimming drafts by it never conditions on the trimmed token and
rejection sampling stays lossless.

The coefficients are refit by one damped Newton (IRLS) step every
``refit_interval`` observations. Everything stays on the device and no call
synchronizes with the host, so the estimator runs under overlap scheduling and
inside CUDA graphs. All state updates are deterministic, so ranks that observe
identical inputs keep identical coefficients.
"""

from typing import Tuple

import torch
import triton

from sglang.kernels.ops.speculative.acceptance_estimator import (
    NUM_ACCEPTANCE_STATS,
    max_log_odds,
    observe_acceptance,
)

# The feature log(q / (1 - q)) is infinite for a row with all of its mass on one
# token; clamp it symmetrically.
_MAX_LOG_ODDS = 40.0
# Cold-start coefficients. Log-loss is flat near the optimum, so one starting
# point serves typical draft models and the first refits replace it.
_INIT_SLOPE = 0.5
_INIT_BIAS = -0.2
# Newton steps are damped by n / (n + _DAMPING_OBSERVATIONS): a round with this
# many observations moves a coefficient half of a full step, so data-poor
# positions keep learning, only slowly.
_DAMPING_OBSERVATIONS = 50.0
# Ridge on the Newton solve.
_L2 = 1e-3
# Rows of requests the observe kernel processes per iteration.
_MAX_BLOCK_REQS = 256


class OnlineAcceptanceEstimator:
    """Per-position conditional acceptance probabilities from draft confidence.

    Per verify step the caller:
    1. ``feature_from_scores`` on the draft scores of each position,
    2. ``predict`` to get the conditional acceptance probabilities,
    3. after verification, ``observe`` with the same features and the verify
       outcome.
    ``feature_and_predict`` does steps 1 and 2 in one kernel launch.
    ``observe`` evaluates the Newton weights at the current coefficients, so it
    may run before or after the next step's ``predict``.
    """

    def __init__(
        self,
        num_positions: int,
        max_num_reqs: int,
        device: torch.device,
        refit_interval: int = 100,
    ) -> None:
        assert num_positions > 0 and max_num_reqs > 0 and refit_interval > 0
        self.num_positions = num_positions
        self.refit_interval = refit_interval
        self._block_reqs = min(triton.next_power_of_2(max_num_reqs), _MAX_BLOCK_REQS)
        # Coefficients are updated in place, so CUDA graphs that read them stay
        # valid across refits.
        self.slope = torch.full((1,), _INIT_SLOPE, dtype=torch.float32, device=device)
        self.intercepts = torch.full(
            (num_positions,), _INIT_BIAS, dtype=torch.float32, device=device
        )
        # Per-round Newton statistics per position, cleared at every refit.
        self._stats = torch.zeros(
            (num_positions, NUM_ACCEPTANCE_STATS), dtype=torch.float32, device=device
        )
        # observe() calls since the last refit.
        self._num_calls = torch.zeros((1,), dtype=torch.int32, device=device)

    def feature_from_scores(
        self, scores: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        """scores [bs, num_positions, num_candidates]: the draft logits each
        position sampled from; temperatures: per request, greedy rows <= 0.
        Returns logit(q_max) of the temperature-scaled rows, float32
        [bs, num_positions]."""
        assert scores.dim() == 3 and scores.shape[1] == self.num_positions
        return max_log_odds(scores, temperatures, _MAX_LOG_ODDS)[0]

    def feature_and_predict(
        self, scores: torch.Tensor, temperatures: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``feature_from_scores`` and ``predict`` fused into one kernel launch.
        Returns the features and the conditional acceptance probabilities, both
        float32 [bs, num_positions]. Does not change any state."""
        assert scores.dim() == 3 and scores.shape[1] == self.num_positions
        return max_log_odds(
            scores, temperatures, _MAX_LOG_ODDS, self.slope, self.intercepts
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Conditional acceptance probability of each draft position, float32
        [bs, num_positions]. Does not change any state."""
        assert x.dim() == 2 and x.shape[1] == self.num_positions
        return torch.sigmoid(torch.addcmul(self.intercepts, x.float(), self.slope))

    def observe(
        self,
        x: torch.Tensor,
        num_verified: torch.Tensor,
        num_accepted: torch.Tensor,
    ) -> None:
        """Learn from one verify step.

        x: the features the verified drafts were predicted with, [bs,
        num_positions]; num_verified: draft tokens the target verified per
        request, the anchor excluded; num_accepted: how many of them it
        accepted, the bonus token excluded. Positions up to and including the
        first rejection are observed, labelled by whether they were accepted.
        Rows with num_verified = 0 contribute nothing and their features are
        not read, so padding rows must pass 0. Every ``refit_interval``-th
        call refits the coefficients and starts a new round.

        This updates the estimator, so run it (or replay a graph that captured
        it) exactly once per real verify step, never in warmup or dummy
        forwards.
        """
        assert x.dim() == 2 and x.shape[1] == self.num_positions
        assert num_verified.numel() == num_accepted.numel() == x.shape[0]
        observe_acceptance(
            x,
            num_verified.reshape(-1).contiguous(),
            num_accepted.reshape(-1).contiguous(),
            self.slope,
            self.intercepts,
            self._stats,
            self._num_calls,
            refit_interval=self.refit_interval,
            l2=_L2,
            damping=_DAMPING_OBSERVATIONS,
            block_reqs=self._block_reqs,
        )
