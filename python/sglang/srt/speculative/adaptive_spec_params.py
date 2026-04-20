"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

import json
import logging

logger = logging.getLogger(__name__)


def load_adaptive_config(path: str | None) -> dict[str, object]:
    """Load adaptive speculative config from a JSON file.

    The file may contain any subset of the following keys:
        ema_alpha, update_interval, warmup_batches,
        down_hysteresis, up_hysteresis, candidate_steps

    Returns an empty dict when *path* is ``None``.
    """
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            "speculative_adaptive_config must be a JSON object, "
            f"got {type(cfg).__name__}"
        )
    return cfg


class AdaptiveSpeculativeParams:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    """

    def __init__(
        self,
        initial_steps: int,
        config: dict[str, object] | None = None,
    ):
        cfg = config or {}
        # TODO: Wider range of candidate_steps (once lazy init is supported).
        self.candidate_steps = sorted(set(cfg.get("candidate_steps", [1, 3, 7])))
        assert (
            len(self.candidate_steps) >= 2
        ), "candidate_steps must have at least 2 distinct values"

        self.min_steps = self.candidate_steps[0]
        self.max_steps = self.candidate_steps[-1]
        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)

        self.current_steps = min(
            self.candidate_steps,
            key=lambda step: (abs(step - initial_steps), -step),
        )

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

        logger.info(
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={self.current_steps}, candidate_steps={self.candidate_steps}"
        )

    def update(self, accept_lengths: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            accept_lengths: Per-request accepted draft token counts from last verify.
        """
        if not accept_lengths:
            return False

        batch_avg = sum(accept_lengths) / len(accept_lengths)
        self.ema_accept_len = (
            1 - self.ema_alpha
        ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            drop_threshold = prev_step - 0.5 + self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        while current_idx < len(self.candidate_steps) - 1:
            current_step = self.candidate_steps[current_idx]
            rise_threshold = current_step - 0.5 + self.up_hysteresis
            if self.ema_accept_len > rise_threshold:
                current_idx += 1
            else:
                break

        target = self.candidate_steps[current_idx]

        if target != old_steps:
            self.current_steps = target
            logger.info(
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})"
            )
            return True
        return False
