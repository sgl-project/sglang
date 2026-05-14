"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"
        )
    if server_args.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={server_args.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if server_args.enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    if server_args.enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorker does not implement adaptive)"
        )
    if server_args.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    if server_args.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


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
        candidates = set(cfg.get("candidate_steps", [1, 3, 7]))

        # Ensure the worker's initial speculative_num_steps is itself a candidate.
        # Otherwise AdaptiveController.register() would store the worker's pre-built
        # runtime state under a key that _activate() never queries, leaking that
        # state's draft attn backend and cuda graph buffers for the process lifetime.
        if initial_steps not in candidates:
            log_info_on_rank0(
                logger,
                f"Adding initial speculative_num_steps={initial_steps} to "
                f"candidate_steps={sorted(candidates)} so the pre-built "
                f"runtime state is reused.",
            )
            candidates.add(initial_steps)

        self.candidate_steps = sorted(candidates)
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

        self.current_steps = initial_steps

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={self.current_steps}, candidate_steps={self.candidate_steps}",
        )

    def update(self, num_accepted_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_accepted_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_accepted_drafts_per_req:
            return False

        batch_avg = sum(num_accepted_drafts_per_req) / len(num_accepted_drafts_per_req)
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
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False
