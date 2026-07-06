# SPDX-License-Identifier: Apache-2.0
"""Mixin for per-request rollout scheduler binding in TimestepPreparationStage.

Kept under post_training to keep the core stage lean; mirrors
RolloutDenoisingMixin on DenoisingStage.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RolloutTimestepPreparationMixin:
    """Bind an alternate scheduler to rollout=True requests.

    The rollout SDE/log-prob path needs a first-order flow-match Euler
    scheduler, which not every pipeline serves (e.g. Wan serves UniPC). The
    host stage sets ``self.rollout_scheduler``; None keeps the serving
    scheduler for rollout requests. Downstream stages read the scheduler
    from ``batch.scheduler``, so the host stage is the single switch point.
    """

    # Class-level so the rollout info log prints once per process, not once
    # per stage instance.
    _logged_rollout_scheduler_check = False

    def _resolve_rollout_scheduler(self, batch: Req):
        """Return the rollout scheduler template for this request, or None."""
        if batch.rollout and self.rollout_scheduler is not None:
            return self.rollout_scheduler
        return None

    def _check_rollout_timesteps(self, scheduler) -> None:
        # The rollout SDE/log-prob math assumes the flow-match Euler
        # convention timesteps == sigmas[:-1] * num_train_timesteps.
        sigmas = scheduler.sigmas
        timesteps = scheduler.timesteps
        if sigmas is None or timesteps is None or sigmas.numel() < 2:
            return
        reconstructed = sigmas[:-1].to(device=timesteps.device) * float(
            scheduler.config.num_train_timesteps
        )
        max_abs_diff = (timesteps.float() - reconstructed.float()).abs().max().item()
        if max_abs_diff > 1e-3:
            raise ValueError(
                f"rollout timestep/sigma mismatch: max_abs_diff={max_abs_diff:.6g}"
            )
        if not RolloutTimestepPreparationMixin._logged_rollout_scheduler_check:
            logger.info(
                "RL rollout using %s (timesteps dtype=%s, sigmas dtype=%s, "
                "max_abs_diff=%.6g)",
                type(scheduler).__name__,
                timesteps.dtype,
                sigmas.dtype,
                max_abs_diff,
            )
            RolloutTimestepPreparationMixin._logged_rollout_scheduler_check = True
