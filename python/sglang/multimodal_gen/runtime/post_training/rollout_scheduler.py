# SPDX-License-Identifier: Apache-2.0
"""Per-request scheduler switching for RL rollout.

Serving and RL rollout want different schedulers on the same engine: serving
keeps the model's scheduler (e.g. UniPC for Wan), while the rollout
SDE/log-prob path requires a first-order flow-match Euler scheduler.
``RolloutSchedulerSwitch`` holds both and dispatches per request on
``batch.rollout``; every attribute it does not define delegates to the
active scheduler, so consumers see a plain scheduler either way.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RolloutTimestepPreparationStage(TimestepPreparationStage):
    """Resolve the per-request scheduler before preparing timesteps."""

    def forward(self, batch, server_args):
        self.scheduler.prepare_for_batch(batch)
        return super().forward(batch, server_args)


class RolloutSchedulerSwitch(SchedulerRLMixin):
    """Dispatch between a serving and a rollout scheduler per request."""

    def __init__(self, serving_scheduler, rollout_scheduler):
        self.serving_scheduler = serving_scheduler
        self.rollout_scheduler = rollout_scheduler
        self._active_scheduler = serving_scheduler
        self._logged_rollout_check = False

    def prepare_for_batch(self, batch):
        self._active_scheduler = (
            self.rollout_scheduler if batch.rollout else self.serving_scheduler
        )
        return self._active_scheduler

    @property
    def active_scheduler(self):
        return self._active_scheduler

    @property
    def order(self):
        return self._active_scheduler.order

    @property
    def num_train_timesteps(self):
        return self._active_scheduler.num_train_timesteps

    @property
    def timesteps(self):
        return self._active_scheduler.timesteps

    @property
    def sigmas(self):
        return self._active_scheduler.sigmas

    @property
    def config(self):
        return self._active_scheduler.config

    def __getattr__(self, name):
        return getattr(self._active_scheduler, name)

    def set_shift(self, shift: float) -> None:
        # Fan out so a launch-time flow_shift override reaches both paths.
        self.serving_scheduler.set_shift(shift)
        self.rollout_scheduler.set_shift(shift)

    def set_begin_index(self, begin_index: int = 0):
        return self._active_scheduler.set_begin_index(begin_index)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device=None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
        **kwargs,
    ):
        if self._active_scheduler is self.serving_scheduler:
            if timesteps is not None:
                raise ValueError(
                    "the serving scheduler does not support custom timesteps"
                )
            self.serving_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                **kwargs,
            )
            return

        self.rollout_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
            **kwargs,
        )
        self._check_rollout_timesteps()

    def _check_rollout_timesteps(self) -> None:
        # The rollout SDE/log-prob math assumes the flow-match Euler
        # convention timesteps == sigmas[:-1] * num_train_timesteps.
        sigmas = self.rollout_scheduler.sigmas
        timesteps = self.rollout_scheduler.timesteps
        if sigmas is None or timesteps is None or sigmas.numel() < 2:
            return
        reconstructed = sigmas[:-1].to(device=timesteps.device) * float(
            self.rollout_scheduler.config.num_train_timesteps
        )
        max_abs_diff = (timesteps.float() - reconstructed.float()).abs().max().item()
        if max_abs_diff > 1e-3:
            raise ValueError(
                f"rollout timestep/sigma mismatch: max_abs_diff={max_abs_diff:.6g}"
            )
        if not self._logged_rollout_check:
            logger.info(
                "RL rollout using %s (timesteps dtype=%s, sigmas dtype=%s, "
                "max_abs_diff=%.6g)",
                type(self.rollout_scheduler).__name__,
                timesteps.dtype,
                sigmas.dtype,
                max_abs_diff,
            )
            self._logged_rollout_check = True

    def scale_model_input(self, sample, timestep=None):
        return self._active_scheduler.scale_model_input(sample, timestep)

    def step(
        self,
        model_output,
        timestep,
        sample,
        generator=None,
        batch=None,
        return_dict: bool = True,
        **kwargs,
    ):
        if self._active_scheduler is self.serving_scheduler:
            return self.serving_scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                generator=generator,
                return_dict=return_dict,
            )
        return self.rollout_scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            batch=batch,
            return_dict=return_dict,
            **kwargs,
        )

    def index_for_timestep(self, *args, **kwargs):
        return self._active_scheduler.index_for_timestep(*args, **kwargs)
