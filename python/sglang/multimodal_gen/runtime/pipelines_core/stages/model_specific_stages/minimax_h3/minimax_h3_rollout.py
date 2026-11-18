# SPDX-License-Identifier: Apache-2.0
"""Rollout (RL) helpers for MiniMax H3 denoising: the video target rows, or an image's."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDebugTensors,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)

_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


def _effective_sde_type(batch, loop_step_index: int) -> str:
    sde_type = str(getattr(batch, "rollout_sde_type", "sde"))
    if sde_type == "ode":
        return "ode"
    sde_step_indices = getattr(batch, "rollout_sde_step_indices", None)
    if sde_step_indices is not None and loop_step_index not in sde_step_indices:
        return "ode"
    return sde_type


def _step_is_noisy(batch, loop_step_index: int) -> bool:
    """Whether the rollout draws transition noise on this loop step."""
    return (
        _effective_sde_type(batch, loop_step_index) != "ode"
        and batch.rollout_noise_level != 0.0
    )


def minimax_h3_rollout_update_video_target(
    video_target: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
    batch,
    generator: torch.Generator,
    loop_step_index: int,
    noise_buffer: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Stochastic or deterministic update for video target rows during rollout.

    H3 velocity ``v`` relates to flow-matching model output as ``model_output = -v``.
    Returns ``(updated_target, log_prob_sum[B=1], log_prob_count[B=1], noise_buffer)``.
    """
    sde_type = _effective_sde_type(batch, loop_step_index)
    noise_level = float(getattr(batch, "rollout_noise_level", 0.0))
    log_prob_no_const = bool(getattr(batch, "rollout_log_prob_no_const", False))

    if not _step_is_noisy(batch, loop_step_index):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            _minimax_h3_update_target_rows_,
        )

        out = video_target.clone()
        sigma_t = out.new_tensor(sigma_curr)
        ratio = out.new_tensor(sigma_next / sigma_curr if sigma_curr != 0.0 else 1.0)
        one_minus_ratio = 1.0 - ratio
        scratch = torch.empty_like(out)
        _minimax_h3_update_target_rows_(
            out,
            velocity.float(),
            sigma_t=sigma_t,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
            one_minus_sigma_ratio=one_minus_ratio,
            denoised_scratch=scratch,
        )
        elem_count = float(out.numel())
        log_prob_sum = torch.zeros(1, device=out.device, dtype=torch.float32)
        log_prob_count = out.new_tensor([elem_count])
        return out, log_prob_sum, log_prob_count, noise_buffer

    prev_sample, _, _, variance_noise, log_prob_sum, log_prob_count = _noisy_transition(
        video_target,
        velocity,
        sde_type=sde_type,
        noise_level=noise_level,
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
        sigma_max=float(getattr(batch, "_h3_rollout_sigma_max", 1.0)),
        log_prob_no_const=log_prob_no_const,
        generator=generator,
    )
    if (
        noise_buffer is None
        or noise_buffer.shape != variance_noise.shape
        or noise_buffer.device != variance_noise.device
    ):
        noise_buffer = torch.empty_like(variance_noise)
    noise_buffer.copy_(variance_noise)
    return (
        prev_sample.to(dtype=video_target.dtype),
        log_prob_sum,
        log_prob_count,
        noise_buffer,
    )


def _noisy_transition(
    target: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sde_type: str,
    noise_level: float,
    sigma_curr: float,
    sigma_next: float,
    sigma_max: float,
    log_prob_no_const: bool,
    generator: torch.Generator,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """One cps or flow-sde step of target rows, in fp32.

    Returns ``(next_target, prev_sample_mean, noise_std_dev, variance_noise,
    log_prob_sum[B=1], log_prob_count[B=1])``, with
    ``next_target = prev_sample_mean + variance_noise * noise_std_dev``.
    ``sigma_max`` stands in for sigma == 1 in flow-sde's std.
    """
    model_output = (-velocity).float()
    sample = target.float()
    current_sigma = sample.new_tensor(float(sigma_curr))
    next_sigma = sample.new_tensor(float(sigma_next))

    variance_noise = torch.randn(
        sample.shape,
        generator=generator,
        device=sample.device,
        dtype=torch.float32,
    )

    if sde_type == "cps":
        std_dev_t = next_sigma * math.sin(noise_level * math.pi / 2)
        noise_std_dev = std_dev_t
        pred_original = sample - current_sigma * model_output
        noise_estimate = sample + model_output * (1.0 - current_sigma)
        prev_mean = pred_original * (1.0 - next_sigma) + noise_estimate * torch.sqrt(
            torch.clamp(next_sigma**2 - std_dev_t**2, min=1e-12)
        )
        prev_sample = prev_mean + variance_noise * noise_std_dev
        log_prob_no_const_val = -((variance_noise * noise_std_dev) ** 2)
    elif sde_type == "sde":
        dt = next_sigma - current_sigma
        std_dev_t = (
            torch.sqrt(
                current_sigma
                / (
                    1.0
                    - torch.where(
                        torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                        current_sigma.new_tensor(sigma_max),
                        current_sigma,
                    )
                )
            )
            * noise_level
        )
        noise_std_dev = std_dev_t * torch.sqrt(-1.0 * dt)
        prev_mean = (
            sample * (1.0 + std_dev_t**2 / (2.0 * current_sigma) * dt)
            + model_output
            * (1.0 + std_dev_t**2 * (1.0 - current_sigma) / (2.0 * current_sigma))
            * dt
        )
        prev_sample = prev_mean + variance_noise * noise_std_dev
        log_prob_no_const_val = -((variance_noise * noise_std_dev) ** 2)
    else:
        raise ValueError(f"Unsupported rollout_sde_type for H3: {sde_type!r}")

    if log_prob_no_const:
        log_prob_sum = log_prob_no_const_val.sum().unsqueeze(0)
    else:
        log_prob_sum = (
            (
                log_prob_no_const_val / (2.0 * (noise_std_dev**2))
                - torch.log(noise_std_dev)
                - _LOG_SQRT_2PI
            )
            .sum()
            .unsqueeze(0)
        )
    log_prob_count = sample.new_tensor([float(sample.numel())])
    return (
        prev_sample,
        prev_mean,
        noise_std_dev,
        variance_noise,
        log_prob_sum,
        log_prob_count,
    )


@dataclass
class MiniMaxH3RolloutCtx:
    """Per-denoise-loop rollout state for H3."""

    batch: Any
    generator: torch.Generator
    sigmas_video: list[float]
    collector: MiniMaxH3RolloutCollector
    noise_buffer: torch.Tensor | None = None
    denoising_env_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MiniMaxH3RolloutCollector:
    """Accumulates video-target trajectory and per-step log probs."""

    sigmas_video: list[float]
    return_step_indices: set[int] | None = None
    latent_steps: list[torch.Tensor] = field(default_factory=list)
    latent_step_indices: list[int] = field(default_factory=list)
    log_prob_sums: list[torch.Tensor] = field(default_factory=list)
    log_prob_counts: list[torch.Tensor] = field(default_factory=list)
    pos_cond_kwargs: dict[str, Any] = field(default_factory=dict)
    _next_latent_index: int = 0

    def _record_latent(self, video_target: torch.Tensor) -> None:
        index = self._next_latent_index
        self._next_latent_index += 1
        if (
            self.return_step_indices is not None
            and index not in self.return_step_indices
        ):
            return
        self.latent_steps.append(video_target.detach().cpu().clone())
        self.latent_step_indices.append(index)

    def record_initial(self, video_target: torch.Tensor) -> None:
        self._record_latent(video_target)

    def record_step(
        self,
        video_target: torch.Tensor,
        log_prob_sum: torch.Tensor,
        log_prob_count: torch.Tensor,
    ) -> None:
        self._record_latent(video_target)
        self.log_prob_sums.append(log_prob_sum.detach().cpu())
        self.log_prob_counts.append(log_prob_count.detach().cpu())

    def build_trajectory_data(self) -> RolloutTrajectoryData:
        # latents: [B=1, T+1, num_video_target_rows, width]
        stacked = torch.stack(self.latent_steps, dim=0).unsqueeze(0)
        divisor = 1000.0
        step_sigmas = torch.tensor(
            [float(s) for s in self.sigmas_video[:-1]],
            dtype=torch.float32,
        )
        timesteps = step_sigmas * divisor
        sigmas = torch.tensor(
            [float(s) for s in self.sigmas_video],
            dtype=torch.float32,
        )
        log_probs = None
        if self.log_prob_sums:
            sums = torch.stack(self.log_prob_sums, dim=0)
            counts = torch.stack(self.log_prob_counts, dim=0)
            per_step = sums.squeeze(-1) / counts.squeeze(-1).clamp(min=1.0)
            log_probs = per_step.unsqueeze(0)
        return RolloutTrajectoryData(
            rollout_log_probs=log_probs,
            denoising_env=RolloutDenoisingEnv(
                pos_cond_kwargs=self.pos_cond_kwargs,
            ),
            dit_trajectory=RolloutDitTrajectory(
                latents=stacked,
                timesteps=timesteps,
                sigmas=sigmas,
                latent_step_indices=torch.tensor(
                    self.latent_step_indices, dtype=torch.long
                ),
            ),
        )


class MiniMaxH3ImageRollout:
    """One image request's rollout: its noisy steps, and the trajectory the trainer replays."""

    def __init__(
        self,
        *,
        sigmas: list[float],
        noisy_steps: frozenset[int],
        sde_type: str,
        noise_level: float,
        log_prob_no_const: bool,
        debug_mode: bool,
        return_step_indices: list[int] | None,
        generator: torch.Generator,
        pos_cond_kwargs: dict[str, Any],
    ) -> None:
        self._sigmas = sigmas
        # FlowGRPO's sigma_max, as FlowRolloutController takes it: the grid's second sigma.
        self._sigma_max = sigmas[min(1, len(sigmas) - 1)]
        self._noisy_steps = noisy_steps
        self._sde_type = sde_type
        self._noise_level = noise_level
        self._log_prob_no_const = log_prob_no_const
        self._generator = generator
        self._collector = MiniMaxH3RolloutCollector(
            sigmas_video=sigmas,
            return_step_indices=(
                None if return_step_indices is None else set(return_step_indices)
            ),
            pos_cond_kwargs=pos_cond_kwargs,
        )
        # Per step: variance noise, prev sample mean, noise std dev, model output.
        # Every step is kept, whichever latents the request asks back.
        self._debug_tensors_by_step: list[tuple[torch.Tensor, ...]] | None = (
            [] if debug_mode else None
        )

    @classmethod
    def for_request(
        cls,
        batch,
        *,
        sigmas: list[float],
        generator: torch.Generator,
        pos_cond_kwargs: dict[str, Any],
    ) -> MiniMaxH3ImageRollout:
        return cls(
            sigmas=sigmas,
            noisy_steps=frozenset(
                step for step in range(len(sigmas) - 1) if _step_is_noisy(batch, step)
            ),
            sde_type=batch.rollout_sde_type,
            noise_level=batch.rollout_noise_level,
            log_prob_no_const=batch.rollout_log_prob_no_const,
            debug_mode=batch.rollout_debug_mode,
            return_step_indices=batch.rollout_return_step_indices,
            generator=generator,
            pos_cond_kwargs=pos_cond_kwargs,
        )

    def record_initial(self, target_rows: torch.Tensor) -> None:
        self._collector.record_initial(target_rows)

    def step_is_noisy(self, step: int) -> bool:
        return step in self._noisy_steps

    def take_noisy_step(
        self, target_rows: torch.Tensor, velocity: torch.Tensor, *, step: int
    ) -> torch.Tensor:
        """Record the step's cps or flow-sde transition and return the rows after it."""
        (
            next_rows,
            prev_sample_mean,
            noise_std_dev,
            variance_noise,
            log_prob_sum,
            _,
        ) = _noisy_transition(
            target_rows,
            velocity,
            sde_type=self._sde_type,
            noise_level=self._noise_level,
            sigma_curr=self._sigmas[step],
            sigma_next=self._sigmas[step + 1],
            sigma_max=self._sigma_max,
            log_prob_no_const=self._log_prob_no_const,
            generator=self._generator,
        )
        self._record_step(
            next_rows,
            velocity,
            log_prob_sum=log_prob_sum,
            variance_noise=variance_noise,
            prev_sample_mean=prev_sample_mean,
            noise_std_dev=noise_std_dev,
        )
        return next_rows

    def record_deterministic_step(
        self, next_rows: torch.Tensor, velocity: torch.Tensor
    ) -> None:
        """Record a step the loop took with its own update: no noise, log-prob 0."""
        self._record_step(
            next_rows,
            velocity,
            log_prob_sum=torch.zeros(1, dtype=torch.float32),
            variance_noise=torch.zeros_like(next_rows),
            prev_sample_mean=next_rows,
            noise_std_dev=torch.zeros((), dtype=torch.float32),
        )

    def _record_step(
        self,
        next_rows: torch.Tensor,
        velocity: torch.Tensor,
        *,
        log_prob_sum: torch.Tensor,
        variance_noise: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        noise_std_dev: torch.Tensor,
    ) -> None:
        self._collector.record_step(
            video_target=next_rows,
            log_prob_sum=log_prob_sum,
            log_prob_count=torch.tensor([float(next_rows.numel())]),
        )
        if self._debug_tensors_by_step is not None:
            self._debug_tensors_by_step.append(
                (
                    variance_noise.detach().cpu().clone(),
                    prev_sample_mean.detach().float().cpu().clone(),
                    noise_std_dev.detach().float().cpu().reshape(1),
                    # the trainer's convention: noise - clean
                    (-velocity).detach().float().cpu(),
                )
            )

    def build_trajectory_data(self) -> RolloutTrajectoryData:
        trajectory = self._collector.build_trajectory_data()
        if self._debug_tensors_by_step is not None:
            # [B=1, T, ...], the layout FlowRolloutController returns
            variance_noises, prev_sample_means, noise_std_devs, model_outputs = (
                torch.stack(per_step, dim=0).unsqueeze(0)
                for per_step in zip(*self._debug_tensors_by_step)
            )
            trajectory.rollout_debug_tensors = RolloutDebugTensors(
                rollout_variance_noises=variance_noises,
                rollout_prev_sample_means=prev_sample_means,
                rollout_noise_std_devs=noise_std_devs,
                rollout_model_outputs=model_outputs,
            )
        return trajectory


__all__ = [
    "MiniMaxH3ImageRollout",
    "MiniMaxH3RolloutCollector",
    "MiniMaxH3RolloutCtx",
    "minimax_h3_rollout_update_video_target",
]
