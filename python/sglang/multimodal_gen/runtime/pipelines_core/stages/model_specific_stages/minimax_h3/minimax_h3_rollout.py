# SPDX-License-Identifier: Apache-2.0
"""Rollout (RL) helpers for MiniMax H3 video-only denoising."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
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

    if sde_type == "ode" or noise_level == 0.0:
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

    model_output = (-velocity).float()
    sample = video_target.float()
    current_sigma = sample.new_tensor(float(sigma_curr))
    next_sigma = sample.new_tensor(float(sigma_next))

    if (
        noise_buffer is None
        or noise_buffer.shape != sample.shape
        or noise_buffer.device != sample.device
    ):
        noise_buffer = torch.empty_like(sample)
    variance_noise = torch.randn(
        sample.shape,
        generator=generator,
        device=sample.device,
        dtype=torch.float32,
    )
    noise_buffer.copy_(variance_noise)

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
        sigma_max = float(getattr(batch, "_h3_rollout_sigma_max", 1.0))
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

    if log_prob_no_const or sde_type == "ode":
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
        prev_sample.to(dtype=video_target.dtype),
        log_prob_sum,
        log_prob_count,
        noise_buffer,
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
    latent_steps: list[torch.Tensor] = field(default_factory=list)
    log_prob_sums: list[torch.Tensor] = field(default_factory=list)
    log_prob_counts: list[torch.Tensor] = field(default_factory=list)
    pos_cond_kwargs: dict[str, Any] = field(default_factory=dict)

    def record_initial(self, video_target: torch.Tensor) -> None:
        self.latent_steps.append(video_target.detach().cpu().clone())

    def record_step(
        self,
        video_target: torch.Tensor,
        log_prob_sum: torch.Tensor,
        log_prob_count: torch.Tensor,
    ) -> None:
        self.latent_steps.append(video_target.detach().cpu().clone())
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
            ),
        )


__all__ = [
    "MiniMaxH3RolloutCollector",
    "MiniMaxH3RolloutCtx",
    "minimax_h3_rollout_update_video_target",
]
