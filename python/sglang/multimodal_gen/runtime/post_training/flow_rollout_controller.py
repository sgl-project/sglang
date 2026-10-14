# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDebugTensors,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


@dataclass
class RolloutSessionData:
    """Per-batch rollout state created by prepare_rollout(), lives on the batch object.

    Cleared by setting ``batch._rollout_session_data = None``.
    """

    pipeline_config: Any = None
    sigma_max: float = 0.0
    latents_shape: tuple | None = None
    noise_buffer: torch.Tensor | None = None

    local_log_prob_sum: list[torch.Tensor] = field(default_factory=list)
    local_log_prob_count: list[torch.Tensor] = field(default_factory=list)

    local_variance_noises: list[torch.Tensor] = field(default_factory=list)
    local_prev_sample_means: list[torch.Tensor] = field(default_factory=list)
    local_noise_std_devs: list[torch.Tensor] = field(default_factory=list)
    local_model_outputs: list[torch.Tensor] = field(default_factory=list)

    def _reset_rollout_debug_tensors(self) -> None:
        self.local_variance_noises = []
        self.local_prev_sample_means = []
        self.local_noise_std_devs = []
        self.local_model_outputs = []

    def append_local_rollout_debug_tensors(
        self,
        *,
        variance_noise: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        noise_std_dev: torch.Tensor,
        model_output: torch.Tensor,
    ) -> None:
        batch_size = variance_noise.shape[0]
        # the underlying noise buffer in self.noise_buffer is reused.
        self.local_variance_noises.append(variance_noise.clone())
        self.local_prev_sample_means.append(prev_sample_mean)
        self.local_noise_std_devs.append(noise_std_dev.expand((batch_size, 1)))
        self.local_model_outputs.append(model_output)

    def consume_local_rollout_debug_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        variance_noises = torch.stack(self.local_variance_noises, dim=1)
        prev_sample_means = torch.stack(self.local_prev_sample_means, dim=1)
        noise_std_devs = torch.stack(self.local_noise_std_devs, dim=1)
        model_outputs = torch.stack(self.local_model_outputs, dim=1)
        self._reset_rollout_debug_tensors()
        return variance_noises, prev_sample_means, noise_std_devs, model_outputs

    def collect_rollout_debug_tensors(self, batch: Req) -> RolloutDebugTensors:
        """
        Consume rollout debug tensors and merge for all SP ranks.

        Returns rollout debug tensors with shape [B, T, ...].
        """
        variance_noises, prev_sample_means, noise_std_devs, model_outputs = (
            self.consume_local_rollout_debug_tensors()
        )

        pipeline_config = self.pipeline_config
        variance_noises = gather_stacked_latents_for_sp(
            pipeline_config=pipeline_config,
            batch=batch,
            stacked_latents=variance_noises,
        )
        prev_sample_means = gather_stacked_latents_for_sp(
            pipeline_config=pipeline_config,
            batch=batch,
            stacked_latents=prev_sample_means,
        )
        model_outputs = gather_stacked_latents_for_sp(
            pipeline_config=pipeline_config,
            batch=batch,
            stacked_latents=model_outputs,
        )
        # noise_std_devs is same on every device, not a sharded latent tensor.

        return RolloutDebugTensors(
            rollout_variance_noises=variance_noises.cpu(),
            rollout_prev_sample_means=prev_sample_means.cpu(),
            rollout_noise_std_devs=noise_std_devs.cpu(),
            rollout_model_outputs=model_outputs.cpu(),
        )

    def _get_or_create_rollout_noise_buffer(
        self,
        full_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or create the reusable noise buffer (local or full shape) for rollout."""
        buffer = self.noise_buffer
        if (
            buffer is None
            or buffer.shape != full_shape
            or buffer.dtype != dtype
            or buffer.device != device
        ):
            buffer = torch.empty(full_shape, device=device, dtype=dtype)
            self.noise_buffer = buffer
        return buffer

    def _rollout_variance_noise(
        self,
        batch,
        model_output: torch.FloatTensor,
        generator: Union[torch.Generator, list[torch.Generator]],
    ) -> torch.FloatTensor:
        """Generate variance noise for rollout. If generator is a list, use generator[i] for the i-th batch item."""
        assert generator is not None, "Generator must be provided"

        device = model_output.device
        dtype = model_output.dtype
        local_shape = tuple(model_output.shape)

        B = local_shape[0]
        if isinstance(generator, torch.Generator):
            assert B == 1, "Generator must be a list if batch size is not 1"
            generator = [generator]
        else:
            assert len(generator) == B, (
                "Generator list must have the same length as batch size"
            )

        buffer = self._get_or_create_rollout_noise_buffer(
            self.latents_shape, device, dtype
        )
        for i in range(B):
            torch.randn(
                self.latents_shape,
                out=buffer[i : i + 1],
                generator=generator[i],
            )

        sharded_noise, _ = self.pipeline_config.shard_latents_for_sp(
            batch=batch, latents=buffer
        )
        if tuple(sharded_noise.shape) != local_shape:
            raise ValueError(
                "Rollout SP noise shape mismatch after shard. "
                f"Expected local_shape={local_shape}, got {tuple(sharded_noise.shape)}."
            )
        return sharded_noise

    def flow_sde_sampling(
        self,
        batch,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        current_sigma: torch.FloatTensor,
        next_sigma: torch.FloatTensor,
        generator: torch.Generator,
        step_index: int,
    ) -> torch.Tensor:
        """Flow rollout step for log-prob / sampling (see FlowGRPO-style references).

        ``rollout_sde_type`` (from batch SamplingParams):

        1. ``"sde"``: Standard stochastic differential equation transition (Gaussian).
        2. ``"cps"``: Coupled Particle Sampling.
        3. ``"ode"``: Deterministic ODE step (no diffusion noise).
        """
        sde_type = batch.rollout_sde_type
        noise_level = float(batch.rollout_noise_level)
        log_prob_no_const = batch.rollout_log_prob_no_const
        debug_mode = bool(getattr(batch, "rollout_debug_mode", False))

        if not log_prob_no_const and sde_type != "ode":
            assert noise_level > 0, (
                "True log-probability computation requires a non-zero noise level."
            )

        dt = next_sigma - current_sigma

        sde_step_indices = getattr(batch, "rollout_sde_step_indices", None)
        if (
            sde_type != "ode"
            and sde_step_indices is not None
            and step_index not in sde_step_indices
        ):
            effective_sde_type = "ode"
        else:
            effective_sde_type = sde_type

        # sde/cps: cast to fp32 to match flowGRPO semantics and avoid the
        # 0-dim-fp32 wrapped-scalar promotion demoting log-prob to bf16.
        # ode: keep dtypes unchanged so rollout(ode) stays bit-exact with
        # rollout=False (scheduling_flow_match_euler_discrete.step()).
        # log_prob is computed on the full pre-shard noise buffer so SP ranks
        # produce identical sums — see collect_rollout_log_probs().
        if effective_sde_type == "sde":
            model_output = model_output.float()
            sample = sample.float()
            variance_noise = self._rollout_variance_noise(
                batch, model_output, generator
            )
            full_variance_noise = self.noise_buffer
            std_dev_t = (
                torch.sqrt(
                    current_sigma
                    / (
                        1
                        - torch.where(
                            torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                            self.sigma_max,
                            current_sigma,
                        )
                    )
                )
                * noise_level
            )
            noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
            prev_sample_mean = (
                sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt)
                + model_output
                * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma))
                * dt
            )

            weighted_variance_noise = variance_noise * noise_std_dev
            prev_sample = prev_sample_mean + weighted_variance_noise
            log_prob_no_const_val = -((full_variance_noise * noise_std_dev) ** 2)

        elif effective_sde_type == "cps":
            model_output = model_output.float()
            sample = sample.float()
            variance_noise = self._rollout_variance_noise(
                batch, model_output, generator
            )
            full_variance_noise = self.noise_buffer
            std_dev_t = next_sigma * math.sin(noise_level * math.pi / 2)
            noise_std_dev = std_dev_t
            pred_original_sample = sample - current_sigma * model_output
            noise_estimate = sample + model_output * (1 - current_sigma)
            prev_sample_mean = pred_original_sample * (
                1 - next_sigma
            ) + noise_estimate * torch.sqrt(next_sigma**2 - std_dev_t**2)

            weighted_variance_noise = variance_noise * noise_std_dev
            prev_sample = prev_sample_mean + weighted_variance_noise
            log_prob_no_const_val = -((full_variance_noise * noise_std_dev) ** 2)

        elif effective_sde_type == "ode":
            prev_sample = sample + dt * model_output
            prev_sample_mean = prev_sample
            variance_noise = torch.zeros_like(model_output)
            noise_std_dev = torch.zeros(
                (), device=model_output.device, dtype=model_output.dtype
            )
            log_prob_no_const_val = torch.zeros(
                self.latents_shape,
                device=model_output.device,
                dtype=torch.float32,
            )
            # Only enforce the "no full log-prob with ODE" constraint when the
            # user explicitly chose ODE globally.
            if sde_type == "ode":
                assert log_prob_no_const, (
                    "p_ode is always 0, true log_prob is meaningless, set rollout_log_prob_no_const to True to enable log_prob computation"
                )

        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        reduce_dims = list(range(1, len(log_prob_no_const_val.shape)))
        local_elem_count = log_prob_no_const_val.new_full(
            (log_prob_no_const_val.shape[0],),
            float(math.prod(log_prob_no_const_val.shape[1:])),
        )

        if log_prob_no_const or effective_sde_type == "ode":
            log_prob_local_sum = log_prob_no_const_val.sum(dim=reduce_dims)
        else:
            log_prob_local_sum = (
                log_prob_no_const_val / (2 * (noise_std_dev**2))
                - torch.log(noise_std_dev)
                - _LOG_SQRT_2PI
            ).sum(dim=list(range(1, len(log_prob_no_const_val.shape))))

        if debug_mode:
            self.append_local_rollout_debug_tensors(
                variance_noise=variance_noise,
                prev_sample_mean=prev_sample_mean,
                noise_std_dev=noise_std_dev,
                model_output=model_output,
            )

        self.append_local_rollout_log_probs(log_prob_local_sum, local_elem_count)

        return prev_sample

    def append_local_rollout_log_probs(
        self,
        log_prob_sum: torch.Tensor,
        log_prob_count: torch.Tensor,
    ) -> None:
        self.local_log_prob_sum.append(log_prob_sum)
        self.local_log_prob_count.append(log_prob_count)

    def consume_local_rollout_log_probs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, T]: batch dim 0, denoising step dim 1
        values_sum = torch.stack(self.local_log_prob_sum, dim=1)
        values_count = torch.stack(self.local_log_prob_count, dim=1)
        self.local_log_prob_sum = []
        self.local_log_prob_count = []
        return values_sum, values_count

    def collect_rollout_log_probs(self) -> torch.Tensor | None:
        """Per-step sums are already computed on the full pre-shard noise
        buffer inside flow_sde_sampling, so every SP rank holds identical
        values here and no all-reduce is needed."""

        trajectory_log_prob_sum, trajectory_log_prob_count = (
            self.consume_local_rollout_log_probs()
        )
        rollout_log_probs_tensor = trajectory_log_prob_sum / trajectory_log_prob_count
        return rollout_log_probs_tensor.cpu()


class SchedulerRLMixin:
    @staticmethod
    def _get_rollout_session_data(batch) -> RolloutSessionData:
        """Return the RolloutSessionData attached to *batch*, or raise if not prepared."""
        rollout_session_data = getattr(batch, "_rollout_session_data", None)
        if rollout_session_data is None:
            raise RuntimeError("prepare_rollout() not called before rollout")
        return rollout_session_data

    def release_rollout_resources(self, batch) -> None:
        """Release rollout-owned resources. Call when denoising ends or before a new rollout."""
        batch._rollout_session_data = None

    def prepare_rollout(self, batch: Req, pipeline_config: Any = None) -> None:
        """Enable rollout and set SDE/CPS params. Call once before the denoising loop."""
        if get_sp_world_size() > 1 and pipeline_config is None:
            raise RuntimeError(
                "SP rollout requires pipeline_config to be passed to prepare_rollout()."
            )
        batch._rollout_session_data = RolloutSessionData(
            pipeline_config=pipeline_config,
            sigma_max=self.sigmas[min(1, len(self.sigmas) - 1)].item(),
            latents_shape=(
                tuple(batch.latents.shape) if batch.latents is not None else None
            ),
        )

    def already_prepared_rollout(self, batch) -> bool:
        return getattr(batch, "_rollout_session_data", None) is not None
