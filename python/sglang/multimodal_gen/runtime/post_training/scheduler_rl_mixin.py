# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

import math
from typing import Any, Union

import torch
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import RolloutSessionData
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_debug_mixin import (
    SchedulerRLDebugMixin,
)


class SchedulerRLMixin(SchedulerRLDebugMixin):

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
            latents_shape=tuple(batch.latents.shape) if batch.latents is not None else None,
        )

    def already_prepared_rollout(self, batch) -> bool:
        return getattr(batch, "_rollout_session_data", None) is not None

    def _get_or_create_rollout_noise_buffer(
        self,
        rollout_session_data: RolloutSessionData,
        full_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or create the reusable noise buffer (local or full shape) for rollout."""
        buffer = rollout_session_data.noise_buffer
        if (
            buffer is None
            or buffer.shape != full_shape
            or buffer.dtype != dtype
            or buffer.device != device
        ):
            buffer = torch.empty(full_shape, device=device, dtype=dtype)
            rollout_session_data.noise_buffer = buffer
        return buffer

    def _rollout_variance_noise(
        self,
        batch,
        model_output: torch.FloatTensor,
        generator: Union[torch.Generator, list[torch.Generator]],
    ) -> torch.FloatTensor:
        """Generate variance noise for rollout. If generator is a list, use generator[i] for the i-th batch item."""
        assert generator is not None, "Generator must be provided"

        rollout_session_data = self._get_rollout_session_data(batch)
        device = model_output.device
        dtype = model_output.dtype
        local_shape = tuple(model_output.shape)

        B = local_shape[0]
        if isinstance(generator, torch.Generator):
            assert B == 1, "Generator must be a list if batch size is not 1"
            generator = [generator]
        else:
            assert len(generator) == B, "Generator list must have the same length as batch size"

        buffer = self._get_or_create_rollout_noise_buffer(rollout_session_data, rollout_session_data.latents_shape, device, dtype)
        for i in range(B):
            torch.randn(rollout_session_data.latents_shape, out=buffer[i : i + 1], generator=generator[i])

        sharded_noise, _ = rollout_session_data.pipeline_config.shard_latents_for_sp(batch, buffer)
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
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Flow rollout step for log-prob / sampling (see FlowGRPO-style references).

        ``rollout_sde_type`` (from batch SamplingParams):

        1. ``"sde"``: Standard stochastic differential equation transition (Gaussian).
        2. ``"cps"``: Coupled Particle Sampling.
        3. ``"ode"``: Deterministic ODE step (no diffusion noise).
        """
        rollout_session_data = self._get_rollout_session_data(batch)
        sde_type = batch.rollout_sde_type
        noise_level = float(batch.rollout_noise_level)
        log_prob_no_const = batch.rollout_log_prob_no_const
        debug_mode = bool(getattr(batch, "rollout_debug_mode", False))

        dt = next_sigma - current_sigma
        if sde_type == "sde":
            variance_noise = self._rollout_variance_noise(batch, model_output, generator)
            std_dev_t = torch.sqrt(
                current_sigma /
                (1 - torch.where(torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                                               rollout_session_data.sigma_max, current_sigma))) * noise_level
            noise_std_dev = std_dev_t * torch.sqrt(-1*dt)
            prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt) \
                               + model_output * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma)) * dt

            weighted_variance_noise = variance_noise * noise_std_dev
            prev_sample = prev_sample_mean + weighted_variance_noise
            log_prob_no_const_val = -(weighted_variance_noise ** 2)

        elif sde_type == "cps":
            variance_noise = self._rollout_variance_noise(batch, model_output, generator)
            std_dev_t = next_sigma * math.sin(noise_level * math.pi / 2)
            noise_std_dev = std_dev_t
            pred_original_sample = sample - current_sigma * model_output
            noise_estimate = sample + model_output * (1 - current_sigma)
            prev_sample_mean = pred_original_sample * (1 - next_sigma) + noise_estimate * torch.sqrt(next_sigma**2 - std_dev_t**2)

            weighted_variance_noise = variance_noise * noise_std_dev
            prev_sample = prev_sample_mean + weighted_variance_noise
            log_prob_no_const_val = -(weighted_variance_noise ** 2)

        elif sde_type == "ode":
            prev_sample = sample + dt * model_output
            prev_sample_mean = prev_sample
            variance_noise = torch.zeros_like(model_output)
            noise_std_dev = torch.zeros((), device=model_output.device, dtype=model_output.dtype)
            log_prob_no_const_val = torch.zeros_like(model_output)
            assert log_prob_no_const, "p_ode is always 0, true log_prob is meaningless, set rollout_log_prob_no_const to True to enable log_prob computation"

        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        reduce_dims = list(range(1, len(log_prob_no_const_val.shape)))
        local_elem_count = log_prob_no_const_val.new_full(
            (log_prob_no_const_val.shape[0],),
            float(math.prod(log_prob_no_const_val.shape[1:])),
        )

        if log_prob_no_const:
            log_prob_local_sum = log_prob_no_const_val.sum(dim=reduce_dims)
        else:
            log_prob_local_sum = (
                log_prob_no_const_val / (2 * (noise_std_dev**2))
                - torch.log(noise_std_dev)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi).to(noise_std_dev.device)))
            ).sum(dim=list(range(1, len(log_prob_no_const_val.shape))))

        if debug_mode:
            self.append_local_rollout_debug_tensors(
                batch,
                variance_noise=variance_noise,
                prev_sample_mean=prev_sample_mean,
                noise_std_dev=noise_std_dev,
                model_output=model_output,
            )

        return prev_sample, log_prob_local_sum, local_elem_count

    def append_local_rollout_log_probs(
        self, batch, log_prob_sum: torch.Tensor, log_prob_count: torch.Tensor
    ) -> None:
        rollout_session_data = self._get_rollout_session_data(batch)
        rollout_session_data.local_log_prob_sum.append(log_prob_sum)
        rollout_session_data.local_log_prob_count.append(log_prob_count)

    def consume_local_rollout_log_probs(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_session_data = self._get_rollout_session_data(batch)
        values_sum = torch.stack(rollout_session_data.local_log_prob_sum, dim=-1)
        values_count = torch.stack(rollout_session_data.local_log_prob_count, dim=-1)
        rollout_session_data.local_log_prob_sum = []
        rollout_session_data.local_log_prob_count = []
        return values_sum, values_count

    def collect_rollout_log_probs(self, batch: Req) -> torch.Tensor | None:
        """Consume local rollout log probs and merge for all SP ranks."""

        trajectory_log_prob_sum, trajectory_log_prob_count = (
            self.consume_local_rollout_log_probs(batch)
        )
        if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
            packed = torch.stack(
                [trajectory_log_prob_sum, trajectory_log_prob_count], dim=0
            ).to(
                get_local_torch_device()
            )
            sequence_model_parallel_all_reduce(packed)
            trajectory_log_prob_sum = packed[0]
            trajectory_log_prob_count = packed[1]

        rollout_log_probs_tensor = (
            trajectory_log_prob_sum / trajectory_log_prob_count
        )
        return rollout_log_probs_tensor.cpu()
