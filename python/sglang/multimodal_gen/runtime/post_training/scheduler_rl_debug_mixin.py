# SPDX-License-Identifier: Apache-2.0
"""Debug tensor helpers for rollout-enabled schedulers."""

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDebugTensors,
    RolloutSessionData,
)


class SchedulerRLDebugMixin:
    @staticmethod
    def _reset_rollout_debug_tensors(rollout_session_data: RolloutSessionData) -> None:
        rollout_session_data.local_variance_noises = []
        rollout_session_data.local_prev_sample_means = []
        rollout_session_data.local_noise_std_devs = []
        rollout_session_data.local_model_outputs = []

    def append_local_rollout_debug_tensors(
        self,
        batch,
        *,
        variance_noise: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        noise_std_dev: torch.Tensor,
        model_output: torch.Tensor,
    ) -> None:
        rollout_session_data = batch._rollout_session_data
        batch_size = variance_noise.shape[0]
        rollout_session_data.local_variance_noises.append(variance_noise)
        rollout_session_data.local_prev_sample_means.append(prev_sample_mean)
        rollout_session_data.local_noise_std_devs.append(noise_std_dev.expand((batch_size, 1)))
        rollout_session_data.local_model_outputs.append(model_output)

    def consume_local_rollout_debug_tensors(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rollout_session_data = batch._rollout_session_data
        variance_noises = torch.stack(rollout_session_data.local_variance_noises, dim=1)
        prev_sample_means = torch.stack(rollout_session_data.local_prev_sample_means, dim=1)
        noise_std_devs = torch.stack(rollout_session_data.local_noise_std_devs, dim=1)
        model_outputs = torch.stack(rollout_session_data.local_model_outputs, dim=1)
        self._reset_rollout_debug_tensors(rollout_session_data)
        return variance_noises, prev_sample_means, noise_std_devs, model_outputs

    def collect_rollout_debug_tensors(
        self, batch: Req
    ) -> RolloutDebugTensors:
        """
        Consume rollout debug tensors and merge for all SP ranks.

        Returns rollout debug tensors with shape [B, T, ...].
        """
        rollout_session_data = batch._rollout_session_data
        variance_noises, prev_sample_means, noise_std_devs, model_outputs = (
            self.consume_local_rollout_debug_tensors(batch)
        )

        if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
            variance_noises = variance_noises.to(get_local_torch_device())
            prev_sample_means = prev_sample_means.to(get_local_torch_device())
            noise_std_devs = noise_std_devs.to(get_local_torch_device())
            model_outputs = model_outputs.to(get_local_torch_device())
            pipeline_config = rollout_session_data.pipeline_config
            bsz, num_steps = variance_noises.shape[0], variance_noises.shape[1]

            # [B, T, ...] -> [B*T, ...]
            variance_noises_packed = variance_noises.contiguous().reshape(
                bsz * num_steps, *variance_noises.shape[2:]
            )
            prev_sample_means_packed = prev_sample_means.contiguous().reshape(
                bsz * num_steps, *prev_sample_means.shape[2:]
            )
            model_outputs_packed = model_outputs.contiguous().reshape(
                bsz * num_steps, *model_outputs.shape[2:]
            )

            # Gather on packed tensors first.
            variance_noises_packed = pipeline_config.gather_latents_for_sp(
                variance_noises_packed
            )
            prev_sample_means_packed = pipeline_config.gather_latents_for_sp(
                prev_sample_means_packed
            )
            model_outputs_packed = pipeline_config.gather_latents_for_sp(
                model_outputs_packed
            )

            # Unpack back to [B, T, ...].
            variance_noises = variance_noises_packed.reshape(
                bsz, num_steps, *variance_noises_packed.shape[1:]
            )
            prev_sample_means = prev_sample_means_packed.reshape(
                bsz, num_steps, *prev_sample_means_packed.shape[1:]
            )
            model_outputs = model_outputs_packed.reshape(
                bsz, num_steps, *model_outputs_packed.shape[1:]
            )
            # noise_std_devs is same on every device, not a sharded latent tensor.

        return RolloutDebugTensors(
            rollout_variance_noises=variance_noises.cpu(),
            rollout_prev_sample_means=prev_sample_means.cpu(),
            rollout_noise_std_devs=noise_std_devs.cpu(),
            rollout_model_outputs=model_outputs.cpu(),
        )
