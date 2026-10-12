# SPDX-License-Identifier: Apache-2.0
"""Decode audio before video, preserving the reference component order."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Kandinsky6DecodingStage(DecodingStage):
    """Convert video to channel-first for shared VAE decode; attach the decoded audio."""

    def __init__(self, vae, pipeline=None) -> None:
        super().__init__(vae=vae, pipeline=pipeline)

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if batch.latents is None:
            raise ValueError("latents must be available before Kandinsky6 decoding.")
        batch.latents = batch.latents.permute(0, 4, 1, 2, 3).contiguous()
        output_batch = super().forward(batch, server_args)
        output_batch.audio = batch.audio
        output_batch.audio_sample_rate = batch.audio_sample_rate
        return output_batch


class Kandinsky6AudioDecodingStage(PipelineStage):
    """Decode audio through the bundled mel VAE/vocoder into batch.audio."""

    def __init__(self, audio_vae) -> None:
        super().__init__()
        self.audio_vae = audio_vae

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "audio_vae")]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.audio_latents is None:
            return batch

        device = get_local_torch_device()
        with self.use_declared_component(
            component_name="audio_vae", module=self.audio_vae
        ) as audio_vae:
            assert audio_vae is not None
            self.audio_vae = audio_vae

            # Cast to the module's own loaded weight dtype directly -- no
            # autocast / precision-config knob, matching the existing
            # MMAudio-lineage audio decoding stages in this codebase.
            decoder_dtype = next(audio_vae.parameters()).dtype

            # apply checkpoint latent denormalization before the VAE's mel denormalization
            scaling_factor = audio_vae.scaling_factor
            mean_value = audio_vae.mean_value
            latents = (batch.audio_latents / scaling_factor) + mean_value

            # [B, A, D] -> [B, D, A] to match the audio VAE's 1D-conv
            # (channel, length) convention.
            latents = latents.to(device=device, dtype=decoder_dtype).transpose(1, 2)
            waveform = audio_vae.wrapped_decode(latents)  # [B, 1, samples]

        # preserve [B, samples] so the save path selects the audio track for each output
        batch.audio = waveform[:, 0].clamp(-1.0, 1.0).float().cpu()
        batch.audio_sample_rate = int(server_args.pipeline_config.audio_sample_rate)

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        return result
