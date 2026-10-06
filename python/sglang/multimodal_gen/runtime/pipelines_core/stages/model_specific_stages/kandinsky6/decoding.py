# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 video + audio decoding stages.

Audio decodes BEFORE video (``Kandinsky6AudioDecodingStage`` must run before
``Kandinsky6DecodingStage`` in the pipeline's stage chain), matching the
diffusers reference and FastVideo's own stage order.
"""

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
    """Channel-last [B,T,H,W,C] -> channel-first [B,C,T,H,W], then the
    generic VAE-decode ``DecodingStage`` (matching Kandinsky5DecodingStage's
    own thin-subclass shape in FastVideo). The generic superclass doesn't
    know about audio, so this override also copies ``batch.audio`` /
    ``batch.audio_sample_rate`` (written onto the ``Req`` by
    ``Kandinsky6AudioDecodingStage``, which runs immediately before this
    stage) onto the returned ``OutputBatch`` -- the same first-class slots
    MOVA's combined decoding stage fills in.
    """

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
    """Decode Kandinsky6 audio latents into a waveform.

    A single ``audio_vae`` component bundles the mel-VAE decoder *and* the
    BigVGAN-v2 vocoder (see ``Kandinsky6AudioVAE``) -- there is no separate
    "vocoder" pipeline module. Writes ``batch.audio`` / ``batch.audio_sample_rate``,
    the first-class ``Req``/``OutputBatch`` slots this codebase already uses
    for a synthesized audio track alongside video (see MOVA's combined
    decoding stage, which fills the same fields on ``OutputBatch`` directly).
    """

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

            # Diffusion-latent-level denorm (``audio / scaling_factor +
            # mean_value``), applied BEFORE the VAE's own internal mel
            # data_mean/data_std normalization inside decode(). Read from
            # the loaded module itself -- these values live in the
            # checkpoint's own audio_vae/config.json, not a pipeline-level
            # default.
            scaling_factor = audio_vae.scaling_factor
            mean_value = audio_vae.mean_value
            latents = (batch.audio_latents / scaling_factor) + mean_value

            # [B, A, D] -> [B, D, A] to match the audio VAE's 1D-conv
            # (channel, length) convention.
            latents = latents.to(device=device, dtype=decoder_dtype).transpose(1, 2)
            waveform = audio_vae.wrapped_decode(latents)  # [B, 1, samples]

        # Keep the batch dimension (``waveform`` is [B, 1, samples]; squeeze
        # only the mono-channel dim) rather than collapsing to the first
        # track: for a multi-output request (``num_outputs_per_prompt`` > 1)
        # this is a real per-output batch, and the generic save path's
        # ``select_output_audio`` (entrypoints/utils.py) already knows how to
        # index a [B, samples] tensor per output -- it only no-ops into
        # "reuse the same 1-D waveform for every video" when handed a 1-D
        # tensor, which is what used to happen here.
        batch.audio = waveform[:, 0].clamp(-1.0, 1.0).float().cpu()
        batch.audio_sample_rate = int(server_args.pipeline_config.audio_sample_rate)

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("audio_latents", batch.audio_latents, V.none_or_tensor)
        return result
