# SPDX-License-Identifier: Apache-2.0
"""Joint audio-video denoising for MAGI-2's preview and refiner passes."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    guidance as magi2_guidance,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Magi2DenoisingStage(DenoisingStage):
    """Run one denoise loop over the packed video+audio sequence.

    ``forward`` is model-owned: the two modalities carry separate guidance scales
    and separate multistep scheduler state.
    """

    def __init__(
        self,
        *,
        transformer,
        pipeline=None,
        guidance_key: str = "",
        refiner_only: bool = False,
    ) -> None:
        # scheduler=None: the schedule is per-modality, built in the preparation stage.
        super().__init__(transformer=transformer, scheduler=None, pipeline=pipeline)
        self.guidance_key = guidance_key
        self.refiner_only = refiner_only

    def _owns_compile_warmup_lifecycle(self) -> bool:
        # The base tests whether forward is inherited, so not overriding this would
        # silently disable offload-during-compile rather than raise.
        return True

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def _scales(self, batch: Req) -> tuple[float, float, float | None]:
        params = batch.sampling_params
        if self.guidance_key == "refiner":
            return (
                params.refiner_guidance_scale,
                params.refiner_audio_guidance_scale,
                None,
            )
        return (
            params.guidance_scale,
            params.audio_guidance_scale,
            params.skimmed_guidance_scale if params.use_skimmed_guidance else None,
        )

    def _predict(
        self,
        *,
        video: torch.Tensor,
        audio: torch.Tensor | None,
        text: torch.Tensor,
        layout,
        coords: torch.Tensor,
        timestep: torch.Tensor,
        ref_patches: torch.Tensor | None = None,
        ref_special: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        extra = {}
        # Only the preview accepts ref images; the refiner's allowlist raises on them.
        if ref_patches is not None:
            extra = {"ref_patches": ref_patches, "ref_special": ref_special}
        return self.transformer(
            video_latents=video,
            audio_latents=audio,
            text_embeds=text,
            layout=layout,
            coords=coords,
            timestep=timestep,
            **extra,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self.refiner_only and not batch.extra["magi2_enable_refiner"]:
            return batch

        device = get_local_torch_device()
        video_scale, audio_scale, skimmed_scale = self._scales(batch)
        use_guidance = server_args.pipeline_config.should_use_guidance

        layout = batch.extra["magi2_layout"]
        coords = batch.extra["magi2_coords"]
        prompt = batch.prompt_embeds[0]
        negative = (
            batch.negative_prompt_embeds[0]
            if use_guidance and batch.negative_prompt_embeds
            else None
        )
        uncond_layout = batch.extra.get("magi2_layout_uncond", layout)
        uncond_coords = batch.extra.get("magi2_coords_uncond", coords)

        video = batch.latents
        audio = batch.audio_latents
        scheduler = batch.scheduler

        ref_patches = (
            batch.extra["magi2_ref_patches"] if layout.ref_patch_index.numel() else None
        )
        ref_special = (
            batch.extra["magi2_ref_special"] if ref_patches is not None else None
        )

        for step, timestep in enumerate(batch.timesteps):
            # Per token, not a scalar: text and ref-image rows must read zero.
            step_t = timestep.to(device)
            t = packed_sequence.build_timesteps(
                layout=layout, video_t=step_t, audio_t=step_t
            )

            cond_video, cond_audio = self._predict(
                video=video,
                audio=audio,
                text=prompt,
                layout=layout,
                coords=coords,
                timestep=t,
                ref_patches=ref_patches,
                ref_special=ref_special,
            )

            if negative is not None:
                uncond_video, uncond_audio = self._predict(
                    video=video,
                    audio=audio,
                    text=negative,
                    layout=uncond_layout,
                    coords=uncond_coords,
                    timestep=packed_sequence.build_timesteps(
                        layout=uncond_layout, video_t=step_t, audio_t=step_t
                    ),
                    ref_patches=ref_patches,
                    ref_special=ref_special,
                )
                cond_video = magi2_guidance.apply_guidance(
                    latent=video,
                    cond=cond_video,
                    uncond=uncond_video,
                    guidance_scale=video_scale,
                    skimmed_scale=skimmed_scale,
                )
                if cond_audio is not None and uncond_audio is not None:
                    cond_audio = magi2_guidance.apply_guidance(
                        latent=audio,
                        cond=cond_audio,
                        uncond=uncond_audio,
                        guidance_scale=audio_scale,
                        skimmed_scale=skimmed_scale,
                    )

            video = scheduler.step(cond_video, timestep, video, return_dict=False)[0]
            if cond_audio is not None:
                audio = batch.extra["magi2_audio_scheduler"].step(
                    cond_audio, timestep, audio, return_dict=False
                )[0]

        batch.latents = video
        batch.audio_latents = audio
        return batch
