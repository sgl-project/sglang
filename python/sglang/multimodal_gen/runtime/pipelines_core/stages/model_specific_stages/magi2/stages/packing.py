# SPDX-License-Identifier: Apache-2.0
"""Assemble the packed sequence the DiTs consume."""

from __future__ import annotations

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Magi2PackingStage(PipelineStage):
    """Write the row layout and 9-D coordinates for the next denoise pass."""

    def __init__(
        self,
        *,
        grid_key: str,
        conditions_on_images: bool = False,
        refiner_only: bool = False,
    ) -> None:
        super().__init__()
        self.grid_key = grid_key
        self.conditions_on_images = conditions_on_images
        self.refiner_only = refiner_only

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self.refiner_only and not batch.extra["magi2_enable_refiner"]:
            return batch
        device = get_local_torch_device()
        grid = batch.extra[self.grid_key]
        frames, height, width = grid

        audio_tokens = (
            batch.audio_latents.shape[0] if batch.audio_latents is not None else 0
        )

        # Preview only: the refiner's sequence carries no image segment.
        ref_counts = (
            batch.extra["magi2_ref_patch_counts"] if self.conditions_on_images else []
        )
        ref_latent_hw = (
            batch.extra["magi2_ref_latent_hw"] if self.conditions_on_images else []
        )

        # One layout per branch: each text is trimmed to its own token count.
        branches = {"magi2_layout": batch.prompt_embeds[0].shape[0]}
        if batch.negative_prompt_embeds:
            branches["magi2_layout_uncond"] = batch.negative_prompt_embeds[0].shape[0]

        for key, text_tokens in branches.items():
            batch.extra[key] = packed_sequence.build_layout(
                video_latent_thw=grid,
                audio_tokens=audio_tokens,
                text_tokens=text_tokens,
                device=device,
                ref_patch_counts=ref_counts,
            )
            batch.extra[key.replace("layout", "coords")] = packed_sequence.build_coords(
                video_latent_shape=grid,
                audio_tokens=audio_tokens,
                text_tokens=text_tokens,
                device=device,
                ref_latent_hw=ref_latent_hw,
                ref_patch_counts=ref_counts,
            )
        return batch
