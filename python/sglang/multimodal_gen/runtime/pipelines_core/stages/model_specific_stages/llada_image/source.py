# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs.flux import _patchify_latents
from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    flux2_vae_bn_stats,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


class LLaDAImageSourceImageConditioningStage(PipelineStage):
    def __init__(self, sigvq, vae, image_processor) -> None:
        super().__init__()
        self.sigvq = sigvq
        self.vae = vae
        self.image_processor = image_processor

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        del server_args
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "sigvq"),
            ComponentUse(stage_name, "vae"),
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        if batch.condition_image is None:
            batch.source_latents = None
            return batch

        source_images = (
            batch.condition_image
            if isinstance(batch.condition_image, list)
            else [batch.condition_image]
        )
        if len(source_images) != 1:
            raise ValueError("LLaDA-Image editing requires exactly one source image")

        image = self.image_processor.preprocess(
            source_images[0],
            height=batch.height,
            width=batch.width,
        )
        if batch.batch_size > 1:
            image = image.repeat(batch.batch_size, 1, 1, 1)

        sigvq_pixels = F.interpolate(
            image.float(),
            size=(batch.height // 2, batch.width // 2),
            mode="bilinear",
            align_corners=False,
        )
        with self.use_declared_component(
            component_name="sigvq", module=self.sigvq
        ) as sigvq:
            assert sigvq is not None
            sigvq_device, sigvq_dtype = _module_device_dtype(sigvq)
            semantic_features = sigvq(
                sigvq_pixels.to(device=sigvq_device, dtype=sigvq_dtype)
            )

        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            vae_device, vae_dtype = _module_device_dtype(vae)
            posterior = vae.encode(image.to(device=vae_device, dtype=vae_dtype))
            source_latents = _patchify_latents(posterior.mode())
            mean, std = flux2_vae_bn_stats(vae, source_latents)
            source_latents = (source_latents - mean) / std

        batch.image_embeds = list(semantic_features.unbind(dim=0))
        batch.source_latents = [
            latent.unsqueeze(1) for latent in source_latents.unbind(dim=0)
        ]
        return batch
