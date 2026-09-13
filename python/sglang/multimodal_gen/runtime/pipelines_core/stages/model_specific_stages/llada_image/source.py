# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

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

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("LLaDA-Image source VAE latents must have even dimensions")
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    @staticmethod
    def _normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
        if not hasattr(vae, "bn"):
            raise ValueError("LLaDA-Image editing requires the Flux2 VAE BN state")
        vae_config = getattr(vae.config, "arch_config", vae.config)
        eps = vae_config.batch_norm_eps
        latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
        latent_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + eps).to(latents)
        return (latents - latent_mean) / latent_std

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
            resize_mode="crop",
        )
        if image.shape[0] == 1 and batch.batch_size > 1:
            image = image.repeat(batch.batch_size, 1, 1, 1)
        if image.shape[0] != batch.batch_size:
            raise ValueError(
                "LLaDA-Image source image batch must contain one image or match "
                f"the effective batch size {batch.batch_size}, got {image.shape[0]}"
            )

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
            patch_size = sigvq.config.patch_size
            pad_height = (-sigvq_pixels.shape[-2]) % patch_size
            pad_width = (-sigvq_pixels.shape[-1]) % patch_size
            if pad_height or pad_width:
                sigvq_pixels = F.pad(
                    sigvq_pixels,
                    (0, pad_width, 0, pad_height),
                    mode="replicate",
                )
            sigvq_device, sigvq_dtype = _module_device_dtype(sigvq)
            semantic_features = sigvq(
                sigvq_pixels.to(device=sigvq_device, dtype=sigvq_dtype)
            ).semantic_features

        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            vae_device, vae_dtype = _module_device_dtype(vae)
            posterior = vae.encode(image.to(device=vae_device, dtype=vae_dtype))
            latent_dist = getattr(posterior, "latent_dist", posterior)
            source_latents = self._patchify_latents(latent_dist.mode())
            source_latents = self._normalize_latents(source_latents, vae)

        batch.image_embeds = list(semantic_features.unbind(dim=0))
        batch.source_latents = [
            latent.unsqueeze(1) for latent in source_latents.unbind(dim=0)
        ]
        return batch
