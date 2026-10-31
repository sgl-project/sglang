# SPDX-License-Identifier: Apache-2.0
"""Encode the images Fibo 2.0 edits into per-image source latents.

The generic `ImageVAEEncodingStage` concatenates every reference image onto the
sequence dimension and stores the result in `batch.image_latent`, which is
FLUX.2's convention. Fibo instead conditions on each reference image as its own
spatial tensor: the images may have different sizes, and the transformer
patchifies each of them itself.

So this stage follows the in-tree LLaDA edit model
(`model_specific_stages/llada_image/source.py`) and writes the list to
`batch.source_latents`, which the pipeline config hands to the transformer:

* encode each image on its own,
* patchify to reach the layout the VAE's BatchNorm statistics describe,
* normalize with those statistics,
* unpatchify back to 32 spatial channels, keeping one tensor per image.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    _patchify_latents,
    _unpatchify_latents,
    flux2_vae_bn_stats,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


class BriaFibo2SourceLatentsStage(PipelineStage):
    """VAE-encode the reference images into `batch.source_latents`."""

    def __init__(self, vae, image_processor=None) -> None:
        super().__init__()
        self.vae = vae
        self.image_processor = image_processor

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        del server_args
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "vae")]

    def _preprocess(self, image) -> torch.Tensor:
        """Pixels in [-1, 1] at the generation size."""
        if self.image_processor is not None:
            return self.image_processor.preprocess(image)
        raise ValueError(
            "BriaFibo2SourceLatentsStage needs a VAE image processor to turn the "
            "reference image into tensors."
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        condition_image = batch.condition_image
        if condition_image is None:
            batch.source_latents = None
            return batch

        images = (
            condition_image
            if isinstance(condition_image, (list, tuple))
            else [condition_image]
        )

        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            self.vae = vae
            device, dtype = _module_device_dtype(vae)

            source_latents: list[torch.Tensor] = []
            for image in images:
                pixels = self._preprocess(image).to(device=device, dtype=dtype)

                posterior = vae.encode(pixels)
                # diffusers' AutoencoderKL wraps the distribution
                if hasattr(posterior, "latent_dist"):
                    posterior = posterior.latent_dist
                latents = posterior.mode()

                # The BatchNorm statistics are per patchified channel, so
                # normalize there and come back to 32 spatial channels, which is
                # what the transformer patchifies again itself.
                patchified = _patchify_latents(latents.float())
                mean, std = flux2_vae_bn_stats(vae, patchified)
                normalized = (patchified - mean) / std
                source_latents.append(_unpatchify_latents(normalized))

        batch.source_latents = source_latents
        return batch


__all__ = ["BriaFibo2SourceLatentsStage"]
