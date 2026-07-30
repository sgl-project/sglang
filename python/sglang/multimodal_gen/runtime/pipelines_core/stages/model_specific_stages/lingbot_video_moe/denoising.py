# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.i2v import (
    COND_LATENT_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.image_conditioning import (
    apply_cond_latent,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LingBotVideoDenoisingStage(DenoisingStage):
    """Denoising that re-pins the clean condition latent after every scheduler step."""

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        if batch.extra.get(COND_LATENT_KEY) is not None and self._sp_world_size() > 1:
            raise ValueError(
                "LingBot-Video image conditioning does not support sequence "
                "parallelism yet; run with --ulysses-degree 1 --ring-degree 1."
            )
        batch.latents = apply_cond_latent(batch, batch.latents)
        return super()._prepare_denoising_loop(batch, server_args)

    def post_forward_for_ti2v_task(
        self,
        batch: Req,
        server_args: ServerArgs,
        reserved_frames_mask,
        latents: torch.Tensor,
        z,
    ) -> torch.Tensor:
        return apply_cond_latent(batch, latents)
