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

NUM_TRAIN_TIMESTEPS = 1000


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

    def expand_timestep_before_forward(
        self,
        batch: Req,
        server_args: ServerArgs,
        t_device,
        target_dtype,
        seq_len,
        reserved_frames_mask,
    ):
        # The DiT is trained on the timestep round-tripped through its own dtype as a
        # sigma, quantizing 991 to 992. The scale-back must stay in that dtype:
        # upcasting first lands on 992.1875 and shifts every step.
        if target_dtype in (torch.bfloat16, torch.float16):
            sigma = (t_device.float() / NUM_TRAIN_TIMESTEPS).to(target_dtype)
            t_device = (sigma * NUM_TRAIN_TIMESTEPS).float()
        return super().expand_timestep_before_forward(
            batch,
            server_args,
            t_device,
            target_dtype,
            seq_len,
            reserved_frames_mask,
        )

    def post_forward_for_ti2v_task(
        self,
        batch: Req,
        server_args: ServerArgs,
        reserved_frames_mask,
        latents: torch.Tensor,
        z,
    ) -> torch.Tensor:
        return apply_cond_latent(batch, latents)
