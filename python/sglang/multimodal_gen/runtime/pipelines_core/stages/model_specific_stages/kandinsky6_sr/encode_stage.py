# SPDX-License-Identifier: Apache-2.0
"""Whole-video KVAE encode of the latent-upscaler (LU) path of Kandinsky 6 video SR."""

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_LR_LATENT_KEY,
    SR_TILING_SCALE_KEY,
    SR_VIDEO_KEY,
    uses_latent_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    encode_video_to_lr_latent,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision


class Kandinsky6SREncodeStage(PipelineStage):
    """Encodes the whole clip to the raw LR latent (only when the LU path runs).

    The pixel path encodes per tile later, so this stage is then a no-op and never
    touches (or loads) the VAE.
    """

    def __init__(self, vae, latent_upscaler=None) -> None:
        super().__init__()
        self.vae = vae
        # Only consulted to pick the path (does the bank have this scale?).
        self.latent_upscaler = latent_upscaler

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                phase="encode_video",
                target_dtype=vae_dtype,
                start_at_stage_entry=False,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        tiling_scale = batch.extra[SR_TILING_SCALE_KEY]
        if not uses_latent_path(self.latent_upscaler, tiling_scale):
            return batch
        video = batch.extra.pop(SR_VIDEO_KEY)
        with self.use_declared_component(
            component_name="vae", module=self.vae, phase="encode_video"
        ) as vae:
            assert vae is not None
            self.vae = vae
            lr_latent = encode_video_to_lr_latent(
                video, vae, device=get_local_torch_device()
            )
        batch.extra[SR_LR_LATENT_KEY] = lr_latent.cpu()
        return batch
