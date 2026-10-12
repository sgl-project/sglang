# SPDX-License-Identifier: Apache-2.0
"""Decode stage of Kandinsky 6 video SR: denoised latent chunks -> uint8 tiles."""

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_group_coordinator,
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.common import (
    can_install_spatial_shard_parallel_decode,
    has_decode_parallel_world,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_DENOISED_KEY,
    SR_TILES_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    decode_chunks,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision


class Kandinsky6SRDecodeStage(PipelineStage):
    """VAE-decodes every denoised chunk into row-major uint8 tiles (``[3, T, Hb, Wb]``)."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

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
                phase="decode_tiles",
                target_dtype=vae_dtype,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        denoised = batch.extra.pop(SR_DENOISED_KEY)
        device = get_local_torch_device()
        group = None
        if (
            server_args.pipeline_config.vae_config.use_parallel_tiling
            and has_decode_parallel_world()
            and not can_install_spatial_shard_parallel_decode(
                server_args.pipeline_config.vae_config
            )
        ):
            group = get_decode_parallel_group_coordinator()
        with self.use_declared_component(
            component_name="vae", module=self.vae, phase="decode_tiles"
        ) as vae:
            assert vae is not None
            self.vae = vae
            tiles = decode_chunks(
                denoised,
                vae,
                scaling_factor=vae.scaling_factor,
                device=device,
                group=group,
            )
        batch.extra[SR_TILES_KEY] = tiles
        return batch
