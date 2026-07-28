# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationSpec,
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class RealtimeChunkLatentPreparationStage(LatentPreparationStage):
    """Prepare one realtime causal DiT chunk from the encoded condition shape."""

    def get_forward_latent_num_frames(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> int:
        return int(
            batch.realtime_chunk_size
            or self.transformer.config.arch_config.num_frames_per_block
        )

    def get_latent_preparation_spec(
        self,
        batch: Req,
        server_args: ServerArgs,
        batch_size: int,
        num_frames: int,
        device: torch.device | str,
    ) -> LatentPreparationSpec:
        condition_latent = batch.image_latent
        assert condition_latent is not None, (
            "Realtime chunk latent preparation requires image_latent. "
            "Ensure the condition VAE encoding stage runs before this stage."
        )
        return LatentPreparationSpec(
            shape=(
                condition_latent.shape[0],
                self.transformer.config.arch_config.out_channels,
                num_frames,
                condition_latent.shape[3],
                condition_latent.shape[4],
            ),
            dtype=condition_latent.dtype,
            device=device,
            prepare_latent_ids=False,
            pack_latents=False,
        )

    def should_scale_initial_noise(self, batch: Req, server_args: ServerArgs) -> bool:
        return False

    def requires_batch_height_width(self, batch: Req, server_args: ServerArgs) -> bool:
        return False

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result
