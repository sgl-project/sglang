# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.configs.sample.magi2 import (
    MAGI2_CLIP_SECONDS,
    MAGI2_REFINER_FPS,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# Both passes use this, so the handoff's 2*T-1 upsample lands exactly.
MAGI2_BASE_FRAMES = round(MAGI2_CLIP_SECONDS * MAGI2_REFINER_FPS)


def latent_grid(
    *, height: int, width: int, frames: int, vae_stride: tuple[int, int, int]
) -> tuple[int, int, int]:
    # The temporal VAE is causal, so the first frame is its own latent.
    stride_t, stride_h, stride_w = vae_stride
    return (
        (frames - 1) // stride_t + 1,
        height // stride_h,
        width // stride_w,
    )


class Magi2InputStage(PipelineStage):
    """Its own stage because the packing stage runs once per denoise pass and both instances need these numbers."""

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_not_empty)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        config = server_args.pipeline_config
        params = batch.sampling_params

        # At preview size the refiner has nothing to do, even if its weights loaded.
        preview_size = (params.preview_height, params.preview_width)
        enable_refiner = (
            config.enable_refiner
            and (
                batch.height,
                batch.width,
            )
            != preview_size
        )

        batch.extra["magi2_enable_refiner"] = enable_refiner
        batch.extra["magi2_preview_grid"] = latent_grid(
            height=params.preview_height,
            width=params.preview_width,
            frames=MAGI2_BASE_FRAMES,
            vae_stride=config.dit_config.arch_config.vae_stride,
        )
        if enable_refiner:
            batch.extra["magi2_refiner_grid"] = latent_grid(
                height=batch.height,
                width=batch.width,
                frames=MAGI2_BASE_FRAMES,
                vae_stride=config.refiner_dit_config.arch_config.vae_stride,
            )
        return batch
