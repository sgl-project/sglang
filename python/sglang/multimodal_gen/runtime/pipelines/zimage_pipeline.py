# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline, Req
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.zimage import (
    ZImageProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
    image_seq_len = ((int(height) // vae_scale_factor) // 2) * (
        (int(width) // vae_scale_factor) // 2
    )
    return "mu", calculate_linear_shift(image_seq_len)


class ZImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "ZImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_standard_t2i_stages(
            prepare_extra_timestep_kwargs=[prepare_mu],
            progressive_denoising_stage_cls=ZImageProgressiveDenoisingStage,
        )


EntryClass = ZImagePipeline
