# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
    FluxProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = (
        server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
    )
    image_seq_len = (int(height) // (vae_scale_factor * 2)) * (
        int(width) // (vae_scale_factor * 2)
    )

    return "mu", calculate_linear_shift(image_seq_len)


class FluxPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "FluxPipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_standard_t2i_stages(
            text_encoder_key=["text_encoder", "text_encoder_2"],
            tokenizer_key=["tokenizer", "tokenizer_2"],
            text_encoding_stage_name="prompt_encoding_stage_primary",
            prepare_extra_timestep_kwargs=[prepare_mu],
            progressive_denoising_stage_cls=FluxProgressiveDenoisingStage,
        )


EntryClass = FluxPipeline
