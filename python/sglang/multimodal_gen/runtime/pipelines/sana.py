# SPDX-License-Identifier: Apache-2.0
#
# SANA text-to-image pipeline.
#
# Stage order matches Flux (InputValidation -> TextEncoding -> TimestepPrep ->
# LatentPrep -> Denoising -> Decoding) rather than the add_standard_t2i_stages
# helper (which puts LatentPrep before TimestepPrep). Both orderings are
# functionally equivalent since these stages are independent.
#
# SANA uses a single text encoder (Gemma2), so only one text_encoder + tokenizer
# pair is registered — unlike Flux which has text_encoder + text_encoder_2.
# The pipeline_name must match the _class_name in HF model_index.json.

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SanaPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "SanaPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())

        self.add_stage(
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
            "prompt_encoding_stage_primary",
        )

        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = SanaPipeline
