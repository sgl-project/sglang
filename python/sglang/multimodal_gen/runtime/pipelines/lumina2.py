# SPDX-License-Identifier: Apache-2.0
#
# Lumina-Image-2.0 text-to-image pipeline.
#
# Lumina-2 fits the standard modular T2I shape.
# All model-specific behavior (system prompt, hidden_states[-2] conditioning,
# renorm-CFG, 4D latents) lives in Lumina2PipelineConfig.
#
# pipeline_name must match _class_name in the HF model_index.json.

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


class Lumina2Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "Lumina2Pipeline"

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


EntryClass = Lumina2Pipeline
