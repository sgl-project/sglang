# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.anima import (
    AnimaTextConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)


class AnimaPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "AnimaModularPipeline"
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "t5_tokenizer",
        "text_conditioner",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args):
        self.add_stage(InputValidationStage())
        self.add_stage(
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_stage(
            AnimaTextConditioningStage(
                self.get_module("text_conditioner"), self.get_module("t5_tokenizer")
            )
        )
        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = AnimaPipeline
