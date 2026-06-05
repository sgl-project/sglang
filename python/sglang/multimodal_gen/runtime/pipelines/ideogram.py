# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ideogram import (
    Ideogram4DecodingStage,
    Ideogram4DenoisingStage,
    Ideogram4TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Ideogram4Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "Ideogram4Pipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "unconditional_transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_stage_factory(
            RoleType.ENCODER,
            lambda: Ideogram4TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
            ),
            "ideogram4_text_encoding_stage",
        )
        self.add_standard_latent_preparation_stage()
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: Ideogram4DenoisingStage(
                transformer=self.get_module("transformer"),
                unconditional_transformer=self.get_module("unconditional_transformer"),
                pipeline=self,
            ),
            "ideogram4_denoising_stage",
        )
        self.add_stage_factory(
            RoleType.DECODER,
            lambda: Ideogram4DecodingStage(vae=self.get_module("vae")),
            "ideogram4_decoding_stage",
        )


EntryClass = Ideogram4Pipeline
