# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion XL text-to-image pipeline built from standard stages."""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class StableDiffusionXLPipeline(ComposedPipelineBase):
    pipeline_name = "StableDiffusionXLPipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "unet",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_stage(
            TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
            "prompt_encoding_stage_primary",
        )
        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage(transformer_key="unet")
        self.add_standard_denoising_stage(
            transformer_key="unet", transformer_2_key=None
        )
        self.add_standard_decoding_stage()


EntryClass = StableDiffusionXLPipeline
