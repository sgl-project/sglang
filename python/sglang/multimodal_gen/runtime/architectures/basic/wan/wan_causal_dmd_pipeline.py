# SPDX-License-Identifier: Apache-2.0
"""
Wan causal DMD pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from sgl_diffusion.runtime.pipelines import ComposedPipelineBase, LoRAPipeline

# isort: off
from sgl_diffusion.runtime.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    CausalDMDDenosingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
)
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.logging_utils import init_logger

# isort: on

logger = init_logger(__name__)


class WanCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=CausalDMDDenosingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


EntryClass = WanCausalDMDPipeline
