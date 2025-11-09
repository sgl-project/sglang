# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Wan video diffusion pipeline with LoRA support.
    """

    pipeline_name = "WanPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        # We use UniPCMScheduler from Wan2.1 official repo, not the one in diffusers.
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift
        )

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
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )


EntryClass = WanPipeline
