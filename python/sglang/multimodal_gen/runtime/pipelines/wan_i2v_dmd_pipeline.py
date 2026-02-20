# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# isort: off
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ImageEncodingStage,
    DmdDenoisingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
)

# isort: on
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

logger = init_logger(__name__)


class WanImageToVideoDmdPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanImageToVideoDmdPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "image_encoder",
        "image_processor",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=server_args.pipeline_config.flow_shift
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stages(
            [
                InputValidationStage(),
            ]
        )

        self.add_standard_text_encoding_stage()

        self.add_stage_if(
            self.get_module("image_encoder") is not None
            and self.get_module("image_processor") is not None,
            ImageEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                image_processor=self.get_module("image_processor"),
            ),
        )


        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage()

        self.add_stages(
            [
                ImageVAEEncodingStage(vae=self.get_module("vae")),
                DmdDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    transformer_2=self.get_module("transformer_2"),
                ),
            ]
        )

        self.add_standard_decoding_stage()


EntryClass = WanImageToVideoDmdPipeline
