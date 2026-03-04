# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Wan causal DMD pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline

# isort: off
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    CausalDMDDenoisingStage,
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# isort: on

logger = init_logger(__name__)


class WanCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanCausalDMDPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()

        self.add_stage(
            CausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_standard_decoding_stage()


EntryClass = WanCausalDMDPipeline
