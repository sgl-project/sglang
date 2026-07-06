# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
    WanProgressiveDenoisingStage,
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
        shift = server_args.pipeline_config.flow_shift
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()
        # Serving keeps UniPC; requests with rollout=True bind a first-order
        # flow-match Euler scheduler, which the RL SDE/log-prob path requires.
        self.add_stage(
            TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                rollout_scheduler=FlowMatchEulerDiscreteScheduler(
                    shift=1.0 if shift is None else shift
                ),
            )
        )
        self.add_progressive_denoising_stage(WanProgressiveDenoisingStage)
        self.add_standard_decoding_stage()


EntryClass = WanPipeline
