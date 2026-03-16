# SPDX-License-Identifier: Apache-2.0
"""
Helios video diffusion pipeline implementation.

This module contains an implementation of the Helios video diffusion pipeline
using the modular pipeline architecture. Phase 1: T2V only.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_decoding import (
    HeliosDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_denoising import (
    HeliosChunkedDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class HeliosPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Helios video diffusion pipeline with LoRA support.

    Implements the Helios T2V pipeline with chunked denoising,
    multi-term memory history, and CFG Zero Star guidance.
    """

    pipeline_name = "HeliosPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        # Use the scheduler loaded from model's scheduler_config.json as-is.
        # It contains critical config: use_dynamic_shifting=true,
        # time_shift_type="exponential", etc.
        scheduler = self.modules.get("scheduler")
        if scheduler is not None and server_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(server_args.pipeline_config.flow_shift)

        # Configure scheduler for Stage 2/3 if enabled
        pipeline_config = server_args.pipeline_config
        if scheduler is not None and pipeline_config.is_enable_stage2:
            scheduler.config.stages = pipeline_config.pyramid_num_stages
            scheduler.config.scheduler_type = pipeline_config.scheduler_type
            scheduler.config.gamma = pipeline_config.gamma
            scheduler.init_sigmas_for_each_stage()

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()
        # Skip standard timestep preparation — the Helios denoising stage
        # handles scheduler.set_timesteps internally per-chunk with mu.
        self.add_stage(
            HeliosChunkedDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.modules["scheduler"],
            ),
            "helios_chunked_denoising_stage",
        )
        # Helios-specific decoding: decode each chunk's latents separately
        # to avoid temporal artifacts from Wan VAE causal convolutions
        self.add_stage(
            HeliosDecodingStage(vae=self.get_module("vae"), pipeline=self),
            "helios_decoding_stage",
        )


class HeliosPyramidPipeline(HeliosPipeline):
    """Helios pyramid SR pipeline (used by Helios-Mid and Helios-Distilled)."""

    pipeline_name = "HeliosPyramidPipeline"


EntryClass = [HeliosPipeline, HeliosPyramidPipeline]
