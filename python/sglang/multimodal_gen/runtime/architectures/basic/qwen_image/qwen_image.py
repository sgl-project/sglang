# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan video diffusion pipeline implementation.

This module contains an implementation of the Hunyuan video diffusion pipeline
using the modular pipeline architecture.
"""
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.runtime.pipelines import ComposedPipelineBase, Req
from sglang.multimodal_gen.runtime.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    ImageEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# TODO(will): move PRECISION_TO_TYPE to better place

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
    image_seq_len = (int(height) // vae_scale_factor) * (int(width) // vae_scale_factor)

    mu = calculate_shift(
        image_seq_len,
        # hard code, since scheduler_config is not in PipelineConfig now
        256,
        4096,
        0.5,
        1.15,
    )
    return "mu", mu


class QwenImagePipeline(ComposedPipelineBase):
    pipeline_name = "QwenImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                ],
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


class QwenImageEditPipeline(ComposedPipelineBase):
    pipeline_name = "QwenImageEditPipeline"

    _required_config_modules = [
        "processor",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=ImageEncodingStage(
                image_processor=self.get_module("processor"),
                text_encoder=self.get_module("text_encoder"),
                vae_image_processor=VaeImageProcessor(
                    vae_scale_factor=server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
                    * 2
                ),
                vae=self.get_module("vae"),
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


EntryClass = [QwenImagePipeline, QwenImageEditPipeline]
