# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline, Req
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def compute_empirical_mu(batch: Req, server_args: ServerArgs):
    num_steps = batch.num_inference_steps
    image_seq_len = batch.raw_latent_shape[1]
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return "mu", float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return "mu", float(mu)


class Flux2Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "Flux2Pipeline"

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
            stage_name="input_validation_stage",
            stage=InputValidationStage(
                vae_image_processor=VaeImageProcessor(
                    vae_scale_factor=server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
                    * 2
                ),
            ),
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

        self.add_stage(
            stage_name="image_encoding_stage_primary",
            stage=ImageVAEEncodingStage(
                vae_image_processor=VaeImageProcessor(
                    vae_scale_factor=server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
                    * 2
                ),
                vae=self.get_module("vae"),
            ),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[compute_empirical_mu],
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


EntryClass = Flux2Pipeline
