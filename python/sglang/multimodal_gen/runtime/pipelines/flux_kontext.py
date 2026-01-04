# SPDX-License-Identifier: Apache-2.0
"""
Flux Kontext Pipeline for image editing tasks.

This pipeline implements support for the FLUX.1-Kontext-dev model which enables
instruction-based image editing. The key difference from the standard Flux pipeline
is that it takes an input image and an editing instruction, then generates an edited image.

Architecture difference:
- Standard Flux (T2I): hidden_states = [B, L, D] (noise latent only)
- Flux Kontext (I2I): hidden_states = [B, 2L, D] (noise latent + image latent concatenated)

The transformer outputs predictions for both, but only the noise portion is used.
"""

from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
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


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate the shift value for the scheduler based on sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    """Prepare the mu parameter for the scheduler based on image dimensions."""
    height = batch.height
    width = batch.width
    vae_scale_factor = (
        server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
    )
    image_seq_len = (int(height) // vae_scale_factor) * (int(width) // vae_scale_factor)

    mu = calculate_shift(
        image_seq_len,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    )
    return "mu", mu


class FluxKontextPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Pipeline for FLUX.1-Kontext-dev image editing.

    This pipeline enables instruction-based image editing using the Kontext model.
    It takes an input image and a text instruction, then generates an edited image.

    The key architectural difference from standard Flux is:
    - Input: concatenation of noise latents and encoded image latents [B, 2L, D]
    - Output: only the noise portion of the prediction is used for denoising

    Example usage:
        sglang serve --model-path black-forest-labs/FLUX.1-Kontext-dev --port 3000

        curl http://127.0.0.1:3000/v1/images/edits \\
          -H "Content-Type: multipart/form-data" \\
          -F image=@input.png \\
          -F prompt="Add a hat to the cat" \\
          -F n=1 \\
          -F size="1024x1024"
    """

    pipeline_name = "FluxKontextPipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # Input validation stage
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(
                vae_image_processor=VaeImageProcessor(
                    vae_scale_factor=server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
                    * 2
                )
            ),
        )

        # Text encoding stage (same as standard Flux)
        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )

        # Image VAE encoding stage (encode input image to latent space)
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

        # Conditioning stage
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        # Timestep preparation stage
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )

        # Latent preparation stage (prepare noise latents)
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # Denoising stage (main diffusion loop)
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Decoding stage (decode latents to image)
        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


EntryClass = FluxKontextPipeline
