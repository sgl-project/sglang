# SPDX-License-Identifier: Apache-2.0
"""
Flux Kontext Pipeline for image editing tasks.

This pipeline extends FluxPipeline to support FLUX.1-Kontext-dev model for
instruction-based image editing. It takes an input image and an editing instruction,
then generates an edited image.

Architecture difference from standard Flux:
- Standard Flux (T2I): hidden_states = [B, L, D] (noise latent only)
- Flux Kontext (I2I): hidden_states = [B, 2L, D] (noise latent + image latent concatenated)

The transformer outputs predictions for both, but only the noise portion is used.
"""

from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.runtime.pipelines.flux import FluxPipeline, prepare_mu
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


class FluxKontextPipeline(FluxPipeline):
    """
    Pipeline for FLUX.1-Kontext-dev image editing.

    Extends FluxPipeline with an additional ImageVAEEncodingStage to encode
    input images into latent space for instruction-based editing.

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

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with image encoding for Kontext."""
        vae_scale_factor = (
            server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
        )
        vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

        # Input validation with image processor for Kontext
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(vae_image_processor=vae_image_processor),
        )

        # Text encoding (same as standard Flux)
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

        # Image VAE encoding (Kontext-specific: encode input image to latent space)
        self.add_stage(
            stage_name="image_encoding_stage_primary",
            stage=ImageVAEEncodingStage(
                vae_image_processor=vae_image_processor,
                vae=self.get_module("vae"),
            ),
        )

        # Remaining stages same as standard Flux
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
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )


EntryClass = FluxKontextPipeline
