# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import PIL
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    qwen_image_postprocess_text,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vision_utils import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ImageEncodingStage(PipelineStage):
    """
    Stage for encoding image prompts into embeddings for diffusion models.

    This stage handles the encoding of image prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(
        self,
        image_processor,
        image_encoder=None,
        text_encoder=None,
    ) -> None:
        """
        Initialize the prompt encoding stage.

        Args:
            text_encoder: An encoder to encode input_ids and pixel values
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def move_to_device(self, device):
        fields = [
            "image_processor",
            "image_encoder",
        ]
        for field in fields:
            processor = getattr(self, field, None)
            if processor and hasattr(processor, "to"):
                setattr(self, field, processor.to(device))

    def encoding_qwen_image_edit(self, outputs, image_inputs):
        # encoder hidden state
        prompt_embeds = qwen_image_postprocess_text(outputs, image_inputs, 64)
        return prompt_embeds

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode the prompt into image encoder hidden states.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with encoded prompt embeddings.
        """

        if batch.condition_image is None:
            return batch
        cuda_device = get_local_torch_device()
        self.move_to_device(cuda_device)

        image = batch.condition_image

        image_processor_kwargs = (
            server_args.pipeline_config.prepare_image_processor_kwargs(batch)
        )

        image_inputs = self.image_processor(
            images=image, return_tensors="pt", **image_processor_kwargs
        ).to(cuda_device)
        if self.image_encoder:
            # if an image encoder is provided
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = self.image_encoder(
                    **image_inputs,
                    **server_args.pipeline_config.image_encoder_extra_args,
                )
                image_embeds = server_args.pipeline_config.postprocess_image(outputs)

            batch.image_embeds.append(image_embeds)
        elif self.text_encoder:
            # if a text encoder is provided, e.g. Qwen-Image-Edit
            # 1. neg prompt embeds
            if batch.prompt:
                prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
                txt = prompt_template_encode.format(batch.negative_prompt)
                neg_image_processor_kwargs = dict(text=[txt], padding=True)
            else:
                neg_image_processor_kwargs = {}

            neg_image_inputs = self.image_processor(
                images=image, return_tensors="pt", **neg_image_processor_kwargs
            ).to(get_local_torch_device())

            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = self.text_encoder(
                    input_ids=image_inputs.input_ids,
                    attention_mask=image_inputs.attention_mask,
                    pixel_values=image_inputs.pixel_values,
                    image_grid_thw=image_inputs.image_grid_thw,
                    output_hidden_states=True,
                )
                neg_outputs = self.text_encoder(
                    input_ids=neg_image_inputs.input_ids,
                    attention_mask=neg_image_inputs.attention_mask,
                    pixel_values=neg_image_inputs.pixel_values,
                    image_grid_thw=neg_image_inputs.image_grid_thw,
                    output_hidden_states=True,
                )
            batch.prompt_embeds.append(
                self.encoding_qwen_image_edit(outputs, image_inputs)
            )

            batch.negative_prompt_embeds.append(
                self.encoding_qwen_image_edit(neg_outputs, neg_image_inputs)
            )

        self.move_to_device("cpu")

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        if batch.debug:
            logger.debug(f"{batch.condition_image=}")
            logger.debug(f"{batch.image_embeds=}")
        result.add_check("pil_image", batch.condition_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        # result.add_check("image_embeds", batch.image_embeds, V.list_of_tensors_dims(3))
        return result


class ImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.

    This stage handles the encoding of pixel representations into the final
    input format (e.g., image_latents).
    """

    def __init__(self, vae: ParallelTiledVAE, **kwargs) -> None:
        super().__init__()
        self.vae: ParallelTiledVAE = vae

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode pixel representations into latent space.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with encoded outputs.
        """

        if batch.condition_image is None:
            return batch

        num_frames = batch.num_frames

        self.vae = self.vae.to(get_local_torch_device())

        image = batch.condition_image
        image = self.preprocess(
            image,
        ).to(get_local_torch_device(), dtype=torch.float32)

        # (B, C, H, W) -> (B, C, 1, H, W)
        image = image.unsqueeze(2)

        if num_frames == 1:
            video_condition = image
        else:
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0],
                        image.shape[1],
                        num_frames - 1,
                        image.shape[3],
                        image.shape[4],
                    ),
                ],
                dim=2,
            )
        video_condition = video_condition.to(
            device=get_local_torch_device(), dtype=torch.float32
        )

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Encode Image
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            # if server_args.vae_sp:
            #     self.vae.enable_parallel()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output: DiagonalGaussianDistribution = self.vae.encode(
                video_condition
            )

        generator = batch.generator
        if generator is None:
            raise ValueError("Generator must be provided")

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()

        latent_condition = self.retrieve_latents(
            encoder_output, generator, sample_mode=sample_mode
        )
        latent_condition = server_args.pipeline_config.postprocess_vae_encode(
            latent_condition, self.vae
        )

        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latent_condition.device,
                dtype=latent_condition.dtype,
                vae=self.vae,
            )
        )

        # apply shift & scale if needed
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(latent_condition.device)

        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latent_condition.device)

        latent_condition -= shift_factor
        latent_condition = latent_condition * scaling_factor

        batch.image_latent = server_args.pipeline_config.postprocess_image_latent(
            latent_condition, batch
        )

        self.maybe_free_model_hooks()

        self.vae.to("cpu")

        return batch

    def retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def preprocess(
        self,
        image: torch.Tensor | PIL.Image.Image,
    ) -> torch.Tensor:

        if isinstance(image, PIL.Image.Image):
            image = pil_to_numpy(image)  # to np
            image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()

        assert batch.condition_image is None or (
            isinstance(batch.condition_image, PIL.Image.Image)
            or isinstance(batch.condition_image, torch.Tensor)
        )
        assert batch.height is not None and isinstance(batch.height, int)
        assert batch.width is not None and isinstance(batch.width, int)
        assert batch.num_frames is not None and isinstance(batch.num_frames, int)

        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        # result.add_check(
        #     "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        # )
        return result
