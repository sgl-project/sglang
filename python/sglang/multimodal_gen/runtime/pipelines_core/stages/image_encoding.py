# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import PIL
import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
    _pack_latents,
    qwen_image_postprocess_text,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vision_utils import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
    resize,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ExecutionMode, ServerArgs
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
        vae_image_processor=None,
    ) -> None:
        """
        Initialize the prompt encoding stage.

        Args:
            text_encoder: An encoder to encode input_ids and pixel values
        """
        super().__init__()
        self.image_processor = image_processor
        self.vae_image_processor = vae_image_processor
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

    @torch.inference_mode()
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

        cuda_device = get_local_torch_device()
        self.move_to_device(cuda_device)

        image = batch.pil_image

        # preprocess the imag_processor
        prompt_image = server_args.pipeline_config.preprocess_image(
            image, self.vae_image_processor
        )

        if batch.prompt and (
            isinstance(server_args.pipeline_config, QwenImageEditPipelineConfig)
            or isinstance(server_args.pipeline_config, QwenImagePipelineConfig)
        ):
            prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
            txt = prompt_template_encode.format(batch.prompt)
            image_processor_kwargs = dict(text=[txt], padding=True)
        else:
            image_processor_kwargs = {}

        image_inputs = self.image_processor(
            images=prompt_image, return_tensors="pt", **image_processor_kwargs
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
                images=prompt_image, return_tensors="pt", **neg_image_processor_kwargs
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
            logger.debug(f"{batch.pil_image=}")
            logger.debug(f"{batch.image_embeds=}")
        result.add_check("pil_image", batch.pil_image, V.not_none)
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
    input format (e.g., latents).
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
        assert batch.pil_image is not None
        if server_args.mode == ExecutionMode.INFERENCE:
            assert batch.pil_image is not None and isinstance(
                batch.pil_image, PIL.Image.Image
            )
            assert batch.height is not None and isinstance(batch.height, int)
            assert batch.width is not None and isinstance(batch.width, int)
            assert batch.num_frames is not None and isinstance(batch.num_frames, int)
            height = batch.height
            width = batch.width
            num_frames = batch.num_frames
        elif server_args.mode == ExecutionMode.PREPROCESS:
            assert batch.pil_image is not None and isinstance(
                batch.pil_image, torch.Tensor
            )
            assert batch.height is not None and isinstance(batch.height, list)
            assert batch.width is not None and isinstance(batch.width, list)
            assert batch.num_frames is not None and isinstance(batch.num_frames, list)
            num_frames = batch.num_frames[0]
            height = batch.height[0]
            width = batch.width[0]

        self.vae = self.vae.to(get_local_torch_device())

        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio

        image = batch.pil_image
        image = self.preprocess(
            image,
            vae_scale_factor=self.vae.spatial_compression_ratio,
            height=height,
            width=width,
        ).to(get_local_torch_device(), dtype=torch.float32)

        # (B, C, H, W) -> (B, C, 1, H, W)
        image = image.unsqueeze(2)

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
            encoder_output = self.vae.encode(video_condition)

        if server_args.mode == ExecutionMode.PREPROCESS:
            latent_condition = encoder_output.mean
        else:
            generator = batch.generator
            if generator is None:
                raise ValueError("Generator must be provided")
            latent_condition = self.retrieve_latents(encoder_output, generator)

        # Apply shifting if needed
        if hasattr(self.vae, "shift_factor") and self.vae.shift_factor is not None:
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype
                )
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype
            )
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        if server_args.mode == ExecutionMode.PREPROCESS:
            batch.image_latent = latent_condition
        else:
            if isinstance(server_args.pipeline_config, QwenImageEditPipelineConfig):
                batch_size = batch.batch_size
                if (
                    batch_size > latent_condition.shape[0]
                    and batch_size % latent_condition.shape[0] == 0
                ):
                    # expand init_latents for batch_size
                    additional_image_per_prompt = (
                        batch_size // latent_condition.shape[0]
                    )
                    image_latents = torch.cat(
                        [latent_condition] * additional_image_per_prompt, dim=0
                    )
                elif (
                    batch_size > latent_condition.shape[0]
                    and batch_size % latent_condition.shape[0] != 0
                ):
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {latent_condition.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([latent_condition], dim=0)
                image_latent_height, image_latent_width = image_latents.shape[3:]
                num_channels_latents = (
                    self.server_args.pipeline_config.dit_config.arch_config.in_channels
                    // 4
                )
                image_latents = _pack_latents(
                    image_latents,
                    batch_size,
                    num_channels_latents,
                    image_latent_height,
                    image_latent_width,
                )
            else:
                mask_lat_size = torch.ones(
                    1, 1, num_frames, latent_height, latent_width
                )
                mask_lat_size[:, :, list(range(1, num_frames))] = 0
                first_frame_mask = mask_lat_size[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(
                    first_frame_mask,
                    repeats=self.vae.temporal_compression_ratio,
                    dim=2,
                )
                mask_lat_size = torch.concat(
                    [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
                )
                mask_lat_size = mask_lat_size.view(
                    1,
                    -1,
                    self.vae.temporal_compression_ratio,
                    latent_height,
                    latent_width,
                )
                mask_lat_size = mask_lat_size.transpose(1, 2)
                mask_lat_size = mask_lat_size.to(latent_condition.device)
                image_latents = torch.concat([mask_lat_size, latent_condition], dim=1)

            batch.image_latent = image_latents

        # Offload models if needed
        if hasattr(self, "maybe_free_model_hooks"):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")

        return batch

    def retrieve_latents(
        self,
        encoder_output: torch.Tensor,
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
        vae_scale_factor: int,
        height: int | None = None,
        width: int | None = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:

        if isinstance(image, PIL.Image.Image):
            width, height = (
                self.server_args.pipeline_config.vae_config.calculate_dimensions(
                    image, vae_scale_factor, width, height
                )
            )
            image = resize(image, height, width, resize_mode=resize_mode)
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
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        if server_args.mode == ExecutionMode.PREPROCESS:
            result.add_check("height", batch.height, V.list_not_empty)
            result.add_check("width", batch.width, V.list_not_empty)
            result.add_check("num_frames", batch.num_frames, V.list_not_empty)
        else:
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
