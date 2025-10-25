# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import PIL
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vision_utils import (
    get_default_height_width,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
    resize,
)
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines.stages.validators import VerificationResult
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

    def __init__(self, image_encoder, image_processor) -> None:
        """
        Initialize the prompt encoding stage.

        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary image encoder.
        """
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder

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
        self.image_encoder = self.image_encoder.to(get_local_torch_device())

        image = batch.pil_image

        image_inputs = self.image_processor(images=image, return_tensors="pt").to(
            get_local_torch_device()
        )
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.image_encoder(**image_inputs)
            image_embeds = outputs.last_hidden_state

        batch.image_embeds.append(image_embeds)

        if server_args.image_encoder_cpu_offload:
            self.image_encoder.to("cpu")

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage inputs."""
        result = VerificationResult()
        result.add_check("pil_image", batch.pil_image, V.not_none)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify image encoding stage outputs."""
        result = VerificationResult()
        result.add_check("image_embeds", batch.image_embeds, V.list_of_tensors_dims(3))
        return result


class ImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.

    This stage handles the encoding of pixel representations into the final
    input format (e.g., latents).
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
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
            mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width)
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask, dim=2, repeats=self.vae.temporal_compression_ratio
            )
            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(
                1, -1, self.vae.temporal_compression_ratio, latent_height, latent_width
            )
            mask_lat_size = mask_lat_size.transpose(1, 2)
            mask_lat_size = mask_lat_size.to(latent_condition.device)

            batch.image_latent = torch.concat([mask_lat_size, latent_condition], dim=1)

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
            height, width = get_default_height_width(
                image, vae_scale_factor, height, width
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
        result.add_check(
            "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        )
        return result
