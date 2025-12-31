# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs import WanI2V480PConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.models.vision_utils import load_image, load_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import best_output_size

logger = init_logger(__name__)

# Alias for convenience
V = StageValidators


# TODO: since this might change sampling params after logging, should be do this beforehand?


class InputValidationStage(PipelineStage):
    """
    Stage for validating and preparing inputs for diffusion pipelines.

    This stage validates that all required inputs are present and properly formatted
    before proceeding with the diffusion process.

    In this stage, input image and output image may be resized
    """

    def __init__(self, vae_image_processor=None):
        super().__init__()
        self.vae_image_processor = vae_image_processor

    def _generate_seeds(self, batch: Req, server_args: ServerArgs):
        """Generate seeds for the inference"""
        seed = batch.seed
        num_videos_per_prompt = batch.num_outputs_per_prompt

        assert seed is not None
        seeds = [seed + i for i in range(num_videos_per_prompt)]
        batch.seeds = seeds

        # Create generators based on generator_device parameter
        # Note: This will overwrite any existing batch.generator
        generator_device = batch.generator_device

        if generator_device == "cpu":
            device_str = "cpu"
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        batch.generator = [
            torch.Generator(device_str).manual_seed(seed) for seed in seeds
        ]

    def preprocess_condition_image(
        self,
        batch: Req,
        server_args: ServerArgs,
        condition_image_width,
        condition_image_height,
    ):
        """
        preprocess condition image
        NOTE: condition image resizing is only allowed in InputValidationStage
        """
        if server_args.pipeline_config.task_type == ModelTaskType.I2I:
            # calculate new condition image size
            calculated_size = (
                server_args.pipeline_config.calculate_condition_image_size(
                    batch.condition_image,
                    condition_image_width,
                    condition_image_height,
                )
            )

            # preprocess condition image if necessary
            if calculated_size is not None:
                calculated_width, calculated_height = calculated_size
                condition_image, calculated_size = (
                    server_args.pipeline_config.preprocess_condition_image(
                        batch.condition_image,
                        calculated_width,
                        calculated_height,
                        self.vae_image_processor,
                    )
                )
                batch.condition_image = condition_image

            # adjust output image size
            calculated_width, calculated_height = calculated_size
            width = batch.width or calculated_width
            height = batch.height or calculated_height
            multiple_of = (
                server_args.pipeline_config.vae_config.get_vae_scale_factor() * 2
            )
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of
            batch.width = width
            batch.height = height
        elif server_args.pipeline_config.task_type == ModelTaskType.TI2V:
            # duplicate with vae_image_processor
            # further processing for ti2v task
            img = batch.condition_image
            ih, iw = img.height, img.width
            patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
            vae_stride = (
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
            )
            dh, dw = patch_size[1] * vae_stride, patch_size[2] * vae_stride
            max_area = 704 * 1280
            ow, oh = best_output_size(iw, ih, dw, dh, max_area)

            scale = max(ow / iw, oh / ih)
            img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
            logger.info("resized img height: %s, img width: %s", img.height, img.width)

            # center-crop
            x1 = (img.width - ow) // 2
            y1 = (img.height - oh) // 2
            img = img.crop((x1, y1, x1 + ow, y1 + oh))
            assert img.width == ow and img.height == oh

            # to tensor
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
            img = img.unsqueeze(0)
            batch.height = oh
            batch.width = ow
            # TODO: should we store in a new field: pixel values?
            batch.condition_image = img

        elif isinstance(server_args.pipeline_config, WanI2V480PConfig):
            # TODO: could we merge with above?
            # resize image only, Wan2.1 I2V
            max_area = server_args.pipeline_config.max_area
            aspect_ratio = condition_image_height / condition_image_width
            mod_value = (
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
                * server_args.pipeline_config.dit_config.arch_config.patch_size[1]
            )
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

            batch.condition_image = batch.condition_image.resize((width, height))
            batch.height = height
            batch.width = width

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Validate and prepare inputs.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The validated batch information.
        """

        self._generate_seeds(batch, server_args)

        # Ensure prompt is properly formatted
        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided")

        # Ensure negative prompt is properly formatted if using classifier-free guidance
        if (
            batch.do_classifier_free_guidance
            and batch.negative_prompt is None
            and batch.negative_prompt_embeds is None
        ):
            raise ValueError(
                "For classifier-free guidance, either `negative_prompt` or "
                "`negative_prompt_embeds` must be provided"
            )

        # Validate number of inference steps
        if batch.num_inference_steps <= 0:
            raise ValueError(
                f"Number of inference steps must be positive, but got {batch.num_inference_steps}"
            )

        # Validate guidance scale if using classifier-free guidance
        if batch.do_classifier_free_guidance and batch.guidance_scale < 0:
            raise ValueError(
                f"Guidance scale must be positive, but got {batch.guidance_scale}"
            )

        # for i2v, get image from image_path
        # @TODO(Wei) hard-coded for wan2.2 5b ti2v for now. Should put this in image_encoding stage
        if batch.image_path is not None:
            if batch.image_path.endswith(".mp4"):
                image = load_video(batch.image_path)[0]
            else:
                image = load_image(batch.image_path)
            batch.condition_image = image
            condition_image_width, condition_image_height = image.width, image.height
            batch.original_condition_image_size = image.size

            self.preprocess_condition_image(
                batch, server_args, condition_image_width, condition_image_height
            )

        # if height or width is not specified at this point, set default to 720p
        default_height = 720
        default_width = 1080
        if batch.height is None and batch.width is None:
            batch.height = default_height
            batch.width = default_width
        elif batch.height is None:
            batch.height = batch.width * default_height // default_width
        elif batch.width is None:
            batch.width = batch.height * default_width // default_height

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify input validation stage inputs."""
        result = VerificationResult()
        result.add_check("seed", batch.seed, [V.not_none, V.non_negative_int])
        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds),
        )

        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check(
            "guidance_scale",
            batch.guidance_scale,
            lambda x: not batch.do_classifier_free_guidance or V.non_negative_float(x),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify input validation stage outputs."""
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        # Validate height and width

        if batch.height % 8 != 0 or batch.width % 8 != 0:
            raise ValueError(
                f"Height and width must be divisible by 8 but are {batch.height} and {batch.width}."
            )
        result.add_check("seeds", batch.seeds, V.list_not_empty)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result
