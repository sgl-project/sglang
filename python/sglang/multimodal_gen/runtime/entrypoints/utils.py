# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sgl-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import logging
import math

# Suppress verbose logging from imageio, which is triggered when saving images.
logging.getLogger("imageio").setLevel(logging.WARNING)
logging.getLogger("imageio_ffmpeg").setLevel(logging.WARNING)

from sglang.multimodal_gen.configs.sample.base import DataType, SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import shallow_asdict

logger = init_logger(__name__)


def prepare_sampling_params(
    prompt: str,
    server_args: ServerArgs,
    sampling_params: SamplingParams,
):
    pipeline_config = server_args.pipeline_config
    # Validate inputs
    if not isinstance(prompt, str):
        raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")

    # Process negative prompt
    if (
        sampling_params.negative_prompt is not None
        and not sampling_params.negative_prompt.isspace()
    ):
        # avoid stripping default negative prompt: ' ' for qwen-image
        sampling_params.negative_prompt = sampling_params.negative_prompt.strip()

    # Validate dimensions
    if sampling_params.num_frames <= 0:
        raise ValueError(
            f"height, width, and num_frames must be positive integers, got "
            f"height={sampling_params.height}, width={sampling_params.width}, "
            f"num_frames={sampling_params.num_frames}"
        )

    if pipeline_config.task_type.is_image_gen():
        # settle num_frames
        logger.debug(f"Setting num_frames to 1 because this is a image-gen model")
        sampling_params.num_frames = 1
        sampling_params.data_type = DataType.IMAGE
    else:
        # Adjust number of frames based on number of GPUs for video task
        use_temporal_scaling_frames = (
            pipeline_config.vae_config.use_temporal_scaling_frames
        )
        num_frames = sampling_params.num_frames
        num_gpus = server_args.num_gpus
        temporal_scale_factor = (
            pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )

        if use_temporal_scaling_frames:
            orig_latent_num_frames = (num_frames - 1) // temporal_scale_factor + 1
        else:  # stepvideo only
            orig_latent_num_frames = sampling_params.num_frames // 17 * 3

        if orig_latent_num_frames % server_args.num_gpus != 0:
            # Adjust latent frames to be divisible by number of GPUs
            if sampling_params.num_frames_round_down:
                # Ensure we have at least 1 batch per GPU
                new_latent_num_frames = (
                    max(1, (orig_latent_num_frames // num_gpus)) * num_gpus
                )
            else:
                new_latent_num_frames = (
                    math.ceil(orig_latent_num_frames / num_gpus) * num_gpus
                )

            if use_temporal_scaling_frames:
                # Convert back to number of frames, ensuring num_frames-1 is a multiple of temporal_scale_factor
                new_num_frames = (new_latent_num_frames - 1) * temporal_scale_factor + 1
            else:  # stepvideo only
                # Find the least common multiple of 3 and num_gpus
                divisor = math.lcm(3, num_gpus)
                # Round up to the nearest multiple of this LCM
                new_latent_num_frames = (
                    (new_latent_num_frames + divisor - 1) // divisor
                ) * divisor
                # Convert back to actual frames using the StepVideo formula
                new_num_frames = new_latent_num_frames // 3 * 17

            logger.info(
                "Adjusting number of frames from %s to %s based on number of GPUs (%s)",
                sampling_params.num_frames,
                new_num_frames,
                server_args.num_gpus,
            )
            sampling_params.num_frames = new_num_frames

        sampling_params.num_frames = server_args.pipeline_config.adjust_num_frames(
            sampling_params.num_frames
        )

    sampling_params.set_output_file_ext()
    sampling_params.log(server_args=server_args)
    return sampling_params


def prepare_request(
    prompt: str,
    server_args: ServerArgs,
    sampling_params: SamplingParams,
) -> Req:
    """
    Settle SamplingParams according to ServerArgs

    """
    # Create a copy of inference args to avoid modifying the original

    sampling_params = prepare_sampling_params(prompt, server_args, sampling_params)

    req = Req(
        **shallow_asdict(sampling_params),
        VSA_sparsity=server_args.VSA_sparsity,
    )
    # req.set_width_and_height(server_args)

    # if (req.width <= 0
    #     or req.height <= 0):
    #     raise ValueError(
    #         f"Height, width must be positive integers, got "
    #         f"height={req.height}, width={req.width}"
    #     )

    return req
