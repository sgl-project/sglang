# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import os

import imageio
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger

logger = init_logger(__name__)


def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
) -> Req:
    """
    Create a Req object with sampling_params as a parameter.
    """
    req = Req(sampling_params=sampling_params, VSA_sparsity=server_args.VSA_sparsity)
    diffusers_kwargs = getattr(sampling_params, "diffusers_kwargs", None)
    if diffusers_kwargs:
        req.extra["diffusers_kwargs"] = diffusers_kwargs

    req.adjust_size(server_args)

    if (req.width is not None and req.width <= 0) or (
        req.height is not None and req.height <= 0
    ):
        raise ValueError(
            f"Height and width must be positive, got height={req.height}, width={req.width}"
        )

    return req


def post_process_sample(
    sample: torch.Tensor,
    data_type: DataType,
    fps: int,
    save_output: bool = True,
    save_file_path: str = None,
):
    """
    Process sample output and save video if necessary
    """
    # 1. Vectorized processing on GPU/CPU tensor
    if sample.dim() == 3:
        # for images, dim t is missing
        sample = sample.unsqueeze(1)

    # Convert to uint8 and move to CPU in bulk
    # Shape: [C, T, H, W] -> [T, H, W, C]
    sample = (sample * 255).clamp(0, 255).to(torch.uint8)
    videos = sample.permute(1, 2, 3, 0).cpu().numpy()

    # Convert to list of frames for imageio
    frames = list(videos)

    # 2. Save outputs if requested
    if save_output:
        if save_file_path:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            if data_type == DataType.VIDEO:
                # TODO: make this configurable
                quality = 5
                imageio.mimsave(
                    save_file_path,
                    frames,
                    fps=fps,
                    format=data_type.get_default_extension(),
                    codec="libx264",
                    quality=quality,
                )
            else:
                quality = 75
                if len(frames) > 1:
                    for i, image in enumerate(frames):
                        parts = save_file_path.rsplit(".", 1)
                        if len(parts) == 2:
                            indexed_path = f"{parts[0]}_{i}.{parts[1]}"
                        else:
                            indexed_path = f"{save_file_path}_{i}"
                        imageio.imwrite(indexed_path, image, quality=quality)
                else:
                    imageio.imwrite(save_file_path, frames[0], quality=quality)
            logger.info(f"Output saved to {CYAN}{save_file_path}{RESET}")
        else:
            logger.info(f"No output path provided, output not saved")

    return frames
