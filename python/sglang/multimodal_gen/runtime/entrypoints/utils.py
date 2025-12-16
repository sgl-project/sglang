# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import dataclasses
import os

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger
from sglang.multimodal_gen.utils import shallow_asdict

logger = init_logger(__name__)


def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
) -> Req:
    """
    Settle SamplingParams according to ServerArgs

    """
    # Create a copy of inference args to avoid modifying the original.
    # Filter out fields not defined in Req to avoid unexpected-kw TypeError.
    params_dict = shallow_asdict(sampling_params)
    req_field_names = {f.name for f in dataclasses.fields(Req)}
    filtered_params = {k: v for k, v in params_dict.items() if k in req_field_names}
    req = Req(**filtered_params, VSA_sparsity=server_args.VSA_sparsity)
    req.adjust_size(server_args)

    if (req.width is not None and req.width <= 0) or (
        req.height is not None and req.height <= 0
    ):
        raise ValueError(
            f"Height, width must be positive integers, got "
            f"height={req.height}, width={req.width}"
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
    # Process outputs
    if sample.dim() == 3:
        # for images, dim t is missing
        sample = sample.unsqueeze(1)
    videos = rearrange(sample, "c t h w -> t c h w")
    frames = []
    # TODO: this can be batched
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        frames.append((x * 255).numpy().astype(np.uint8))

    # Save outputs if requested
    if save_output:
        if save_file_path:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            if data_type == DataType.VIDEO:
                imageio.mimsave(
                    save_file_path,
                    frames,
                    fps=fps,
                    format=data_type.get_default_extension(),
                )
            else:
                imageio.imwrite(save_file_path, frames[0])
            logger.info(f"Saved output to {CYAN}{save_file_path}{RESET}")
        else:
            logger.info(f"No output path provided, output not saved")

    return frames
