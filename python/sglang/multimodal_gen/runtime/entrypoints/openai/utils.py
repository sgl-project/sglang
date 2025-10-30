# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import os

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from fastapi import UploadFile

from sglang.multimodal_gen.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
            logger.info(f"Saved output to {save_file_path}")
        else:
            logger.info(f"No output path provided, output not saved")

    return frames


def _parse_size(size: str) -> tuple[int, int]:
    try:
        parts = size.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        # Fallback to default portrait 720x1280
        return 720, 1280


# Helpers
async def _save_upload_to_path(upload: UploadFile, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    content = await upload.read()
    with open(target_path, "wb") as f:
        f.write(content)
    return target_path
