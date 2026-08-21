# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded helpers shared by GLM vision encoders."""

import logging
import math
from collections.abc import Callable

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def run_glm_visual_chunked(
    run_visual: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Run a block-diagonal GLM ViT in row-aligned chunks.

    Each ``grid_thw`` row is an independent image/frame attention block, so
    splitting only between rows preserves model semantics while releasing the
    largest intermediate activations between forwards.
    """
    max_patches = envs.SGLANG_VLM_MAX_PATCHES_PER_VIT.get()
    max_images = envs.SGLANG_VLM_MAX_IMAGES_PER_VIT.get()
    if max_patches <= 0 and max_images <= 0:
        return run_visual(pixel_values, grid_thw)

    patches_per_row = [int(math.prod(row)) for row in grid_thw.tolist()]
    cumulative = [0]
    for patch_count in patches_per_row:
        cumulative.append(cumulative[-1] + patch_count)
    if pixel_values.size(0) != cumulative[-1]:
        raise ValueError(
            "GLM visual patch count does not match grid_thw: "
            f"{pixel_values.size(0)} != {cumulative[-1]}"
        )

    outputs = []
    start = 0
    while start < len(patches_per_row):
        end = start
        chunk_patches = 0
        chunk_rows = 0
        while end < len(patches_per_row):
            next_patches = patches_per_row[end]
            if max_patches > 0 and chunk_patches + next_patches > max_patches:
                break
            if max_images > 0 and chunk_rows + 1 > max_images:
                break
            chunk_patches += next_patches
            chunk_rows += 1
            end += 1

        # A single high-resolution row may exceed the configured patch limit;
        # keep it intact because vision attention cannot be split within a row.
        if end == start:
            end = start + 1
        outputs.append(
            run_visual(
                pixel_values[cumulative[start] : cumulative[end]],
                grid_thw[start:end],
            )
        )
        start = end

    logger.debug(
        "Chunked GLM ViT: %d rows / %d patches -> %d forwards",
        len(patches_per_row),
        cumulative[-1],
        len(outputs),
    )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
