# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

AUDIO_TIME_COMPRESSION = 8

# Ref images sit past the clip with a one-step gap so they cannot alias the last frame.
REF_IMAGE_TIME_GAP = 2


def grid_coords(
    *,
    shape: tuple[int, int, int],
    ref_shape: tuple[int, int, int],
    offset: tuple[int, int, int] = (0, 0, 0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    ranges = [
        torch.arange(size, device=device, dtype=dtype) + off
        for size, off in zip(shape, offset)
    ]
    grid = torch.meshgrid(*ranges, indexing="ij")
    positions = torch.stack(grid, dim=-1).reshape(-1, 3)
    meta = torch.tensor([*shape, *ref_shape], device=device, dtype=dtype)
    return torch.cat([positions, meta.expand(positions.shape[0], -1)], dim=-1)


def video_coords(
    *,
    latent_shape: tuple[int, int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Video grid coordinates; ref grid is the grid itself, so the rope scale is 1."""
    return grid_coords(shape=latent_shape, ref_shape=latent_shape, device=device)


def audio_coords(
    *, num_tokens: int, device: torch.device | None = None
) -> torch.Tensor:
    """Audio coordinates on a compressed time axis."""
    ref_t = (num_tokens - 1) // AUDIO_TIME_COMPRESSION + 1
    return grid_coords(shape=(num_tokens, 1, 1), ref_shape=(ref_t, 1, 1), device=device)


def text_coords(*, num_tokens: int, device: torch.device | None = None) -> torch.Tensor:
    """``ref_T = 1`` zeroes the rope scale; the negative offset keeps text disjoint from video time."""
    return grid_coords(
        shape=(num_tokens, 1, 1),
        ref_shape=(1, 1, 1),
        offset=(-num_tokens, 0, 0),
        device=device,
    )


def ref_image_coords(
    *,
    token_counts: list[int],
    feat_shapes: list[tuple[int, int] | None],
    video_time_steps: int,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Per-image sentinel row plus its spatial grid."""
    out: list[torch.Tensor] = []
    for index, tokens in enumerate(token_counts):
        shape = feat_shapes[index] if index < len(feat_shapes) else None
        if shape is None:
            side = int(math.ceil(math.sqrt(tokens)))
            height = width = side
        else:
            height, width = shape
        time_offset = video_time_steps + REF_IMAGE_TIME_GAP + index
        # (-1, -1) marks a whole-image token, not a position within the image.
        out.append(
            torch.tensor(
                [[time_offset, -1, -1, 1, height, width, 1, height, width]],
                device=device,
                dtype=torch.float32,
            )
        )
        out.append(
            grid_coords(
                shape=(1, height, width),
                ref_shape=(1, height, width),
                offset=(time_offset, 0, 0),
                device=device,
            )[:tokens]
        )
    return out
