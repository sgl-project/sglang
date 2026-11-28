# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_layers import (
    MAGI2_MODALITY_AUDIO,
    MAGI2_MODALITY_TEXT,
    MAGI2_MODALITY_VIDEO,
    Magi2SegmentLayout,
)

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


def build_layout(
    *,
    video_latent_thw: tuple[int, int, int],
    audio_tokens: int,
    text_tokens: int,
    device: torch.device,
    ref_patch_counts: Sequence[int] = (),
) -> Magi2SegmentLayout:
    """Order is fixed: video, audio, text, then per reference image one special token followed by its patches."""
    video_tokens = math.prod(video_latent_thw)
    base = video_tokens + audio_tokens + text_tokens
    total = base + sum(1 + count for count in ref_patch_counts)

    video_index = torch.arange(video_tokens, device=device)
    audio_index = torch.arange(video_tokens, video_tokens + audio_tokens, device=device)
    text_index = torch.arange(video_tokens + audio_tokens, base, device=device)

    # One image's two halves are tagged differently: the special token enters
    # through the text embedder, its patches through the video embedder.
    special_positions: list[int] = []
    patch_positions: list[int] = []
    cursor = base
    for count in ref_patch_counts:
        special_positions.append(cursor)
        patch_positions.extend(range(cursor + 1, cursor + 1 + count))
        cursor += 1 + count

    ref_special_index = torch.tensor(special_positions, dtype=torch.long, device=device)
    ref_patch_index = torch.tensor(patch_positions, dtype=torch.long, device=device)

    modality_ids = torch.empty(total, dtype=torch.long, device=device)
    modality_ids[video_index] = MAGI2_MODALITY_VIDEO
    modality_ids[audio_index] = MAGI2_MODALITY_AUDIO
    modality_ids[text_index] = MAGI2_MODALITY_TEXT
    modality_ids[ref_special_index] = MAGI2_MODALITY_TEXT
    modality_ids[ref_patch_index] = MAGI2_MODALITY_VIDEO

    # One segment: every modality attends to every other, so the varlen span is
    # the whole sequence rather than one range per modality.
    cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)

    return Magi2SegmentLayout(
        total_tokens=total,
        video_index=video_index,
        audio_index=audio_index,
        text_index=text_index,
        modality_ids=modality_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=total,
        video_latent_thw=tuple(video_latent_thw),
        ref_special_index=ref_special_index,
        ref_patch_index=ref_patch_index,
    )


def build_timesteps(
    *,
    layout: Magi2SegmentLayout,
    video_t: torch.Tensor,
    audio_t: torch.Tensor,
) -> torch.Tensor:
    """Text and reference rows must read exactly zero, or they present as partially-noised inputs."""
    times = torch.zeros(
        layout.total_tokens, device=layout.modality_ids.device, dtype=torch.float32
    )
    times[layout.video_index] = video_t.to(times.dtype)
    if layout.audio_index.numel():
        times[layout.audio_index] = audio_t.to(times.dtype)
    return times


def build_coords(
    *,
    video_latent_shape: tuple[int, int, int],
    audio_tokens: int,
    text_tokens: int,
    device: torch.device,
    ref_latent_hw: Sequence[tuple[int, int]] = (),
    ref_patch_counts: Sequence[int] = (),
) -> torch.Tensor:
    parts = [
        video_coords(latent_shape=video_latent_shape, device=device),
    ]
    if audio_tokens:
        parts.append(audio_coords(num_tokens=audio_tokens, device=device))
    parts.append(text_coords(num_tokens=text_tokens, device=device))
    if ref_latent_hw:
        parts.extend(
            ref_image_coords(
                token_counts=list(ref_patch_counts),
                feat_shapes=list(ref_latent_hw),
                video_time_steps=video_latent_shape[0],
                device=device,
            )
        )
    return torch.cat(parts, dim=0)
