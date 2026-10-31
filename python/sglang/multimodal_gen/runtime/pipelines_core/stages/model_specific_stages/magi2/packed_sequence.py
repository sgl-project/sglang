# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    MAGI2_MODALITY_AUDIO,
    MAGI2_MODALITY_TEXT,
    MAGI2_MODALITY_VIDEO,
    Magi2SegmentLayout,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    coords as magi2_coords,
)


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
        magi2_coords.video_coords(latent_shape=video_latent_shape, device=device),
    ]
    if audio_tokens:
        parts.append(magi2_coords.audio_coords(num_tokens=audio_tokens, device=device))
    parts.append(magi2_coords.text_coords(num_tokens=text_tokens, device=device))
    if ref_latent_hw:
        parts.extend(
            magi2_coords.ref_image_coords(
                token_counts=list(ref_patch_counts),
                feat_shapes=list(ref_latent_hw),
                video_time_steps=video_latent_shape[0],
                device=device,
            )
        )
    return torch.cat(parts, dim=0)
