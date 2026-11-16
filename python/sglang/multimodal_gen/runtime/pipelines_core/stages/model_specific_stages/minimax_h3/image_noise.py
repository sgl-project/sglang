# SPDX-License-Identifier: Apache-2.0
"""Per-request noise for MiniMax H3 image releases (t2i / edit).

One ``torch.Generator`` per request, built the way ``InputValidationStage``
builds ``batch.generator`` (``sampling_params.generator_device``, else the
pipeline config's ``generator_device``, else the platform device; seeded with
the request seed) and consumed in this order:

1. target noise ``torch.randn(1, 24, 1, H / 16, W / 16)``, patchified
   ``[1, 2, 2]`` into the packed rows the denoise loop starts from;
2. per edit reference, in condition order: anchor noise
   ``torch.randn(1, 24, 1, h, w)`` on the reference's own even latent grid,
   patchified; the denoise loop restores references to
   ``t * clean + (1 - t) * noise`` with ``t = MINIMAX_H3_IMGVID_COND_TIMESTEP``;
3. per edit reference, in condition order: the VAE posterior sample inside
   ``video_vae.encode_images(..., generator=...)``.

Draws 1 and 2 depend only on the request geometry, so every rank of a replica
and every dedup sibling (``n > 1``) performs them with an identical result;
whichever image stage runs first on a rank caches them in ``batch.extra``
(visual encoding on the encoding rank, latent preparation elsewhere). Draw 3
happens only on the encoding rank, which is why it is last: nothing is drawn
after it, so the other ranks' generators never desynchronise. The
process-global RNG is never touched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

# batch.extra key of the MiniMaxH3ImageNoise drawn for the request.
MINIMAX_H3_IMAGE_NOISE_EXTRA_KEY = "minimax_h3_image_noise"


@dataclass(frozen=True)
class MiniMaxH3ImageNoise:
    """Draws 1 and 2 of the request, plus the generator positioned for draw 3."""

    reference_grids: tuple[tuple[int, int], ...]
    target_rows: torch.Tensor
    anchor_noise_rows: tuple[torch.Tensor, ...]
    generator: torch.Generator


def minimax_h3_image_reference_grid(width: int, height: int) -> tuple[int, int]:
    """``(latent_h, latent_w)`` of an edit reference: 16px cells, even count."""
    return int(height) // 16 // 2 * 2, int(width) // 16 // 2 * 2


def minimax_h3_image_reference_grids(batch: Any) -> tuple[tuple[int, int], ...]:
    """Reference grids from the published reference rows (any rank/sibling)."""
    payload = batch.extra.get(MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY)
    if payload is None:
        return ()
    return tuple(
        (int(entry["latent_h"]), int(entry["latent_w"])) for entry in payload["images"]
    )


def minimax_h3_image_generator_device(batch: Any, server_args: Any) -> str:
    """The InputValidationStage rule: request, then pipeline, then platform."""
    device = (
        batch.generator_device
        or server_args.pipeline_config.generator_device
        or current_platform.device_type
    )
    return "cpu" if device == "cpu" else current_platform.device_type


def _noise_rows(generator: torch.Generator, latent_h: int, latent_w: int):
    noise = torch.randn(
        1,
        24,
        1,
        latent_h,
        latent_w,
        generator=generator,
        device=generator.device,
        dtype=torch.float32,
    )
    return minimax_h3_patchify_video_latent(noise.cpu(), patch_size=[1, 2, 2])


def minimax_h3_image_noise(
    batch: Any,
    plan: Any,
    server_args: Any,
    *,
    reference_grids: Sequence[Sequence[int]],
) -> MiniMaxH3ImageNoise:
    """Draw (once per request) the target and anchor noise; cached in batch.extra."""
    grids = tuple((int(h), int(w)) for h, w in reference_grids)
    cached = batch.extra.get(MINIMAX_H3_IMAGE_NOISE_EXTRA_KEY)
    if cached is not None:
        if cached.reference_grids != grids:
            raise ValueError(
                "MiniMax H3 image noise was drawn for reference grids "
                f"{cached.reference_grids!r}, now asked for {grids!r}"
            )
        return cached
    shape = plan.shape
    generator = torch.Generator(
        device=minimax_h3_image_generator_device(batch, server_args)
    ).manual_seed(int(plan.seed))
    target_rows = _noise_rows(
        generator, int(shape["height"]) // 16, int(shape["width"]) // 16
    )
    anchor_noise_rows = tuple(_noise_rows(generator, h, w) for h, w in grids)
    noise = MiniMaxH3ImageNoise(grids, target_rows, anchor_noise_rows, generator)
    batch.extra[MINIMAX_H3_IMAGE_NOISE_EXTRA_KEY] = noise
    return noise


def minimax_h3_image_anchor_rows(
    batch: Any, clean_rows: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Edit anchors: each reference's clean rows noised with its draw 2 noise."""
    noise = batch.extra.get(MINIMAX_H3_IMAGE_NOISE_EXTRA_KEY)
    if noise is None:
        raise ValueError(
            "MiniMax H3 image anchor rows need the request noise "
            "(minimax_h3_image_noise) drawn before denoising"
        )
    if len(noise.anchor_noise_rows) != len(clean_rows):
        raise ValueError(
            f"{len(noise.anchor_noise_rows)} anchor noise draws for "
            f"{len(clean_rows)} references"
        )
    timestep = torch.tensor(MINIMAX_H3_IMGVID_COND_TIMESTEP, dtype=torch.float32)
    return torch.cat(
        [
            timestep * clean.to(torch.float32) + (1.0 - timestep) * drawn
            for clean, drawn in zip(clean_rows, noise.anchor_noise_rows)
        ],
        dim=0,
    )


__all__ = [
    "MINIMAX_H3_IMAGE_NOISE_EXTRA_KEY",
    "MiniMaxH3ImageNoise",
    "minimax_h3_image_anchor_rows",
    "minimax_h3_image_generator_device",
    "minimax_h3_image_noise",
    "minimax_h3_image_reference_grid",
    "minimax_h3_image_reference_grids",
]
