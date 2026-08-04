# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 visual/audio condition-noise augmentation.

The request's condition timestep is applied to both the tensor value and the
DiT timestep. Tokenizer artifacts remain clean
and reusable; this module materializes the fixed noised anchors immediately
before the denoise loop.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)

# Channel-major packed audio rows always carry a stereo layout.
MINIMAX_H3_AUDIO_COND_CHANNELS = 2


def minimax_h3_imgvid_cond_noise_aug_rows(
    clean_rows: torch.Tensor,
    *,
    condition_shapes: Sequence[Sequence[int]],
    target_latent_t: int,
    imgvid_cond_num_frames: int,
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    """Apply the imgvid-condition RF noise recipe to packed clean rows.

    ``condition_shapes`` contains ``(latent_t, latent_h, latent_w)`` in packed
    visual-condition order. A new CPU generator with the same row seed is
    created for every condition. Under the dependent-noise policy, each draw
    uses the target temporal length plus the template's imgvid-condition frame
    count, then slices the prefix matching the current condition.
    """

    noise_aug = float(noise_aug)
    if not 0.0 <= noise_aug <= 1.0:
        raise ValueError(f"noise_aug must be in [0, 1], got {noise_aug}")
    if noise_aug == 1.0:
        return clean_rows
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != 96:
        raise ValueError(
            "clean imgvid condition rows must have shape [n, 96], got "
            f"{list(clean_rows.shape)}"
        )

    target_latent_t = int(target_latent_t)
    imgvid_cond_num_frames = int(imgvid_cond_num_frames)
    if target_latent_t <= 0:
        raise ValueError(f"target_latent_t must be positive, got {target_latent_t}")
    if imgvid_cond_num_frames <= 0:
        raise ValueError(
            "imgvid_cond_num_frames must be positive when condition rows exist, "
            f"got {imgvid_cond_num_frames}"
        )

    parsed_shapes: list[tuple[int, int, int]] = []
    expected_rows = 0
    for raw_shape in condition_shapes:
        if len(raw_shape) != 3:
            raise ValueError(
                "each imgvid condition shape must be (latent_t, latent_h, latent_w), "
                f"got {list(raw_shape)}"
            )
        latent_t, latent_h, latent_w = (int(value) for value in raw_shape)
        if latent_t <= 0 or latent_h <= 0 or latent_w <= 0:
            raise ValueError(
                f"imgvid condition shape must be positive, got {list(raw_shape)}"
            )
        if latent_h % 2 or latent_w % 2:
            raise ValueError(
                "imgvid condition spatial dimensions must be divisible by 2, "
                f"got {(latent_t, latent_h, latent_w)}"
            )
        parsed_shapes.append((latent_t, latent_h, latent_w))
        expected_rows += latent_t * (latent_h // 2) * (latent_w // 2)
    if not parsed_shapes:
        raise ValueError("condition_shapes must not be empty")
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"clean imgvid condition rows {int(clean_rows.shape[0])} != "
            f"shape-derived rows {expected_rows}"
        )

    out: list[torch.Tensor] = []
    row_offset = 0
    timestep = torch.tensor(noise_aug, dtype=torch.float32, device=clean_rows.device)
    for latent_t, latent_h, latent_w in parsed_shapes:
        full_t = target_latent_t + imgvid_cond_num_frames
        if full_t < latent_t:
            raise ValueError(
                f"condition latent_t {latent_t} exceeds the noise draw "
                f"length {full_t}"
            )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            1,
            24,
            full_t,
            latent_h,
            latent_w,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )[:, :, :latent_t]
        noise_rows = minimax_h3_patchify_video_latent(noise, patch_size=[1, 2, 2]).to(
            device=clean_rows.device, dtype=torch.float32
        )
        row_count = int(noise_rows.shape[0])
        clean_part = clean_rows[row_offset : row_offset + row_count].to(torch.float32)
        out.append(timestep * clean_part + (1.0 - timestep) * noise_rows)
        row_offset += row_count
    return (out[0] if len(out) == 1 else torch.cat(out, dim=0)).contiguous()


def minimax_h3_audio_cond_noise_aug_rows(
    clean_rows: torch.Tensor,
    *,
    condition_audio_t: Sequence[int],
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    """Apply the audio-condition RF noise recipe to packed clean rows.

    ``condition_audio_t`` contains the latent T of each audio-bearing
    condition in canonical request order. Noise is drawn per condition
    element, with a fresh CPU generator seeded with ``seed + 1`` for every
    element.  Consequently each condition restarts the
    same RNG stream; concatenating the rows and drawing once would be
    numerically different for ordered multi-reference requests.

    The mix is intentionally evaluated on CPU in fp32 before the packed rows
    are transferred to the DiT device.
    """

    noise_aug = float(noise_aug)
    if not 0.0 <= noise_aug <= 1.0:
        raise ValueError(f"noise_aug must be in [0, 1], got {noise_aug}")
    if noise_aug == 1.0:
        return clean_rows
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != 32:
        raise ValueError(
            "clean audio condition rows must have shape [n, 32], got "
            f"{list(clean_rows.shape)}"
        )

    audio_channels = MINIMAX_H3_AUDIO_COND_CHANNELS
    parsed_audio_t = [int(value) for value in condition_audio_t]
    if not parsed_audio_t:
        raise ValueError("condition_audio_t must not be empty")
    if any(value <= 0 for value in parsed_audio_t):
        raise ValueError(
            f"condition audio latent lengths must be positive, got {parsed_audio_t}"
        )
    expected_rows = audio_channels * sum(parsed_audio_t)
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"clean audio condition rows {int(clean_rows.shape[0])} != "
            f"shape-derived rows {expected_rows}"
        )

    out: list[torch.Tensor] = []
    row_offset = 0
    timestep = torch.tensor(noise_aug, dtype=torch.float32, device="cpu")
    for audio_t in parsed_audio_t:
        row_count = audio_channels * audio_t
        clean_part = (
            clean_rows[row_offset : row_offset + row_count]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        noise = torch.randn(
            clean_part.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        out.append(timestep * clean_part + (1.0 - timestep) * noise)
        row_offset += row_count
    rows = out[0] if len(out) == 1 else torch.cat(out, dim=0)
    return rows.to(device=clean_rows.device, dtype=torch.float32).contiguous()


__all__ = [
    "minimax_h3_audio_cond_noise_aug_rows",
    "minimax_h3_imgvid_cond_noise_aug_rows",
]
