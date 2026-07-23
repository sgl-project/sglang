# SPDX-License-Identifier: Apache-2.0
"""Sampling params for NVIDIA OmniDreams (autoregressive video world model)."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class OmniDreamsSamplingParams(SamplingParams):
    data_type: DataType = DataType.VIDEO

    # 720p single-view i2v defaults (latent grid is /8 spatial, /4 temporal).
    height: int = 704
    width: int = 1280
    # 2-step distilled flow-match schedule; CFG disabled.
    num_inference_steps: int = 2
    guidance_scale: float = 1.0

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(1280, 704)]
    )

    # --- Autoregressive rollout knobs (see FlashDreams streaming inference) ---
    # Number of latent frames produced per chunk.
    len_t: int = 2
    # Rolling KV-cache window (in latent frames) and permanent sink size.
    window_size_t: int = 6
    sink_size_t: int = 0
    # Raw timestep injected as context noise on cached/clean frames.
    context_noise: int = 128

    # HD-map conditioning input -- OmniDreams' central per-frame control signal.
    # Accepts a video path (``.mp4``/``.gif``/...; decoded to per-frame rasters),
    # a per-frame list of image paths, or -- degenerate fallback -- a single image
    # broadcast across every frame (no temporal motion). ``None`` disables HDMap.
    hdmap_path: str | list[str] | None = None
