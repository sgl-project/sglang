# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import torch

    # The rectified-flow sigma range is fixed at [1.0, 0.0].  The Euler
    # denoise loop performs ``len(sigmas) - 1`` updates, so the schedule must
    # hold exactly ``num_steps + 1`` points to honor the requested step count.
    # torch.linspace already includes both endpoints, and the time-shift
    # mapping is strictly monotonic, so no post-hoc deduplication is needed
    # (it could only ever shrink the schedule and break the N+1 contract).
    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps) + 1,
        device="cpu",
        dtype=torch.float32,
    )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    return [float(value) for value in shifted.tolist()]
