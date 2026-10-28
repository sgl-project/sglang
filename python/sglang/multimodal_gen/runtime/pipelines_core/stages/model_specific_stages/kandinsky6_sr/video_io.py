# SPDX-License-Identifier: Apache-2.0
# Frame selection and resize adapted from the Kandinsky 6 SR inference reference
# (k6_video / sr_core, Apache-2.0).
"""Bounded-memory SR video/audio I/O, preserving reference frame selection and resize."""

import logging

import msgspec
import numpy as np
import torch
from torch.nn import functional as F

from sglang.multimodal_gen.configs.sample.kandinsky6_sr_resolution import (
    ASPECT_MISMATCH_TOLERANCE,
)

logger = logging.getLogger(__name__)
# trained on 5 s at 24 fps; select at most 121 frames, aligned to 1+8k
TARGET_FPS = 24
MAX_NUM_FRAMES = 121
RESAMPLE_FPS_TOLERANCE = 1.5
SOURCE_AUDIO_SAMPLE_RATE = 44100
OUTPUT_FRAME_CHUNK = 8


class DecodedClip(msgspec.Struct, frozen=True):
    """Frames ready for SR plus the rate they play at."""

    frames: torch.Tensor  # [T, 3, H, W] uint8, T = 1 + 8k <= 121
    fps: int  # effective fps (24 when downsampled, else round(source fps))
    source_fps: float


class FrameSelector:
    """Stream reference indices round(i*step), capped at int(total/step) and max_frames.

    In-tolerance and slower sources keep every frame. Stop only once the source
    length guarantees that no retained frame will be trimmed by limit()."""

    def __init__(
        self,
        source_fps: float,
        *,
        target_fps: int = TARGET_FPS,
        max_frames: int = MAX_NUM_FRAMES,
    ) -> None:
        self.max_frames = max_frames
        self.downsample = (
            abs(source_fps - target_fps) >= RESAMPLE_FPS_TOLERANCE
            and source_fps > target_fps
        )
        self.step = source_fps / target_fps
        self.effective_fps = target_fps if self.downsample else round(source_fps)
        self._kept = 0
        self._next_index = 0

    def keep(self, index: int) -> bool:
        if self._kept >= self.max_frames:
            return False
        return not self.downsample or index == self._next_index

    def mark_kept(self) -> None:
        self._kept += 1
        if self.downsample:
            self._next_index = round(self._kept * self.step)

    def enough(self, frames_seen: int) -> bool:
        """True once ``max_frames`` are kept and the source is known to be long enough."""
        if self._kept < self.max_frames:
            return False
        return not self.downsample or int(frames_seen / self.step) >= self.max_frames

    def limit(self, total_frames: int, kept: int) -> int:
        """Number of kept frames the reference would return for a source of this length."""
        if self.downsample:
            kept = min(kept, int(total_frames / self.step))
        return min(kept, self.max_frames)


def decode_clip(
    path: str,
    *,
    target_fps: int = TARGET_FPS,
    max_frames: int = MAX_NUM_FRAMES,
) -> DecodedClip:
    """Decode the first video stream into the aligned uint8 clip the SR model expects."""
    import av

    frames: list[torch.Tensor] = []
    seen = 0
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.base_rate or 0.0)
        if source_fps <= 0:
            raise ValueError(f"Video {path} reports no usable fps (got {source_fps}).")
        selector = FrameSelector(
            source_fps, target_fps=target_fps, max_frames=max_frames
        )
        if selector.effective_fps < 1:
            raise ValueError(f"Video {path} has an unusable fps of {source_fps}.")
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if selector.keep(seen):
                array = frame.to_ndarray(format="rgb24")
                frames.append(torch.from_numpy(array).permute(2, 0, 1))
                selector.mark_kept()
            seen += 1
            if selector.enough(seen):
                break
    kept = selector.limit(seen, len(frames))
    if kept == 0:
        raise ValueError(f"Video {path} has no readable frames.")
    kept = 1 + 8 * ((kept - 1) // 8)
    return DecodedClip(
        frames=torch.stack(frames[:kept]).contiguous(),
        fps=selector.effective_fps,
        source_fps=source_fps,
    )


def extract_source_audio(
    path: str,
    *,
    sample_rate: int = SOURCE_AUDIO_SAMPLE_RATE,
    max_seconds: float | None = None,
) -> np.ndarray | None:
    """Decode the first audio stream as mono float32 at ``sample_rate`` (None if absent).

    ``max_seconds`` stops decoding once that much audio has been produced; the result is
    trimmed to exactly that length.
    """
    import av

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    limit = None if max_seconds is None else round(max_seconds * sample_rate)
    chunks: list[np.ndarray] = []
    produced = 0
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunk = resampled.to_ndarray().reshape(-1)
                chunks.append(chunk)
                produced += chunk.shape[0]
            if limit is not None and produced >= limit:
                break
        else:
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    if not chunks:
        return None
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    return audio if limit is None else audio[:limit]


def synthetic_clip(*, height: int, width: int, frames: int) -> torch.Tensor:
    """Deterministic ``[T, 3, H, W]`` uint8 clip (warmup input; no file needed)."""
    shape = (frames, 1, height, width)
    ys = torch.linspace(0, 255, height).view(1, 1, height, 1).expand(shape)
    xs = torch.linspace(0, 255, width).view(1, 1, 1, width).expand(shape)
    ts = torch.linspace(0, 255, frames).view(frames, 1, 1, 1).expand(shape)
    return (torch.cat([ys, xs, ts], dim=1) / 2 + 64).clamp(0, 255).to(torch.uint8)


def to_output_video(video: torch.Tensor) -> torch.Tensor:
    """uint8 ``[3, T, H, W]`` -> fp16 ``[1, 3, T, H, W]`` in [0, 1] that saves losslessly.

    The framework writers truncate ``(x * 255)`` to uint8.  fp16 halves the memory of a
    large clip, but a plain ``k / 255`` stored in fp16 truncates to ``k - 1`` for over half
    of the values when a consumer upcasts before multiplying.  ``(k + 0.5) / 255`` stays
    inside the bucket of ``k`` in every evaluation order.  Converted in frame chunks to
    bound the float32 temporaries.
    """
    channels, frames, height, width = video.shape
    out = torch.empty((1, channels, frames, height, width), dtype=torch.float16)
    for start in range(0, frames, OUTPUT_FRAME_CHUNK):
        chunk = video[:, start : start + OUTPUT_FRAME_CHUNK]
        out[0, :, start : start + OUTPUT_FRAME_CHUNK] = (
            (chunk.float() + 0.5) / 255.0
        ).to(torch.float16)
    return out


def resize_to_target(video: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Downscale [C, T, H, W] uint8 video in bounded-memory chunks, without cropping."""
    channels, frames, height, width = video.shape
    target_h, target_w = target_hw
    if (height, width) == (target_h, target_w):
        return video
    if target_h > height or target_w > width:
        raise ValueError(
            f"target {target_w}x{target_h} exceeds the SR result {width}x{height}; "
            "pick a lower tier or a higher --resolution-scale for this source."
        )
    source_ratio, target_ratio = width / height, target_w / target_h
    if abs(source_ratio - target_ratio) / target_ratio > ASPECT_MISMATCH_TOLERANCE:
        logger.warning(
            "Aspect mismatch: SR %sx%s is %.3f, target %sx%s is %.3f; resizing "
            "anisotropically (no crop), the picture will be squeezed by %.1f%%.",
            width,
            height,
            source_ratio,
            target_w,
            target_h,
            target_ratio,
            abs(source_ratio / target_ratio - 1) * 100,
        )
    out = torch.empty((channels, frames, target_h, target_w), dtype=torch.uint8)
    for start in range(0, frames, OUTPUT_FRAME_CHUNK):
        chunk = video[:, start : start + OUTPUT_FRAME_CHUNK].permute(1, 0, 2, 3).float()
        resized = F.interpolate(
            chunk, size=target_hw, mode="bilinear", antialias=True, align_corners=False
        )
        out[:, start : start + OUTPUT_FRAME_CHUNK] = (
            resized.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3)
        )
    logger.info(
        "Final output: %sx%s -> %sx%s (downscale x%.2f / x%.2f)",
        width,
        height,
        target_w,
        target_h,
        width / target_w,
        height / target_h,
    )
    return out
