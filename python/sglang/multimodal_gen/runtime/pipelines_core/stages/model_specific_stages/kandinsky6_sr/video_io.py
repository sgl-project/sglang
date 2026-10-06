# SPDX-License-Identifier: Apache-2.0
"""Streaming video / audio input and output helpers of Kandinsky 6 video SR.

Decoding follows the reference CLI (``read_video_tchw_uint8`` ->
``resample_to_target_fps`` -> ``clip_to_aligned_frames``) frame for frame, but streams:
only the frames that survive fps resampling and the 121-frame cap are converted to RGB
and kept, and decoding stops as soon as the outcome is known.  PyAV is imported lazily
(it is part of the ``diffusion`` extra).
"""

import msgspec
import numpy as np
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_utils import (
    MAX_NUM_FRAMES,
    RESAMPLE_FPS_TOLERANCE,
    TARGET_FPS,
    clip_to_aligned_frames,
)

SOURCE_AUDIO_SAMPLE_RATE = 44100
OUTPUT_FRAME_CHUNK = 8


class DecodedClip(msgspec.Struct, frozen=True):
    """Frames ready for SR plus the rate they play at."""

    frames: torch.Tensor  # [T, 3, H, W] uint8, T = 1 + 8k <= 121
    fps: int  # effective fps (24 when downsampled, else round(source fps))
    source_fps: float


class FrameSelector:
    """Streaming form of ``resample_to_target_fps`` + the first-121-frames cap.

    Feed source frames in order: :meth:`keep` says whether frame ``index`` is one of the
    selected ones (call :meth:`mark_kept` for each), :meth:`enough` says when no further
    frame can change the result, and :meth:`limit` trims the selection once the source
    length is known.  For a source faster than the target the selected indices are
    ``round(i * step)``, ``step = source_fps / target_fps``, of which there are
    ``int(total / step)`` (those below ``total``), exactly as in the reference;
    otherwise every frame is selected.
    """

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
    video = torch.stack(frames[:kept]).contiguous()
    return DecodedClip(
        frames=clip_to_aligned_frames(video, max_frames),
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
