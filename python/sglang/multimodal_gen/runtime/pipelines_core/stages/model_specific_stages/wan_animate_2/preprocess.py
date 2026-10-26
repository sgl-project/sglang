# SPDX-License-Identifier: Apache-2.0
# Adapted from the Wan-Animate-2 reference implementation (Apache-2.0):
# https://github.com/Wan-Video/Wan-Animate-2
#
# Keep the original Wan preprocessing numerics; the newer Diffusers modular
# processor is not a numerically interchangeable replacement. The
# zigzag_padding / resize_by_area names are kept for parity with the official helpers;
# make_conditioning_mask is the official get_i2v_mask.
from __future__ import annotations

import math
from copy import deepcopy
from io import BytesIO
from typing import Literal

import cv2
import msgspec
import numpy as np
import torch

from sglang.srt.utils.common import get_video_bytes

__all__ = [
    "LetterboxInfo",
    "MIN_SINGLE_CLIP_LEN",
    "resize_by_area",
    "letterbox_resize",
    "get_frame_indices",
    "get_padding_len",
    "single_clip_reference_video_len",
    "zigzag_padding",
    "make_conditioning_mask",
    "read_reference_video_frames",
    "write_reference_video",
]


class LetterboxInfo(msgspec.Struct, frozen=True):
    """Where letterbox_resize put the content inside the padded canvas."""

    # Axis the padding bars sit on: "width" pads left/right, "height" pads top/bottom.
    pad_axis: Literal["width", "height"]
    # Bar size in pixels before the content along pad_axis.
    padding: int
    # Content extent in pixels along pad_axis.
    content_len: int

    def crop(self, frames: np.ndarray) -> np.ndarray:
        """Cut the bars back out of ``[T, H, W, C]`` frames."""
        start, stop = self.padding, self.padding + self.content_len
        if self.pad_axis == "width":
            return frames[:, :, start:stop, :]
        return frames[:, start:stop, :, :]


def resize_by_area(
    image: np.ndarray,
    target_area: int,
    keep_aspect_ratio: bool = True,
    divisor: int = 16,
    padding_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize ``image`` (``[H, W, C]`` uint8) so its area is <= ``target_area``, then letterbox-pad to that size."""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if keep_aspect_ratio:
        height_target = math.sqrt(target_area / aspect_ratio)
        width_target = target_area / height_target
    else:
        width_target = height_target = math.sqrt(target_area)
    width_target, height_target = (
        int((width_target // divisor) * divisor),
        int((height_target // divisor) * divisor),
    )

    # Resize (INTER_AREA when shrinking, INTER_LINEAR when enlarging)
    # Taken as is from the reference implementation.
    interpolation = (
        cv2.INTER_AREA if (width_target * height_target < w * h) else cv2.INTER_LINEAR
    )

    return letterbox_resize(
        image,
        height=height_target,
        width=width_target,
        padding_color=padding_color,
        interpolation=interpolation,
    )


def letterbox_resize(
    image: np.ndarray,
    height: int = 512,
    width: int = 512,
    padding_color: tuple[int, int, int] = (0, 0, 0),
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Aspect-preserving resize of ``image`` into a ``[height, width, C]`` uint8 canvas with letterbox padding."""
    height_source = image.shape[0]
    width_source = image.shape[1]
    num_channels_source = image.shape[2]

    canvas = np.zeros((height, width, num_channels_source))
    if num_channels_source == 1:
        canvas[:, :, 0] = padding_color[0]
    else:
        canvas[:, :, 0] = padding_color[0]
        canvas[:, :, 1] = padding_color[1]
        canvas[:, :, 2] = padding_color[2]

    if (height_source / width_source) > (height / width):
        width_target = int(height / height_source * width_source)
        img = cv2.resize(image, (width_target, height), interpolation=interpolation)
        padding = int((width - width_target) / 2)
        if len(img.shape) == 2:
            img = img[
                :, :, np.newaxis
            ]  # grayscale image: add a num_channels_source dimension
        canvas[:, padding : padding + width_target, :] = img
        pad_axis = "width"
        content_len = width_target
    else:
        height_target = int(width / width_source * height_source)
        img = cv2.resize(image, (width, height_target), interpolation=interpolation)
        padding = int((height - height_target) / 2)
        if len(img.shape) == 2:
            img = img[
                :, :, np.newaxis
            ]  # grayscale image: add a num_channels_source dimension
        canvas[padding : padding + height_target, :, :] = img
        pad_axis = "height"
        content_len = height_target

    canvas = np.uint8(canvas)
    return canvas, LetterboxInfo(
        pad_axis=pad_axis, padding=padding, content_len=content_len
    )


def get_frame_indices(
    num_frames_source: int, fps_source: float, num_frames_target: int, fps_target: float
) -> list[int]:
    """Source frame index (``list[int]``, nearest frame clipped to the source range) for each of the
    ``num_frames_target`` output frames when resampling from ``fps_source`` to ``fps_target``."""

    times = np.arange(0, num_frames_target) / fps_target
    frame_indices_source = np.round(times * fps_source).astype(int)
    frame_indices_source = np.clip(frame_indices_source, 0, num_frames_source - 1)

    return frame_indices_source.tolist()


# Fewest frames the last clip may add beyond its overlap; a multiple of 4 so it also sits
# on the temporal-VAE grid. Taken as is from the reference.
_MIN_LAST_CLIP_NEW_FRAMES = 28


def get_padding_len(
    num_frames: int, clip_len: int, num_frames_conditioning: int = 1
) -> int:
    """Reference-video length after zigzag padding so the last clip is long enough and
    4k-aligned; taken as is from the reference implementation."""

    remaining = (num_frames - num_frames_conditioning) % (
        clip_len - num_frames_conditioning
    )
    # Short leftover: raise it to the minimum. Otherwise: round up to the next multiple
    # of 4 (VAE temporal stride); the minimum is itself a multiple of 4.
    if remaining < _MIN_LAST_CLIP_NEW_FRAMES:
        padding_needed = _MIN_LAST_CLIP_NEW_FRAMES - remaining
    else:
        padding_needed = 4 - remaining % 4

    return num_frames + padding_needed


# get_padding_len extends every reference video to at least this many frames, so a shorter
# clip_len can never hold the whole video in one clip.
MIN_SINGLE_CLIP_LEN = _MIN_LAST_CLIP_NEW_FRAMES + 1


def single_clip_reference_video_len(
    clip_len: int, num_frames_conditioning: int = 1
) -> int:
    """Reference-video frame count that get_padding_len extends to exactly ``clip_len``,
    so build_schedule yields one full-length clip; used to size the synthetic warmup video."""
    num_frames = clip_len - 1
    if (
        clip_len < MIN_SINGLE_CLIP_LEN
        or get_padding_len(num_frames, clip_len, num_frames_conditioning) != clip_len
    ):
        raise ValueError(
            f"clip_len={clip_len} cannot hold a reference video in a single clip; "
            f"the schedule needs clip_len >= {MIN_SINGLE_CLIP_LEN}"
        )
    return num_frames


def zigzag_padding(array: list, target_len: int) -> list:
    """Pad ``array`` up to ``target_len`` by bouncing the read index back and forth (deep copies)."""
    if target_len < len(array):
        raise ValueError(
            f"zigzag_padding only extends: target_len {target_len} < len(array) {len(array)}"
        )
    index = 0
    flip = False
    target_array = []
    while len(target_array) < target_len:
        target_array.append(deepcopy(array[index]))
        if flip:
            index -= 1
        else:
            index += 1
        if index == 0 or index == len(array) - 1:
            flip = not flip
    return target_array


def make_conditioning_mask(
    latent_t: int,
    latent_h: int,
    latent_w: int,
    num_prefix_conditioning_frames: int = 1,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build the ``[4, latent_t, latent_h, latent_w]`` fp32 i2v conditioning mask; ``num_prefix_conditioning_frames`` is the
    conditioned prefix length in pixel frames."""

    # The VAE maps the first frame to 1 frame in latent space.
    # For each subsequent 4 pixel frames it maps them to 1 frame in the latent space.
    # Hence, the ``(latent_t - 1) * 4 + 1``.
    # VAE: 4k + 1 pixel frames -> k + 1 latent frames
    mask = torch.zeros(1, (latent_t - 1) * 4 + 1, latent_h, latent_w, device=device)

    mask[:, :num_prefix_conditioning_frames] = 1
    mask = torch.concat(
        [torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1
    )  # repeat the first pixel frame's flag 4x: (k - 1) * 4 + 1 -> 4 * k pixel slots

    mask = mask.view(1, mask.shape[1] // 4, 4, latent_h, latent_w)
    # (1, 4*(k + 1), latent_h, latent_w) -> (1, (k + 1), 4, latent_h, latent_w)

    mask = mask.transpose(1, 2)[0]
    # (1, (k + 1), 4, latent_h, latent_w) -> (4, (k + 1), latent_h, latent_w)
    return mask


def read_reference_video_frames(path: str, fps_target: int) -> np.ndarray:
    """Decode the reference video resampled to ``fps_target``; returns ``[N, H, W, 3]`` uint8 RGB.

    Three readers: decord on CPU as the reference implementation does (its GPU decoder
    exists only in CUDA builds); decord2 is the arm64 build of the same API; PyAV is the
    fallback when neither is installed.
    """
    source = (
        BytesIO(get_video_bytes(path))
        if path.startswith(("http://", "https://", "data:"))
        else path
    )
    try:
        from decord import VideoReader
    except ImportError:
        try:
            from decord2 import VideoReader
        except ImportError:
            VideoReader = None

    if VideoReader is not None:
        video_reader = VideoReader(source)
        num_frames_source = len(video_reader)
        fps_source = video_reader.get_avg_fps()
        num_frames_target = int(num_frames_source / fps_source * fps_target)
        indices = get_frame_indices(
            num_frames_source, fps_source, num_frames_target, fps_target
        )
        return video_reader.get_batch(indices).asnumpy()

    import av

    # Decode everything once so num_frames_source matches decord's len() exactly.
    with av.open(source) as container:
        stream = container.streams.video[0]
        fps_source = float(stream.average_rate)
        frames_source = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    num_frames_source = len(frames_source)
    num_frames_target = int(num_frames_source / fps_source * fps_target)
    indices = get_frame_indices(
        num_frames_source, fps_source, num_frames_target, fps_target
    )
    return np.stack([frames_source[i] for i in indices])


def write_reference_video(path: str, frames: np.ndarray, fps: int) -> None:
    """Write ``[N, H, W, 3]`` uint8 RGB frames to ``path`` as an mp4 at ``fps`` through the
    runtime's video writer (the one request outputs go through), so the synthetic warmup
    video is encoded exactly like a request output and decodes with the readers above."""
    from sglang.multimodal_gen.configs.sample.sampling_params import DataType
    from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample

    try:
        post_process_sample(frames, DataType.VIDEO, fps, save_file_path=path)
    except Exception as e:
        raise RuntimeError(
            f"the shared video writer (post_process_sample) could not write the "
            f"synthetic reference video {path}: {e}"
        ) from e


def validate_and_get_single_string(value: str | list[str] | None, name: str) -> str:
    """This pipeline produces one video per request, so a batched prompt field,
    video path or image path must hold exactly one string."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"Wan-Animate-2 takes exactly one '{name}' per request, got {len(value)}"
            )
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(
            f"Wan-Animate-2 requires '{name}' to be a string, got {value!r}"
        )
    return value
