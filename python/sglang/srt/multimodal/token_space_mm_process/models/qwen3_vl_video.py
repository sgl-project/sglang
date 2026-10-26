"""Qwen3-VL family video sampling and resizing, as qwen-vl-utils 0.0.14 ``fetch_video`` does.

The Qwen3-VL processor family (Qwen3-VL, Qwen3.5, Qwen3.8) is fed frames that
qwen-vl-utils already sampled and resized, with ``do_resize=False`` and the
sampling metadata. This module reproduces that step on a decoded video so the
token-space route produces the frames, timestamps and pixel values of the
reference pipeline.
"""

import math

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from sglang.srt.multimodal.qwen_vl_media_processing import smart_nframes

# qwen-vl-utils 0.0.14 constants.
_MAX_RATIO = 200
_VIDEO_MIN_TOKEN_NUM = 128
_VIDEO_MAX_TOKEN_NUM = 768
_MODEL_SEQ_LEN = 128000
_FRAME_FACTOR = 2


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > _MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {_MAX_RATIO}")
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_qwen3_vl_video(
    decoder, config: dict, factor: int
) -> tuple[torch.Tensor, dict]:
    """Sampled TCHW float frames and their metadata for a decoded video.

    ``config`` takes the qwen-vl-utils video keys (fps or nframes, min/max_frames,
    min/max/total_pixels, resized_height/width); ``factor`` is patch size times
    spatial merge size.
    """
    total_frames, video_fps = len(decoder), decoder.avg_fps
    nframes = smart_nframes(config, total_frames=total_frames, video_fps=video_fps)
    indices = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    frames = torch.as_tensor(decoder.get_frames_at(indices)).permute(0, 3, 1, 2)

    nframes, _, height, width = frames.shape
    min_pixels = config.get("min_pixels", _VIDEO_MIN_TOKEN_NUM * factor * factor)
    total_pixels = config.get("total_pixels", _MODEL_SEQ_LEN * factor * factor * 0.9)
    max_pixels = max(
        min(
            _VIDEO_MAX_TOKEN_NUM * factor * factor,
            total_pixels / nframes * _FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )
    max_pixels = min(config.get("max_pixels", max_pixels), max_pixels)
    if "resized_height" in config and "resized_width" in config:
        resized_height, resized_width = _smart_resize(
            config["resized_height"],
            config["resized_width"],
            factor,
            min_pixels=_VIDEO_MIN_TOKEN_NUM * factor * factor,
            max_pixels=16384 * factor * factor,
        )
    else:
        resized_height, resized_width = _smart_resize(
            height, width, factor, min_pixels, max_pixels
        )
    frames = F.resize(
        frames,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    metadata = {
        "fps": video_fps,
        "frames_indices": indices,
        "total_num_frames": total_frames,
        "video_backend": "torchcodec",
    }
    return frames, metadata
