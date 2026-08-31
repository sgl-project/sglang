"""Multimodal preprocessing for Qwen3-VL and Qwen3.5."""

import time

import numpy as np
from transformers.image_utils import SizeDict
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as smart_resize_video,
)

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
    smart_nframes,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.utils import logger

MAX_VIDEO_DECODE_CHUNK_BYTES = 512 * 1024 * 1024
QWEN3_VL_MODEL_TYPES = frozenset(
    {
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
    }
)
QWEN3_EARLY_VIDEO_CONFIG_KEYS = frozenset(
    {"fps", "max_frames", "min_frames", "nframes", "resample", "size"}
)


def _as_size_dict(size) -> SizeDict:
    return size if isinstance(size, SizeDict) else SizeDict(**size)


def _sampling_config(video_processor, video_config):
    config = dict(video_config)
    if "nframes" not in config:
        for key in ("fps", "min_frames", "max_frames"):
            value = getattr(video_processor, key, None)
            if value is not None:
                config.setdefault(key, value)
    return config


def _resize_geometry(video_processor, video_config, *, num_frames, height, width):
    size = _as_size_dict(video_config.get("size", video_processor.size))
    patch_size = video_config.get("patch_size", video_processor.patch_size)
    merge_size = video_config.get("merge_size", video_processor.merge_size)
    temporal_patch_size = video_config.get(
        "temporal_patch_size", video_processor.temporal_patch_size
    )
    resized_height, resized_width = smart_resize_video(
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_factor=temporal_patch_size,
        factor=patch_size * merge_size,
        min_pixels=size.shortest_edge,
        max_pixels=size.longest_edge,
    )
    return SizeDict(height=resized_height, width=resized_width)


def _decode_and_resize(
    vr,
    indices,
    *,
    video_processor,
    video_config,
    max_decode_chunk_bytes,
):
    height, width, channels = vr.frame_shape
    frame_bytes = height * width * channels
    frames_per_chunk = max_decode_chunk_bytes // frame_bytes
    if frames_per_chunk < 1:
        raise ValueError(
            f"A single decoded video frame requires {frame_bytes} bytes, exceeding "
            f"the {max_decode_chunk_bytes}-byte Qwen3 video preprocessing limit."
        )

    resize_size = _resize_geometry(
        video_processor,
        video_config,
        num_frames=len(indices),
        height=height,
        width=width,
    )
    resample = video_config.get("resample", video_processor.resample)
    video = None
    for start in range(0, len(indices), frames_per_chunk):
        chunk_indices = indices[start : start + frames_per_chunk]
        chunk = vr.get_frames_as_tensor(chunk_indices).permute(0, 3, 1, 2)
        resized_chunk = video_processor.resize(
            chunk, size=resize_size, resample=resample
        )
        if video is None:
            video = resized_chunk.new_empty((len(indices), *resized_chunk.shape[1:]))
        video[start : start + len(chunk_indices)].copy_(resized_chunk)
        del chunk, resized_chunk
    return video


async def preprocess_video(
    vr,
    *,
    video_processor,
    video_config: dict | None = None,
    max_decode_chunk_bytes: int = MAX_VIDEO_DECODE_CHUNK_BYTES,
):
    if not isinstance(vr, VideoDecoderWrapper):
        return vr, None

    video_config = video_config or {}
    entry_time = time.perf_counter()
    total_frames, video_fps = len(vr), vr.avg_fps
    temporal_patch_size = video_config.get(
        "temporal_patch_size", video_processor.temporal_patch_size
    )
    nframes = smart_nframes(
        _sampling_config(video_processor, video_config),
        total_frames=total_frames,
        video_fps=video_fps,
        frame_factor=temporal_patch_size,
    )
    indices = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    indices = np.unique(indices)

    video = _decode_and_resize(
        vr,
        indices.tolist(),
        video_processor=video_processor,
        video_config=video_config,
        max_decode_chunk_bytes=max_decode_chunk_bytes,
    )
    metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": indices,
        "video_backend": "torchvision",
    }
    logger.debug(
        f"[Qwen3VL preprocess_video Perf], "
        f"spatial_resize: chunked_model_native, "
        f"total_time: {(time.perf_counter() - entry_time) * 1000:.2f} ms"
    )
    return video, metadata


class Qwen3VLImageProcessor(QwenVLImageProcessor):
    models = [
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5ForCausalLMMTP,
    ]

    async def _preprocess_video(self, video):
        return await preprocess_video(
            video,
            video_processor=self._processor.video_processor,
            video_config=self.video_config,
        )

    def _processor_video_config(self, video_metadata):
        config = super()._processor_video_config(video_metadata)
        if config is not None:
            for key in QWEN3_EARLY_VIDEO_CONFIG_KEYS:
                config.pop(key, None)
            config["do_resize"] = False
        return config
