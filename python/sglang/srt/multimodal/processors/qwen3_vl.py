"""Multimodal preprocessing for Qwen3-VL and Qwen3.5."""

import time
from typing import Optional

import numpy as np

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


async def preprocess_video(vr, video_config: Optional[dict] = None):
    # Spatial resize stays with the model-native processor; pre-resizing here
    # (as the Qwen2 path does) double-resizes with the wrong geometry.
    if not isinstance(vr, VideoDecoderWrapper):
        return vr, None

    video_config = video_config or {}
    entry_time = time.perf_counter()
    total_frames, video_fps = len(vr), vr.avg_fps
    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )
    indices = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    indices = np.unique(indices)

    video = vr.get_frames_as_tensor(indices.tolist())
    video = video.permute(0, 3, 1, 2).pin_memory()
    metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": indices,
        "video_backend": "torchvision",
    }
    logger.debug(
        f"[Qwen3VL preprocess_video Perf], "
        f"spatial_resize: downstream_processor, "
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
        return await preprocess_video(video, video_config=self.video_config)
