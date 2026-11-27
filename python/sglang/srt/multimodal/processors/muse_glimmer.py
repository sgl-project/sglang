# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SGLang multimodal processor for Muse Glimmer."""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.muse_glimmer import MuseGlimmerForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

logger = logging.getLogger(__name__)

_VIDEO_SAMPLING_KEYS = frozenset(
    {"fps", "num_frames", "max_frames", "do_sample_frames"}
)
_DEFAULT_VIDEO_FPS = 24.0


def _valid_fps(fps: Optional[float]) -> Optional[float]:
    if fps is None:
        return None
    fps_val = float(fps)
    if not math.isfinite(fps_val) or fps_val <= 0:
        return None
    return fps_val


def _frame_to_chw_tensor(frame: Any) -> torch.Tensor:
    if isinstance(frame, Image.Image):
        arr = np.array(frame.convert("RGB"), dtype=np.uint8, copy=True)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if isinstance(frame, np.ndarray):
        arr = frame if frame.flags.writeable else frame.copy()
        tensor = torch.from_numpy(arr)
    elif isinstance(frame, torch.Tensor):
        tensor = frame.cpu() if frame.is_cuda else frame
    else:
        raise TypeError(f"Unsupported video frame type: {type(frame).__name__}")

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected 3D video frame tensor, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor


def _frames_to_tchw_tensor(frames: Any) -> torch.Tensor:
    if isinstance(frames, list):
        if not frames:
            raise ValueError("Video frame list cannot be empty.")
        return torch.stack([_frame_to_chw_tensor(f) for f in frames], dim=0)
    if isinstance(frames, np.ndarray):
        arr = frames if frames.flags.writeable else frames.copy()
        tensor = torch.from_numpy(arr)
    elif isinstance(frames, torch.Tensor):
        tensor = frames.cpu() if frames.is_cuda else frames
    else:
        raise TypeError(f"Unsupported video frames type: {type(frames).__name__}")

    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3, 4) and tensor.shape[1] not in (1, 3, 4):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


def _extract_video_sampling_overrides(
    video_data: Optional[Any],
) -> List[Dict[str, Any]]:
    if not video_data:
        return []
    from sglang.srt.managers.io_struct import VideoData

    items = video_data if isinstance(video_data, list) else [video_data]
    overrides: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, VideoData) and item.preprocess_kwargs:
            overrides.append(dict(item.preprocess_kwargs))
        else:
            overrides.append({})
    return overrides


def _sample_and_decode_video_sync(
    video: Any,
    video_processor: Any,
    sampling_config: Dict[str, Any],
) -> Tuple[torch.Tensor, VideoMetadata]:
    """Decode/normalize a single video into a (T, C, H, W) tensor and VideoMetadata."""
    do_sample_frames = bool(
        sampling_config.get(
            "do_sample_frames",
            getattr(video_processor, "do_sample_frames", True),
        )
    )
    target_fps = sampling_config.get("fps", getattr(video_processor, "fps", 2.0))
    target_num_frames = sampling_config.get(
        "num_frames", getattr(video_processor, "num_frames", 96)
    )
    max_frames = sampling_config.get("max_frames")
    if max_frames is not None:
        target_num_frames = (
            min(target_num_frames, max_frames)
            if target_num_frames is not None
            else max_frames
        )
    temporal_patch_size = sampling_config.get(
        "temporal_patch_size",
        getattr(video_processor, "temporal_patch_size", 2),
    )

    if isinstance(video, VideoDecoderWrapper):
        total_frames = len(video)
        if total_frames <= 0:
            raise ValueError("Video must contain at least one frame.")
        source_fps = _valid_fps(video.avg_fps)
        if source_fps is None:
            logger.warning(
                "Could not infer valid fps from video decoder; defaulting to %.1f.",
                _DEFAULT_VIDEO_FPS,
            )
            source_fps = _DEFAULT_VIDEO_FPS

        metadata = VideoMetadata(
            total_num_frames=total_frames,
            fps=source_fps,
            duration=total_frames / source_fps,
        )
        if do_sample_frames:
            indices = video_processor.sample_frames(
                metadata,
                temporal_patch_size=temporal_patch_size,
                num_frames=target_num_frames,
                fps=target_fps,
            ).tolist()
        else:
            indices = list(range(total_frames))

        if hasattr(video, "get_frames_as_tensor"):
            raw_frames = video.get_frames_as_tensor(indices)
        else:
            raw_frames = video.get_frames_at(indices)

        if isinstance(raw_frames, np.ndarray):
            frames = torch.from_numpy(raw_frames)
        elif isinstance(raw_frames, torch.Tensor):
            frames = raw_frames.cpu() if raw_frames.is_cuda else raw_frames
        else:
            frames = torch.as_tensor(raw_frames)
        # VideoDecoderWrapper returns NHWC uint8 frames.
        frames = frames.permute(0, 3, 1, 2).contiguous()

        metadata.frames_indices = indices
        metadata.height = int(frames.shape[-2])
        metadata.width = int(frames.shape[-1])
        return frames, metadata

    if isinstance(video, tuple) and len(video) == 2:
        raw_frames, raw_metadata = video
        frames = _frames_to_tchw_tensor(raw_frames)
        if isinstance(raw_metadata, VideoMetadata):
            metadata = raw_metadata
        elif isinstance(raw_metadata, dict):
            metadata = VideoMetadata(**raw_metadata)
        else:
            raise TypeError(
                f"Unsupported video metadata type: {type(raw_metadata).__name__}"
            )
        source_fps = _valid_fps(metadata.fps) or _DEFAULT_VIDEO_FPS
        metadata.fps = source_fps
        if metadata.total_num_frames is None:
            metadata.total_num_frames = int(frames.shape[0])
        if (
            do_sample_frames
            and metadata.frames_indices is None
            and int(frames.shape[0]) == metadata.total_num_frames
        ):
            indices = video_processor.sample_frames(
                metadata,
                temporal_patch_size=temporal_patch_size,
                num_frames=target_num_frames,
                fps=target_fps,
            ).tolist()
            frames = frames[indices]
            metadata.frames_indices = indices
        elif metadata.frames_indices is None:
            metadata.frames_indices = list(range(int(frames.shape[0])))
        metadata.height = int(frames.shape[-2])
        metadata.width = int(frames.shape[-1])
        return frames, metadata

    if isinstance(video, list):
        # Pre-extracted frame lists are preserved without resampling so caller-selected
        # frames are not dropped; assign target_fps for monotonic timestamps.
        frames = _frames_to_tchw_tensor(video)
        total_frames = int(frames.shape[0])
        effective_fps = _valid_fps(target_fps) or _DEFAULT_VIDEO_FPS
        metadata = VideoMetadata(
            total_num_frames=total_frames,
            fps=effective_fps,
            duration=total_frames / effective_fps,
            height=int(frames.shape[-2]),
            width=int(frames.shape[-1]),
            frames_indices=list(range(total_frames)),
        )
        return frames, metadata

    frames = _frames_to_tchw_tensor(video)
    total_frames = int(frames.shape[0])
    if total_frames <= 0:
        raise ValueError("Video must contain at least one frame.")
    source_fps = _DEFAULT_VIDEO_FPS
    metadata = VideoMetadata(
        total_num_frames=total_frames,
        fps=source_fps,
        duration=total_frames / source_fps,
    )
    if do_sample_frames:
        indices = video_processor.sample_frames(
            metadata,
            temporal_patch_size=temporal_patch_size,
            num_frames=target_num_frames,
            fps=target_fps,
        ).tolist()
        frames = frames[indices]
    else:
        indices = list(range(total_frames))
    metadata.frames_indices = indices
    metadata.height = int(frames.shape[-2])
    metadata.width = int(frames.shape[-1])
    return frames, metadata


class MuseGlimmerMultimodalProcessor(SGLangBaseProcessor):
    models = [MuseGlimmerForConditionalGeneration]
    video_preprocessing_device = "cpu"

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.image_token_id = _processor.image_token_id
        self.image_start_token_id = getattr(_processor, "image_start_token_id", None)
        self.image_end_token_id = getattr(_processor, "image_end_token_id", None)
        self.video_token_id = getattr(_processor, "video_token_id", None)
        if self.video_token_id is None:
            self.video_token_id = getattr(hf_config, "video_token_id", None)

        video_token = getattr(_processor, "video_token", "<|video|>")
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.image_token,
            image_token_id=self.image_token_id,
            video_token=video_token,
            video_token_id=self.video_token_id,
        ).build(_processor)

    def _expand_input_ids(
        self,
        original_ids: List[int],
        counts: List[int],
        placeholder_token_id: int,
    ) -> List[int]:
        if (
            placeholder_token_id != self.image_token_id
            or not isinstance(self.image_start_token_id, int)
            or not isinstance(self.image_end_token_id, int)
        ):
            return super()._expand_input_ids(original_ids, counts, placeholder_token_id)

        result: List[int] = []
        count_iter = iter(counts)
        for token_id in original_ids:
            if token_id == placeholder_token_id:
                try:
                    count = next(count_iter)
                except StopIteration:
                    raise ValueError(
                        "More image placeholders in input_ids than resolved counts"
                    ) from None
                result.append(self.image_start_token_id)
                result.extend([placeholder_token_id] * count)
                result.append(self.image_end_token_id)
            else:
                result.append(token_id)

        try:
            next(count_iter)
            raise ValueError(
                "Fewer image placeholders in input_ids than resolved counts"
            )
        except StopIteration:
            pass
        return result

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        video_data = kwargs.get("video_data")
        if video_data is None and request_obj is not None:
            video_data = getattr(request_obj, "video_data", None)

        video_overrides = _extract_video_sampling_overrides(video_data)

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )

        combine_kwargs: Dict[str, Any] = {}
        raw_video_positions = [
            idx
            for idx, item in enumerate(base_output.videos or [])
            if not isinstance(item, dict)
        ]

        if raw_video_positions:
            video_processor = getattr(self._processor, "video_processor", None)
            if video_processor is None:
                raise RuntimeError(
                    "MuseGlimmer processor is missing `video_processor`; cannot "
                    "process video inputs."
                )

            loop = asyncio.get_running_loop()
            raw_videos = [base_output.videos[idx] for idx in raw_video_positions]
            try:
                tasks = []
                for pos, video in zip(raw_video_positions, raw_videos):
                    sampling_config = dict(self.video_config)
                    if pos < len(video_overrides):
                        sampling_config.update(video_overrides[pos])
                    tasks.append(
                        loop.run_in_executor(
                            self.io_executor,
                            _sample_and_decode_video_sync,
                            video,
                            video_processor,
                            sampling_config,
                        )
                    )
                decoded = await asyncio.gather(*tasks)
            finally:
                for video in raw_videos:
                    if isinstance(video, VideoDecoderWrapper):
                        video.close()

            video_metadata: List[VideoMetadata] = []
            for pos, (frames, metadata) in zip(raw_video_positions, decoded):
                base_output.videos[pos] = frames
                video_metadata.append(metadata)

            combine_kwargs["video_metadata"] = video_metadata
            combine_kwargs["do_sample_frames"] = False
            combine_kwargs["processor_video_config"] = {
                k: v
                for k, v in self.video_config.items()
                if k not in _VIDEO_SAMPLING_KEYS
            }

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens, **combine_kwargs
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )
