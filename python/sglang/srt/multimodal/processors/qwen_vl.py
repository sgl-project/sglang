import math
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import BaseImageProcessor

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.cosmos3 import Cosmos3ForConditionalGeneration
from sglang.srt.models.interns2_mobius import (
    InternS2MobiusForConditionalGeneration,
)
from sglang.srt.models.interns2preview import InternS2PreviewForConditionalGeneration
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.multimodal.cache import resolve_multimodal_item_hash
from sglang.srt.multimodal.media_artifacts.base import (
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.srt.utils import cpu_has_amx_support, is_cpu
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.utils import logger

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        f"Ignoring and using default."
    )
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


@dataclass(frozen=True)
class QwenVLImagePreprocessArtifact:
    """Prompt-independent Qwen-VL processor output for one image."""

    content_digest: str
    artifact_key: str
    feature_hash: int
    feature: Optional[torch.Tensor]
    model_specific_data: dict[str, Any]

    @property
    def has_feature(self) -> bool:
        return self.feature is not None

    def cache_value(self) -> "QwenVLImagePreprocessArtifact":
        """Keep CPU processor outputs but never retain a CUDA tensor."""
        if self.feature is None or self.feature.device.type == "cpu":
            return self
        return replace(self, feature=None)

    def cache_size_items(self) -> tuple:
        return (
            self.content_digest,
            self.artifact_key,
            self.feature_hash,
            self.feature,
            self.model_specific_data,
        )


QWEN_VIDEO_PREPROCESS_CONFIG_KEYS = frozenset(
    {
        "fps",
        "nframes",
        "min_frames",
        "max_frames",
        "min_pixels",
        "max_pixels",
        "total_pixels",
        "resized_height",
        "resized_width",
    }
)


def _get_processor_video_config(video_config, video_metadata):
    if video_metadata and all(metadata is not None for metadata in video_metadata):
        return {
            key: value
            for key, value in video_config.items()
            if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
        }
    return None


_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
if _is_cpu and _is_cpu_amx_available:
    try:
        import transformers

        from sglang.srt.layers.amx_utils import fast_preprocess_cpu

        transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess = fast_preprocess_cpu
    except Exception as e:
        logger.warning(
            f"Failed to hack Qwen2VLImageProcessorFast with AMX optimization: {e}"
        )


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), (
        "Only accept either `fps` or `nframes`"
    )
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


# process video, qwen-specific
async def preprocess_video(
    vr,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = {},
) -> torch.Tensor:
    # preprocessed video
    is_video_obj = isinstance(vr, VideoDecoderWrapper)
    if not is_video_obj:
        return vr, None
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.avg_fps

    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    video = vr.get_frames_as_tensor(idx.tolist())

    video = video.permute(0, 3, 1, 2)  # NHWC -> TCHW

    nframes, _, height, width = video.shape
    min_pixels = video_config.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = video_config.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(
            video_config.get("max_pixels", VIDEO_MAX_PIXELS),
            total_pixels / nframes * FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )

    get_batch_time = time.perf_counter()

    max_pixels_supposed = video_config.get("max_pixels", max_pixels)

    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in video_config and "resized_width" in video_config:
        resized_height, resized_width = smart_resize(
            video_config["resized_height"],
            video_config["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    smart_resize_time = time.perf_counter()
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )
    video = video.pin_memory()
    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,
        "video_backend": "torchvision",
    }
    torchvision_resize_time = time.perf_counter()
    logger.debug(
        f"[preprocess_video Perf], "
        f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms, "
        f"smart_resize_time: {(smart_resize_time - get_batch_time) * 1000:.2f} ms, "
        f"torchvision_resize_time: {(torchvision_resize_time - smart_resize_time) * 1000:.2f} ms, "
        f"total_time: {(torchvision_resize_time - entry_time) * 1000:.2f} ms"
    )
    return video, video_metadata


# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(MediaArtifactCacheMixin, SGLangBaseProcessor):
    supports_transformers_backend = True
    generates_input_ids_from_raw_prompt = True
    artifact_modality = Modality.IMAGE
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5ForCausalLMMTP,
        InternS2PreviewForConditionalGeneration,
        InternS2MobiusForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
        Cosmos3ForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        self.uses_media_artifacts_without_cache = self.model_type in (
            "qwen3_vl",
            "qwen3_vl_moe",
        )
        if self.model_type in (
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
            "interns2_mobius",
        ):
            # Two workers overlap CPU preprocessing without over-fragmenting
            # burst arrivals into smaller GPU prefill batches. Higher counts can
            # improve short-output TTFT, but regress long-output throughput on
            # Blackwell when requests reach the scheduler too far apart.
            self.auto_mm_processor_worker_num = 2
            self.auto_mm_io_worker_num = 16
            self.supports_mm_processor_concurrency = True
        if hf_config.model_type == "qwen3_omni_moe":
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)

        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

        self._spatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        self._tokens_per_second = getattr(
            self.hf_config.vision_config, "tokens_per_second", None
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=self.VIDEO_TOKEN_ID,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    @property
    def spatial_merge_size(self):
        return self._spatial_merge_size

    def build_input_ids_with_timestamps(
        self, prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
    ):
        """
        Build input_ids with timestamps for qwen3_vl models.
        """
        if not isinstance(prompt, list):
            prompt = self._processor.tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        spatial_merge_size = self.spatial_merge_size
        vision_start_token_id = getattr(self, "vision_start_token_id", None)
        vision_end_token_id = getattr(self, "vision_end_token_id", None)

        input_ids = []
        offsets = []
        modality_list = []
        cur_idx = 0

        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))

        img_idx = 0
        video_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            modality_list.append(modality)
            video_tokens = None
            if modality == Modality.IMAGE:
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
                mm_token_id = img_token_id
                img_idx += 1
            elif modality == Modality.VIDEO:
                curr_timestamps = video_timestamps[video_idx]
                num_frames = video_grid_thw[video_idx][0]
                frame_seqlen = video_grid_thw[video_idx][1:].prod().item() // (
                    spatial_merge_size**2
                )
                video_tokens = []
                _current_offset = len(input_ids) + mm_start_idx + 1 - cur_idx
                # take single frame as one mm_item
                for frame_idx in range(num_frames):
                    if frame_idx > 0:
                        modality_list.append(Modality.VIDEO)
                    curr_time = curr_timestamps[frame_idx]
                    timestamp_text = f"<{curr_time:.1f} seconds>"
                    timestamp_tokens = self._processor.tokenizer.encode(
                        timestamp_text, add_special_tokens=False
                    )
                    video_tokens.extend(timestamp_tokens)
                    _current_offset += len(timestamp_tokens)
                    if vision_start_token_id is not None:
                        video_tokens.append(vision_start_token_id)
                        _current_offset += 1
                    video_tokens.extend([video_token_id] * frame_seqlen)
                    if vision_end_token_id is not None:
                        video_tokens.append(vision_end_token_id)
                    offsets.append(
                        (_current_offset, _current_offset + frame_seqlen - 1)
                    )
                    _current_offset += (
                        frame_seqlen + 1
                        if vision_end_token_id is not None
                        else frame_seqlen
                    )  # for vision_end_token_id
                mm_token_num = len(video_tokens)
                mm_token_id = None
                video_idx += 1
            else:
                logger.warning(
                    f"{modality} modality is not supported for qwen3_vl models with timestamps."
                )
                continue
            assert cur_idx <= mm_start_idx
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            if modality == Modality.VIDEO:
                input_ids.extend(video_tokens)
            else:
                mm_offset_start = len(input_ids)
                input_ids.extend([mm_token_id] * mm_token_num)
                offsets.append((mm_offset_start, len(input_ids) - 1))
            cur_idx = mm_start_idx + 2  # jump to vision_end_id
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def compute_mrope_positions(self, input_ids, mm_items):
        image_grid_thw = self._concat_mm_item_grid(
            mm_items, "image_grid_thw", Modality.IMAGE
        )
        video_grid_thw = self._concat_mm_item_grid(
            mm_items, "video_grid_thw", Modality.VIDEO
        )

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=self._tokens_per_second,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    @staticmethod
    def _get_processor_output_value(ret, key):
        if ret is None:
            return None
        return ret.get(key) if hasattr(ret, "get") else getattr(ret, key, None)

    def _get_precomputed_mrope_from_output(self, ret):
        mrope_positions = self._get_processor_output_value(ret, "mrope_positions")
        mrope_position_delta = self._get_processor_output_value(
            ret, "mrope_position_delta"
        )
        if mrope_positions is None or mrope_position_delta is None:
            return None

        mrope_positions = torch.as_tensor(mrope_positions)
        if mrope_positions.ndim == 3:
            if mrope_positions.shape[1] != 1:
                return None
            mrope_positions = mrope_positions.squeeze(1)
        if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:
            return None

        mrope_position_delta = torch.as_tensor(mrope_position_delta)
        if mrope_position_delta.ndim <= 1:
            mrope_position_delta = mrope_position_delta.reshape(-1, 1)
        return mrope_positions, mrope_position_delta

    @staticmethod
    def _as_grid_batch(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0) if value.ndim == 1 else value
        tensor = torch.as_tensor(value, dtype=torch.long)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def _compute_image_only_mrope_positions_from_offsets(
        self,
        input_len: int,
        mm_items: List[MultimodalDataItem],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """instead of calling get_rope_index, build mrope position from mm_items.offsets and image_grid_thw of each image
        basically a simplified version of get_rope_index for image-only reqs
        """
        if self.model_type not in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
            "interns2_mobius",
            "cosmos3_omni",
        ):
            return None

        image_items = [item for item in mm_items if item.is_image()]
        if not image_items or len(image_items) != len(mm_items):
            return None

        spatial_merge_size = self._spatial_merge_size
        sorted_items = sorted(image_items, key=lambda item: item.offsets[0][0])
        position_segments = []
        st = 0
        next_pos = 0

        for item in sorted_items:
            if item.offsets is None or len(item.offsets) != 1:
                return None

            start, end = item.offsets[0]
            if start < st or end >= input_len:
                return None

            text_len = start - st
            if text_len > 0:
                position_segments.append(
                    torch.arange(text_len, dtype=dtype, device=device)
                    .view(1, -1)
                    .expand(3, -1)
                    + next_pos
                )
                next_pos += text_len

            grid = self._as_grid_batch(item.model_specific_data.get("image_grid_thw"))
            if grid is None or grid.shape[0] != 1:
                return None
            t, h, w = [int(x) for x in grid[0].tolist()]
            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w
            if num_image_tokens != end - start + 1:
                return None

            t_index = (
                torch.arange(llm_grid_t, dtype=dtype, device=device)
                .view(-1, 1)
                .expand(llm_grid_t, llm_grid_h * llm_grid_w)
                .reshape(-1)
            )
            h_index = (
                torch.arange(llm_grid_h, dtype=dtype, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            w_index = (
                torch.arange(llm_grid_w, dtype=dtype, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            position_segments.append(
                torch.stack([t_index, h_index, w_index]) + next_pos
            )
            next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)
            st = end + 1

        if st < input_len:
            text_len = input_len - st
            position_segments.append(
                torch.arange(text_len, dtype=dtype, device=device)
                .view(1, -1)
                .expand(3, -1)
                + next_pos
            )

        mrope_positions = torch.cat(position_segments, dim=1).unsqueeze(1)
        mrope_position_delta = (mrope_positions.max() + 1 - input_len).reshape(1, 1)
        return mrope_positions, mrope_position_delta

    @classmethod
    def _concat_mm_item_grid(cls, mm_items: list[MultimodalDataItem], key, modality):
        grids = []
        for item in mm_items:
            if not item.is_modality(modality):
                continue
            grid = cls._as_grid_batch(item.model_specific_data.get(key))
            if grid is not None:
                grids.append(grid)
        if not grids:
            return None
        if len(grids) == 1:
            return grids[0]
        return torch.cat(grids, dim=0)

    @classmethod
    def _get_grid_from_output_or_items(
        cls, ret, mm_items, key, modality, input_data=None
    ):
        grid = cls._get_processor_output_value(ret, key)
        if grid is None:
            grid = cls._concat_mm_item_grid(mm_items, key, modality)
        if grid is None and input_data and isinstance(input_data[0], dict):
            grid = input_data[0].get(key)
        return grid

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)
        video_timestamps = kwargs.get("video_timestamps", None)
        second_per_grid_ts = kwargs.get("second_per_grid_ts", None)

        audio_seq_lens = None
        if audio_feature_lens is not None:
            if self.model_type == "qwen3_omni_moe":
                # apply _get_feat_extract_lengths to get seq_lens
                input_lengths_leave = audio_feature_lens % 100
                feat_lengths = (input_lengths_leave - 1) // 2 + 1
                audio_seq_lens = (
                    ((feat_lengths - 1) // 2 + 1 - 1) // 2
                    + 1
                    + (audio_feature_lens // 100) * 13
                )
            elif self.model_type == "qwen2_5_omni":
                audio_seq_lens = (audio_feature_lens - 1) // 2 + 1
                audio_seq_lens = (audio_seq_lens - 2) // 2 + 1

        if (
            self.model_type
            in [
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
                "intern_s2_preview",
                "cosmos3_omni",
            ]
            and video_timestamps is not None
        ):
            input_ids, offsets, modality_list = self.build_input_ids_with_timestamps(
                prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
            )
        else:
            input_ids, offsets, modality_list = self.build_input_ids(
                prompt, img_grid_thw, video_grid_thw, audio_seq_lens=audio_seq_lens
            )
        assert all(isinstance(modality, Modality) for modality in modality_list)

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            image_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=(
                audio_feature_lens if self.model_type == "qwen3_omni_moe" else None
            ),
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, "position_id_per_seconds", None
            ),
            tokens_per_second=self._tokens_per_second,
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_items = []
        consumed_per_modality = {}

        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            embedding_start = consumed_per_modality.get(modality, 0)
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    def prepare_artifact_batch(
        self,
        entries: list[MediaArtifactInput],
        *,
        processor=None,
    ) -> list[QwenVLImagePreprocessArtifact]:
        """Preprocess image cache misses without retaining prompt-specific state."""
        if not entries:
            return []

        processor, _ = self._resolve_processor(processor)
        image_kwargs = dict(self.image_config or {})
        processor_device = None
        if (
            isinstance(processor.image_processor, BaseImageProcessor)
            and not self.disable_fast_image_processor
        ):
            processor_device = self._fast_image_processor_device(processor)
            if processor_device is not None:
                image_kwargs["device"] = processor_device

        with self._temporary_fast_processor_cuda_pool(processor_device):
            result = processor.image_processor(
                images=[entry.media for entry in entries],
                return_tensors="pt",
                **image_kwargs,
            )
            features = self._get_processor_output_value(result, "pixel_values")
            image_grid_thw = self._get_processor_output_value(result, "image_grid_thw")
            if (
                isinstance(features, torch.Tensor)
                and (
                    self.mm_preprocess_cache.enabled
                    or not self.keep_mm_features_on_device
                )
                and not self.precompute_hash_before_cpu_transfer
            ):
                features = features.cpu()

        if not isinstance(features, torch.Tensor):
            raise TypeError("Qwen-VL image processor must return pixel_values")
        if not isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = torch.as_tensor(image_grid_thw, dtype=torch.long)
        if image_grid_thw.ndim != 2 or image_grid_thw.shape != (len(entries), 3):
            raise ValueError(
                "Qwen-VL image processor returned an invalid image_grid_thw shape: "
                f"{tuple(image_grid_thw.shape)}"
            )
        feature_lengths = image_grid_thw.prod(dim=1).tolist()
        if sum(feature_lengths) != features.shape[0]:
            raise ValueError(
                "Qwen-VL image processor feature count does not match image grids: "
                f"{features.shape[0]} != {sum(feature_lengths)}"
            )

        artifacts = []
        feature_offset = 0
        for entry, grid, feature_length in zip(
            entries, image_grid_thw, feature_lengths
        ):
            feature = features[
                feature_offset : feature_offset + feature_length
            ].contiguous()
            feature_offset += feature_length
            # The artifact key already commits to the media content, processor
            # fingerprint, and every preprocessing kwarg. Derive the downstream
            # cache identity from it so independently preprocessed copies in
            # different tokenizer workers share the same radix/VLM cache key.
            feature_hash = resolve_multimodal_item_hash(
                existing_hash=0, namespace=entry.artifact_key
            )
            artifacts.append(
                QwenVLImagePreprocessArtifact(
                    content_digest=entry.content_digest,
                    artifact_key=entry.artifact_key,
                    feature_hash=feature_hash,
                    feature=feature,
                    model_specific_data={"image_grid_thw": grid.unsqueeze(0)},
                )
            )
        return artifacts

    def compose_image_artifacts(
        self,
        input_text,
        artifacts: list[QwenVLImagePreprocessArtifact],
    ) -> MultimodalProcessorOutput:
        """Compose prompt tokens and request-owned items from cached images."""
        image_grids = []
        for artifact in artifacts:
            grid = self._as_grid_batch(
                artifact.model_specific_data.get("image_grid_thw")
            )
            if grid is None or grid.shape[0] != 1:
                raise ValueError("Each Qwen-VL image artifact requires one image grid")
            image_grids.append(grid)
        image_grid_thw = torch.cat(image_grids, dim=0)

        grid_key = tuple(
            tuple(int(value) for value in row.tolist()) for row in image_grid_thw
        )
        if isinstance(input_text, str):
            (
                input_ids_tuple,
                offsets,
                mrope_positions,
                mrope_position_delta,
            ) = self._cached_image_prompt_template(input_text, grid_key)
        else:
            (
                input_ids_tuple,
                offsets,
                mrope_positions,
                mrope_position_delta,
            ) = self._build_image_prompt_template(input_text, grid_key)
        input_ids_list = list(input_ids_tuple)

        mm_items = []
        for artifact, offset in zip(artifacts, offsets):
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=artifact.feature,
                offsets=[offset],
                model_specific_data=deepcopy(artifact.model_specific_data),
            )
            item.set_hash(artifact.feature_hash)
            mm_items.append(item)

        padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(
            input_ids_list, mm_items
        )
        mm_items = self._prepare_mm_items_for_transport(mm_items)
        self._mark_cuda_ipc_features_for_deferred_reconstruction(mm_items)
        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=padded_input_ids,
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions.clone(),
            mrope_position_delta=mrope_position_delta.clone(),
        )

    @lru_cache(maxsize=256)
    def _cached_image_prompt_template(self, input_text: str, grid_key: tuple):
        """Cache prompt expansion and M-RoPE by prompt and image-grid shape."""
        return self._build_image_prompt_template(input_text, grid_key)

    def _build_image_prompt_template(self, input_text, grid_key: tuple):
        image_grid_thw = torch.tensor(grid_key, dtype=torch.long)
        input_ids_list, offsets, modalities = self.build_input_ids(
            input_text, img_grid_thw=image_grid_thw
        )
        if modalities != [Modality.IMAGE] * len(grid_key):
            raise ValueError("Qwen-VL image artifacts do not match prompt placeholders")

        template_items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=None,
                offsets=[offset],
                model_specific_data={"image_grid_thw": grid.unsqueeze(0)},
            )
            for grid, offset in zip(image_grid_thw, offsets)
        ]
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        mrope_result = self._compute_image_only_mrope_positions_from_offsets(
            input_len=input_ids.numel(),
            mm_items=template_items,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        if mrope_result is None:
            mrope_result = self.compute_mrope_positions(input_ids_list, template_items)
        mrope_positions, mrope_position_delta = mrope_result
        if mrope_positions is not None and mrope_positions.ndim == 3:
            mrope_positions = mrope_positions.squeeze(1)
        return (
            tuple(input_ids_list),
            tuple(offsets),
            mrope_positions,
            mrope_position_delta,
        )

    async def _process_mm_data_uncached(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        entry_time = time.perf_counter()
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        load_time = time.perf_counter()
        rid = getattr(request_obj, "rid", "anonymous_rid")

        video_metadata = None
        if base_output.videos and not isinstance(base_output.videos[0], dict):
            videos_processed = [
                await preprocess_video(video, video_config=self.video_config)
                for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        preprocess_time = time.perf_counter()

        processor_kwargs = {}
        processor_video_config = _get_processor_video_config(
            self.video_config, video_metadata
        )
        if processor_video_config is not None:
            processor_kwargs["processor_video_config"] = processor_video_config

        # NOTE: for qwen3-vl, video_meta need to be passed in, since do_sample_frames is already done in preprocess_video
        if self.hf_config.model_type in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
            "interns2_mobius",
            "cosmos3_omni",
        ):
            processor_kwargs.update(
                video_metadata=video_metadata,
                do_sample_frames=False,
            )

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens, **processor_kwargs
        )

        self._mark_cuda_ipc_features_for_deferred_reconstruction(mm_items)

        audio_feature_lengths = None

        if self.model_type == "qwen3_omni_moe":
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
            if audio_item:
                audio_feature_lengths = torch.sum(
                    audio_item.feature_attention_mask, dim=1
                )

        second_per_grid_ts = self._get_processor_output_value(ret, "second_per_grid_ts")
        if second_per_grid_ts is None:
            second_per_grid_ts = self._get_processor_output_value(
                ret, "video_second_per_grid"
            )

        process_time = time.perf_counter()

        input_ids = input_ids.flatten()
        base_input_ids = getattr(base_output, "input_ids", None)
        if (
            isinstance(base_input_ids, list)
            and len(base_input_ids) == input_ids.numel()
        ):
            # reuse preprocess input if it already carries list of input_ids
            input_ids_list = base_input_ids
        else:
            input_ids_list = input_ids.tolist()

        # look for if padded_input_ids already exists before computing
        padded_input_ids = self._get_processor_output_value(ret, "padded_input_ids")
        if padded_input_ids is None:
            padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(
                input_ids_list, mm_items
            )
        elif isinstance(padded_input_ids, torch.Tensor):
            # reuse existing padded_input_ids
            padded_input_ids = padded_input_ids.flatten().tolist()
        else:
            padded_input_ids = list(padded_input_ids)

        image_grid_thw = self._get_grid_from_output_or_items(
            ret, mm_items, "image_grid_thw", Modality.IMAGE, image_data
        )
        video_grid_thw = self._get_grid_from_output_or_items(
            ret,
            mm_items,
            "video_grid_thw",
            Modality.VIDEO,
            request_obj.video_data,
        )

        mrope_result = self._get_precomputed_mrope_from_output(ret)
        if mrope_result is None:
            if (
                video_grid_thw is None
                and second_per_grid_ts is None
                and audio_feature_lengths is None
            ):
                mrope_result = self._compute_image_only_mrope_positions_from_offsets(
                    input_len=input_ids.numel(),
                    mm_items=mm_items,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
        if mrope_result is None:
            mrope_result = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self._spatial_merge_size,
                image_token_id=self.mm_tokens.image_token_id,
                video_token_id=self.mm_tokens.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                tokens_per_second=self._tokens_per_second,
                # use the expanded token ids
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                use_audio_in_video=False,
                audio_seqlens=audio_feature_lengths,
                audio_token_id=getattr(self.hf_config, "audio_token_id", None),
                audio_start_token_id=self.audio_start_token_id,
                position_id_per_seconds=getattr(
                    self.hf_config, "position_id_per_seconds", None
                ),
            )

        mrope_positions, mrope_position_delta = mrope_result
        if mrope_positions.ndim == 3:
            mrope_positions = mrope_positions.squeeze(1)
        get_rope_index_time = time.perf_counter()
        logger.debug(
            f"[QwenVLProcessor Perf] {rid=}, "
            f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "
            f"preprocess_time: {(preprocess_time - load_time) * 1000:.2f} ms, "
            f"process_time: {(process_time - preprocess_time) * 1000:.2f} ms, "
            f"get_rope_index_time: {(get_rope_index_time - process_time) * 1000:.2f} ms, "
            f"total_time: {(get_rope_index_time - entry_time) * 1000:.2f} ms"
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=padded_input_ids,
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if (
            not image_data
            or request_obj.video_data
            or request_obj.audio_data
            or any(self._is_preprocessed_input(item) for item in image_data)
            or (
                not self.mm_preprocess_cache.enabled
                and not self.uses_media_artifacts_without_cache
            )
        ):
            return await self._process_mm_data_uncached(
                image_data, input_text, request_obj, *args, **kwargs
            )

        prepare_artifacts = (
            self.prepare_media_artifacts
            if self.mm_preprocess_cache.enabled
            else self.prepare_media_artifacts_without_cache
        )
        artifacts = await prepare_artifacts(
            image_data, content_hashes=getattr(request_obj, "mm_content_hashes", None)
        )
        return self.compose_image_artifacts(input_text, artifacts)

    def _mark_cuda_ipc_features_for_deferred_reconstruction(self, mm_items):
        supports_deferred_reconstruction = self.server_args.mm_enable_dp_encoder or (
            self.server_args.tp_size == 1
            and self.model_type in ("qwen3_vl", "qwen3_vl_moe")
        )
        if not (
            self.keep_mm_features_on_device
            and supports_deferred_reconstruction
            and self.model_type
            in ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe")
        ):
            return
        for item in mm_items:
            if item.is_image() or item.is_video():
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
