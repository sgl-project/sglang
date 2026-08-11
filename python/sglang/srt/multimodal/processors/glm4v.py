import json
import math
from typing import List, Union

import numpy as np
import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)

try:
    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None

try:
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
except ImportError:
    Glm5NextForConditionalGeneration = None


GLM_VIDEO_DEFAULT_FPS = 2.0
GLM_VIDEO_DEFAULT_MAX_FRAMES = 2048
GLM_VIDEO_PATCH_SIZE = 14
GLM_VIDEO_MERGE_SIZE = 2
GLM_VIDEO_PATCH_EXPAND_FACTOR = 4


def _glm_video_metadata(total_num_frames, fps, duration, frames_indices):
    return {
        "total_num_frames": int(total_num_frames),
        "fps": float(fps),
        "duration": float(duration),
        "video_backend": "sglang",
        "frames_indices": list(frames_indices),
    }


def split_glm_video_items(mm_data):
    """Extract per-video sampling overrides from URL dictionaries."""
    items = mm_data if isinstance(mm_data, (list, tuple)) else [mm_data]
    urls, configs = [], []
    for item in items:
        if isinstance(item, dict) and "format" not in item and "url" in item:
            urls.append(item["url"])
            configs.append(
                {
                    key: item[key]
                    for key in (
                        "fps",
                        "max_frames",
                        "max_tokens_per_frame",
                        "max_image_tokens",
                    )
                    if item.get(key) is not None
                }
            )
        else:
            urls.append(item)
            configs.append({})
    return urls, configs


def glm_sample_frame_indices(
    total_frames,
    fps,
    duration,
    *,
    target_fps=None,
    max_frame_count=None,
):
    """Sample GLM video frames deterministically and keep temporal pairs.

    The algorithm mirrors the GLM/HF processor rather than treating target
    timestamps as independent nearest-neighbour lookups.  That keeps the
    no-override EPD path byte-for-byte aligned with ordinary preprocessing.
    """
    if total_frames <= 0:
        return []
    target_fps = GLM_VIDEO_DEFAULT_FPS if target_fps is None else float(target_fps)
    max_frame_count = (
        GLM_VIDEO_DEFAULT_MAX_FRAMES
        if max_frame_count is None
        else int(max_frame_count)
    )
    if target_fps <= 0 or max_frame_count <= 0:
        return []

    max_frame_idx = total_frames - 1
    if not duration:
        duration = round(max_frame_idx / fps) + 1 if fps else 0
    extract_t = min(int(duration * target_fps), int(max_frame_count))
    extract_t = max(1, extract_t)

    if fps:
        duration_per_frame = 1 / fps
        timestamps = [index * duration_per_frame for index in range(total_frames)]
        max_second = int(duration)
        indices = []
        current_second = 0.0
        interval = 1 / target_fps
        for frame_index, timestamp in enumerate(timestamps):
            if timestamp >= current_second:
                current_second += interval
                indices.append(frame_index)
                if current_second >= max_second:
                    break
    else:
        indices = []

    if len(indices) < extract_t:
        start = indices[0] if indices else 0
        end = indices[-1] if indices else max(total_frames - 1, 0)
        indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(indices) > extract_t:
        indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

    seen = set()
    unique_indices = []
    for index in indices:
        index = int(index)
        if index not in seen:
            seen.add(index)
            unique_indices.append(index)
    if len(unique_indices) & 1:
        unique_indices.append(unique_indices[-1])
    return unique_indices


def _resize_frames_to_max_tokens(frames, max_tokens_per_frame):
    """Downscale NHWC frames to the GLM per-frame post-merge token budget."""
    import torchvision.transforms.functional as TF

    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(np.asarray(frames))
    nchw = frames.permute(0, 3, 1, 2)
    _, _, height, width = nchw.shape
    pixels_per_token = (GLM_VIDEO_PATCH_SIZE * GLM_VIDEO_MERGE_SIZE) ** 2
    factor = GLM_VIDEO_PATCH_SIZE * GLM_VIDEO_MERGE_SIZE * GLM_VIDEO_PATCH_EXPAND_FACTOR
    max_pixels = max(
        int(max_tokens_per_frame) * pixels_per_token,
        factor * factor,
    )
    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    if (resized_height, resized_width) != (height, width):
        nchw = TF.resize(
            nchw,
            [resized_height, resized_width],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
    return nchw.permute(0, 2, 3, 1).contiguous()


def preprocess_video_frames_sync(frame_list: List[dict]):
    """Convert a client-provided GLM frame list to processor input."""
    total_num_frames = len(frame_list)
    if total_num_frames == 0:
        raise ValueError("GLM video frame list must not be empty")
    duration = 0.0
    if frame_list[0].get("detail") is not None:
        details = json.loads(frame_list[0]["detail"])
        duration = float(details.get("video_duration", 0))
    if duration == 0:
        duration = float(frame_list[-1]["timestamp"])
    images = [frame["frame_image"] for frame in frame_list]
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images).permute(0, 2, 3, 1).contiguous()
    else:
        images = [np.asarray(image) for image in images]
    fps = total_num_frames / duration if duration else 0
    return images, _glm_video_metadata(
        total_num_frames, fps, duration, range(total_num_frames)
    )


def glm_decode_frames_at(vr, indices, video_config=None):
    """Decode only an explicitly assigned frame subset for one encoder rank."""
    indices = list(indices)
    if not indices:
        return None
    video_config = video_config or {}
    # VideoDecoderWrapper already uses parallel extraction where supported.
    if hasattr(vr, "get_frames_as_tensor"):
        frames = vr.get_frames_as_tensor(indices)
    else:
        frames = vr.get_frames_at(indices)
    max_tokens_per_frame = video_config.get("max_tokens_per_frame")
    if max_tokens_per_frame is not None:
        frames = _resize_frames_to_max_tokens(frames, max_tokens_per_frame)
    return frames


def glm_sample_and_decode_sync(vr, video_config=None):
    video_config = video_config or {}
    fps = vr.avg_fps
    duration = len(vr) / fps if fps else 0
    indices = glm_sample_frame_indices(
        len(vr),
        fps,
        duration,
        target_fps=video_config.get("fps"),
        max_frame_count=video_config.get("max_frames"),
    )
    frames = glm_decode_frames_at(vr, indices, video_config)
    return frames, _glm_video_metadata(len(vr), fps, duration, indices)


class Glm4vImageProcessor(SGLangBaseProcessor):
    models = [
        m
        for m in [
            Glm4vForConditionalGeneration,
            Glm4vMoeForConditionalGeneration,
            Glm5NextForConditionalGeneration,
            GlmOcrForConditionalGeneration,
        ]
        if m is not None
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # GLM-V specific tokens
        self.IMAGE_TOKEN = "<|image|>"
        self.VIDEO_TOKEN = "<|video|>"
        self.IMAGE_START_TOKEN = "<|begin_of_image|>"
        self.IMAGE_END_TOKEN = "<|end_of_image|>"
        self.VIDEO_START_TOKEN = "<|begin_of_video|>"
        self.VIDEO_END_TOKEN = "<|end_of_video|>"

        # Token IDs
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id
        self.VIDEO_START_TOKEN_ID = hf_config.video_start_token_id
        self.VIDEO_END_TOKEN_ID = hf_config.video_end_token_id
        self.IM_START_TOKEN_ID = self.IMAGE_START_TOKEN_ID
        self.IM_END_TOKEN_ID = self.IMAGE_END_TOKEN_ID

        # Vision config
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 112 * 112
        self.MAX_PIXELS = 30000 * 28 * 28 * 2

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            video_token=self.VIDEO_TOKEN,
            # Note: For GLM4v videos, it uses the video token before tokenization but uses image token after tokenization
            video_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    def compute_mrope_positions(self, input_ids, mm_items):
        image_grid_thw = None
        video_grid_thw = None
        for item in mm_items:
            if "image_grid_thw" in item.model_specific_data:
                image_grid_thw = item.model_specific_data["image_grid_thw"]
            if "video_grid_thw" in item.model_specific_data:
                video_grid_thw = item.model_specific_data["video_grid_thw"]

        import torch

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids_tensor)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids_tensor,
            hf_config=self.hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        if base_output.videos:
            base_output.videos = request_obj.video_data
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            attention_mask=getattr(ret, "attention_mask", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
