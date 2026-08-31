import asyncio
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
from sglang.srt.utils import GLM_MEDIA_CONFIG_KEYS
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

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


def _glm_item_config(item):
    config = dict(getattr(item, "preprocess_kwargs", None) or {})
    if isinstance(item, dict):
        config.update(item.get("preprocess_kwargs") or {})
        config.update(
            {
                key: item[key]
                for key in GLM_MEDIA_CONFIG_KEYS
                if item.get(key) is not None
            }
        )
    return {
        key: config[key] for key in GLM_MEDIA_CONFIG_KEYS if config.get(key) is not None
    }


def split_glm_video_items(mm_data):
    if mm_data is None:
        return None, []
    items = mm_data if isinstance(mm_data, (list, tuple)) else [mm_data]
    urls, configs = [], []
    for item in items:
        if isinstance(item, dict) and "format" not in item and "url" in item:
            urls.append(item["url"])
        elif hasattr(item, "url") and hasattr(item, "preprocess_kwargs"):
            urls.append(item.url)
        else:
            urls.append(item)
        configs.append(_glm_item_config(item))
    return urls, configs


def glm_budget_kwargs(processor, user_max_image_tokens=None, count=1, split=False):
    if processor is None:
        return None
    default_max = getattr(processor, "max_image_tokens", None)
    if not default_max:
        return None

    if user_max_image_tokens is not None:
        budget = int(user_max_image_tokens)
    elif split:
        budget = int(default_max)
    else:
        return None

    count = max(int(count or 1), 1)
    effective = max(1, budget // count if split and count > 1 else budget)
    if effective == default_max and user_max_image_tokens is None:
        return None
    return {"max_image_tokens": effective}


def glm_max_image_tokens_from_configs(configs):
    values = [
        int(config["max_image_tokens"])
        for config in configs or []
        if isinstance(config, dict) and config.get("max_image_tokens") is not None
    ]
    return min(values) if values else None


def glm_processor_video_config(processor):
    if processor is None:
        return {}
    config = {
        key: value
        for key in GLM_MEDIA_CONFIG_KEYS
        if (value := getattr(processor, key, None)) is not None
    }
    budget = _glm_processor_resize_budget(processor)
    if budget is not None:
        config["_presize_budget"] = budget
    return config


def _glm_processor_resize_budget(processor):
    """Use token limits because Glm5NextVideoProcessor.size.longest_edge is only a sentinel."""
    max_image_tokens = getattr(processor, "max_image_tokens", None)
    if not max_image_tokens:
        return None
    patch_size = getattr(processor, "patch_size", None) or GLM_VIDEO_PATCH_SIZE
    merge_size = getattr(processor, "merge_size", None) or GLM_VIDEO_MERGE_SIZE
    expand_factor = getattr(processor, "patch_expand_factor", None) or 1
    temporal_factor = getattr(processor, "temporal_patch_size", None) or 2
    pixels_per_token = int(temporal_factor * (patch_size * merge_size) ** 2)
    return {
        "factor": int(patch_size * merge_size * expand_factor),
        "temporal_factor": int(temporal_factor),
        "pixels_per_token": pixels_per_token,
        "min_pixels": int(getattr(processor, "min_image_tokens", None) or 0)
        * pixels_per_token,
        "max_pixels": int(max_image_tokens) * pixels_per_token,
        "resize_mode": getattr(processor, "resize_mode", None) or "resize",
    }


def _glm_effective_presize_budget(video_config, effective_max_image_tokens):
    budget = video_config.get("_presize_budget") if video_config else None
    if not budget or effective_max_image_tokens is None:
        return video_config
    config = dict(video_config)
    config["_presize_budget"] = {
        **budget,
        "max_pixels": int(effective_max_image_tokens) * budget["pixels_per_token"],
    }
    return config


def _merge_glm_video_configs(default_config, item_configs):
    defaults = dict(default_config or {})
    return [{**defaults, **dict(config or {})} for config in item_configs]


def glm_sample_frame_indices(
    total_frames,
    fps,
    duration,
    *,
    target_fps=None,
    max_frame_count=None,
):
    """Mirror HF's sequential sampler so EPD stays byte-identical, then preserve temporal pairs."""
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
    total_num_frames = len(frame_list)
    if total_num_frames == 0:
        raise ValueError("GLM video frame list must not be empty")
    duration = 0.0
    if frame_list[0].get("detail") is not None:
        details = json.loads(frame_list[0]["detail"])
        duration = float(details.get("video_duration", 0))
    if duration == 0:
        base_ts = float(frame_list[0].get("timestamp", 0) or 0)
        duration = float(frame_list[-1].get("timestamp", base_ts) or base_ts) - base_ts
    images = [frame["frame_image"] for frame in frame_list]
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images).permute(0, 2, 3, 1).contiguous()
    else:
        images = [np.asarray(image) for image in images]
    fps = total_num_frames / duration if duration else 0
    return images, _glm_video_metadata(
        total_num_frames, fps, duration, range(total_num_frames)
    )


GLM_VIDEO_PRE_RESIZE_CHUNK = 64


def _vendor_smart_resize_canvas(
    num_frames, height, width, *, temporal_factor, factor, min_pixels, max_pixels
):
    """Replica of the vendor Glm5Next smart_resize (align-ceil + budget search)."""

    def align(value):
        return math.ceil(value / factor) * factor

    def fit_within_budget(aligned_frames):
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height)
            candidate_width = align(content_width)
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(
        temporal_factor, round(num_frames / temporal_factor) * temporal_factor
    )
    canvas_height, canvas_width = align(height), align(width)
    if aligned_frames * canvas_height * canvas_width > max_pixels:
        canvas_height, canvas_width = fit_within_budget(aligned_frames)
    elif aligned_frames * canvas_height * canvas_width < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        canvas_height = align(max(1, math.ceil(height * scale)))
        canvas_width = align(max(1, math.ceil(width * scale)))
        if aligned_frames * canvas_height * canvas_width > max_pixels:
            canvas_height, canvas_width = fit_within_budget(aligned_frames)
    return canvas_height, canvas_width


def _pre_resize_frames_for_processor(
    frames,
    *,
    factor,
    temporal_factor,
    pixels_per_token,
    min_pixels,
    max_pixels,
    resize_mode,
):
    """Pre-resize in chunks to avoid HF's native-resolution float32 intermediate while preserving its output grid."""
    import torchvision.transforms.functional as TF

    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(np.asarray(frames))
    nchw = frames.permute(0, 3, 1, 2)
    num_frames, _, height, width = nchw.shape
    canvas_height, canvas_width = _vendor_smart_resize_canvas(
        num_frames,
        height,
        width,
        temporal_factor=temporal_factor,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if resize_mode == "resize":
        content_height, content_width = canvas_height, canvas_width
    else:
        scale = min(canvas_height / height, canvas_width / width)
        if num_frames * height * width >= min_pixels:
            scale = min(1.0, scale)
        content_height = max(1, min(canvas_height, math.floor(height * scale)))
        content_width = max(1, min(canvas_width, math.floor(width * scale)))
    if (content_height, content_width) != (height, width):
        nchw = torch.cat(
            [
                TF.resize(
                    chunk,
                    [content_height, content_width],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True,
                )
                for chunk in nchw.split(GLM_VIDEO_PRE_RESIZE_CHUNK)
            ]
        )
    if (content_height, content_width) != (canvas_height, canvas_width):
        nchw = torch.nn.functional.pad(
            nchw, (0, canvas_width - content_width, 0, canvas_height - content_height)
        )
    return nchw.permute(0, 2, 3, 1).contiguous()


def glm_decode_frames_at(vr, indices, video_config=None):
    indices = list(indices)
    if not indices:
        return None
    video_config = video_config or {}
    if hasattr(vr, "get_frames_as_tensor"):
        frames = vr.get_frames_as_tensor(indices)
    else:
        frames = vr.get_frames_at(indices)
    max_tokens_per_frame = video_config.get("max_tokens_per_frame")
    if max_tokens_per_frame is not None:
        frames = _resize_frames_to_max_tokens(frames, max_tokens_per_frame)
    elif budget := video_config.get("_presize_budget"):
        frames = _pre_resize_frames_for_processor(frames, **budget)
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
    smart_rgb_conversion = True
    video_preprocessing_device = "cpu"
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
        # Bare base64 video must use SGLang's decoder because HF treats it as a path-like string.
        video_urls, video_configs = split_glm_video_items(request_obj.video_data)
        video_processor = getattr(self._processor, "video_processor", None)
        default_video_config = glm_processor_video_config(video_processor)
        default_video_config.update(self.video_config)
        video_configs = _merge_glm_video_configs(default_video_config, video_configs)

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_urls,
            multimodal_tokens=self.mm_tokens,
        )

        video_metadata = None
        videos_kwargs = None
        if base_output.videos and not isinstance(base_output.videos[0], dict):
            videos_kwargs = glm_budget_kwargs(
                video_processor,
                user_max_image_tokens=glm_max_image_tokens_from_configs(video_configs),
                count=len(base_output.videos),
                split=True,
            )
            effective_max_image_tokens = (videos_kwargs or {}).get("max_image_tokens")
            loop = asyncio.get_running_loop()
            decode_tasks = []
            for index, video in enumerate(base_output.videos):
                video_config = _glm_effective_presize_budget(
                    video_configs[index] if index < len(video_configs) else {},
                    effective_max_image_tokens,
                )
                if isinstance(video, VideoDecoderWrapper):
                    decode_tasks.append(
                        loop.run_in_executor(
                            self.io_executor,
                            glm_sample_and_decode_sync,
                            video,
                            video_config,
                        )
                    )
                elif isinstance(video, list) and (
                    not video or isinstance(video[0], dict)
                ):
                    decode_tasks.append(
                        loop.run_in_executor(
                            self.io_executor, preprocess_video_frames_sync, video
                        )
                    )
                else:
                    decode_tasks.append(asyncio.sleep(0, result=(video, None)))

            videos_processed = await asyncio.gather(*decode_tasks)
            for video in base_output.videos:
                close = getattr(video, "close", None)
                if callable(close):
                    close()
            base_output.videos, metadata = map(list, zip(*videos_processed))
            if metadata and all(item is not None for item in metadata):
                video_metadata = metadata

        combine_kwargs = {}
        if video_metadata is not None:
            # Skip HF resampling because these frames already carry their original indices.
            combine_kwargs["video_metadata"] = video_metadata
            combine_kwargs["do_sample_frames"] = False
            processor_video_config = {
                key: value
                for key, value in self.video_config.items()
                if key not in {"fps", "max_frames", "max_tokens_per_frame"}
            }
            if videos_kwargs is not None:
                processor_video_config.update(videos_kwargs)
            combine_kwargs["processor_video_config"] = processor_video_config

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens, **combine_kwargs
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
