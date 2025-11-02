import asyncio
import math
import os
import re
from typing import List, Union

import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.utils import logger

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

FRAME_FACTOR = 2


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
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


def resize_image(
    image,
    min_pixels: int,
    max_pixels: int,
    size_factor: int,
) -> Image.Image:
    width, height = image.size
    min_pixels = min_pixels
    max_pixels = max_pixels
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height), resample=RESIZE_RESAMPLE)
    return image


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


async def resize_image_async(
    image,
    min_pixels: int,
    max_pixels: int,
    size_factor: int,
):
    return resize_image(image, min_pixels, max_pixels, size_factor)


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
    fps: int | float,
    fps_min_frames: int,
    fps_max_frames: int,
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
        fps (int | float, optional): target frames to sample per second. Defaults to FPS.
        fps_min_frames (int, optional): the minimum number of frames of the video. Defaults to FPS_MIN_FRAMES.
        fps_max_frames (int, optional): the maximum number of frames of the video. Defaults to FPS_MAX_FRAMES.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", fps)
        min_frames = ceil_by_factor(ele.get("min_frames", fps_min_frames), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(fps_max_frames, total_frames)), FRAME_FACTOR
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
    image_factor: int,
    video_min_pixels: int,
    video_max_pixels: int,
    fps: int | float,
    fps_min_frames: int,
    fps_max_frames: int,
    # vr: VideoReader, image_factor: int = IMAGE_FACTOR
) -> torch.Tensor:
    ele = {}
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(
        ele,
        total_frames=total_frames,
        video_fps=video_fps,
        fps=fps,
        fps_min_frames=fps_min_frames,
        fps_max_frames=fps_max_frames,
    )
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", video_min_pixels)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(video_max_pixels, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,
        "video_backend": "torchvision",
    }
    return video, video_metadata


# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        if hf_config.model_type == "qwen3_omni_moe":
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id

        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

        self.NUM_TOKEN_PER_FRAME = 770
        # Use global defaults for other model types

        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()

        self.VIDEO_MIN_PIXELS = 128 * 28 * 28
        self.VIDEO_MAX_PIXELS = 768 * 28 * 28
        self.FPS = 2.0
        self.FPS_MIN_FRAMES = 4
        self.FPS_MAX_FRAMES = 768

        # Dynamically extract preprocessing parameters from processor for qwen3_vl models
        image_processor = getattr(_processor, "image_processor", None)
        video_processor = getattr(_processor, "video_processor", None)

        if hf_config.model_type in ("qwen3_vl", "qwen3_vl_moe"):
            # Extract image processing parameters from processor if available
            if image_processor:
                size = getattr(image_processor, "size", {})
                if size:
                    self.MIN_PIXELS = size.get("shortest_edge", 64 * 32 * 32)
                    self.MAX_PIXELS = size.get("longest_edge", 32 * 32 * 16384)
                patch_size = getattr(image_processor, "patch_size", 16)
                merge_size = getattr(image_processor, "merge_size", 2)
                self.IMAGE_FACTOR = patch_size * merge_size

            # Extract video processing parameters from video processor if available
            if video_processor:
                video_size = getattr(video_processor, "size", {})
                self.VIDEO_MIN_PIXELS = video_size.get("shortest_edge", 4 * 32 * 32)
                self.VIDEO_MAX_PIXELS = video_size.get("longest_edge", 32 * 32 * 24576)
                self.FPS = getattr(video_processor, "fps", 2)
                self.FPS_MIN_FRAMES = getattr(video_processor, "min_frames", 4)
                self.FPS_MAX_FRAMES = getattr(video_processor, "max_frames", 768)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=hf_config.video_token_id,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Qwen-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            resize_tasks = [
                resize_image_async(
                    image,
                    min_pixels=self.MIN_PIXELS,
                    max_pixels=self.MAX_PIXELS,
                    size_factor=self.IMAGE_FACTOR,
                )
                for image in base_output.images
            ]
            base_output.images = await asyncio.gather(*resize_tasks)

        video_metadata = None
        if base_output.videos:
            video_results = await asyncio.gather(
                *[
                    preprocess_video(
                        video,
                        image_factor=self.IMAGE_FACTOR,
                        video_min_pixels=self.VIDEO_MIN_PIXELS,
                        video_max_pixels=self.VIDEO_MAX_PIXELS,
                        fps=self.FPS,
                        fps_min_frames=self.FPS_MIN_FRAMES,
                        fps_max_frames=self.FPS_MAX_FRAMES,
                    )
                    for video in base_output.videos
                ]
            )
            base_output.videos, video_metadata = map(list, zip(*video_results))

        # NOTE: for qwen3-vl, video_meta need to be passed in, since do_sample_frames is already done in preprocess_video
        if self.hf_config.model_type in ("qwen3_vl", "qwen3_vl_moe"):
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
                video_metadata=video_metadata,
                do_sample_frames=False,
            )
        else:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )

        audio_feature_lengths = None

        if self.model_type == "qwen3_omni_moe":
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
            if audio_item:
                audio_feature_lengths = torch.sum(
                    audio_item.feature_attention_mask, dim=1
                )

        second_per_grid_ts = getattr(ret, "second_per_grid_ts", None) or getattr(
            ret, "video_second_per_grid", None
        )

        input_ids = input_ids.flatten()

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=audio_feature_lengths,
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, "position_id_per_seconds", None
            ),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "audio_token_id": self.mm_tokens.audio_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
