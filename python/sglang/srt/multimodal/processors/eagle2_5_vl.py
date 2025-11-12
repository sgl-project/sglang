"""
Eagle2.5-VL Processor for SGLang.

This processor wraps NVIDIA's Eagle2.5-VL HuggingFace processor
to work with SGLang's multimodal pipeline.

References:
- HuggingFace: https://huggingface.co/nvidia/Eagle2.5-8B/blob/main/processing_eagle2_5_vl.py
- Config: https://huggingface.co/nvidia/Eagle2.5-8B/blob/main/config.json
"""

import math
import os
import re
import time
from typing import Any

import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.image_processing_eagle2_5_vl_fast import (
    Eagle2_5_VLImageProcessorFast,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import logger

# Eagle2.5 video processing constants (adapted from Qwen-VL)
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,  # Eagle2.5 uses 28 as factor (patch size)
    min_pixels: int = VIDEO_MIN_PIXELS,
    max_pixels: int = VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the video frame dimensions so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the frame is maintained as closely as possible.
    """
    MAX_RATIO = 16  # Maximum aspect ratio
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
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
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


# Eagle2.5 video preprocessing (adapted from Qwen-VL)
async def preprocess_eagle_video(vr) -> tuple[torch.Tensor, dict]:
    """
    Preprocess video for Eagle2.5 model (adapted from Qwen-VL).

    Args:
        vr: VideoReader object from decord

    Returns:
        Tuple of (processed_video_tensor, metadata_dict)
    """
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes({}, total_frames=total_frames, video_fps=video_fps)
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    # Extract frames
    video_np = vr.get_batch(idx).asnumpy()
    video = torch.from_numpy(video_np).pin_memory()
    video = video.permute(0, 3, 1, 2)  # Convert to TCHW format

    nframes, _, height, width = video.shape
    get_batch_time = time.perf_counter()

    # Resize video frames for Eagle2.5
    ele = {}  # Empty config for now
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
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
            factor=28,  # Eagle2.5 patch size
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,  # Eagle2.5 patch size
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    smart_resize_time = time.perf_counter()

    # Resize frames
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )
    video = video.pin_memory()

    # Create metadata
    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,
        "video_backend": "torchvision",
    }

    torchvision_resize_time = time.perf_counter()
    logger.debug(
        f"[preprocess_eagle_video Perf], "
        f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms, "
        f"smart_resize_time: {(smart_resize_time - get_batch_time) * 1000:.2f} ms, "
        f"torchvision_resize_time: {(torchvision_resize_time - smart_resize_time) * 1000:.2f} ms, "
        f"total_time: {(torchvision_resize_time - entry_time) * 1000:.2f} ms"
    )

    return video, video_metadata


class Eagle2_5_VLProcessor(BaseMultimodalProcessor):
    """
    SGLang multimodal processor for Eagle2.5 Vision-Language Model.

    This processor uses the official HuggingFace Eagle2.5-VL processor
    for image/video processing and tokenization.
    """

    models = [Eagle2_5_VLForConditionalGeneration]

    def __init__(
        self,
        hf_config: PretrainedConfig,
        server_args: ServerArgs,
        _processor: ProcessorMixin,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self._processor: ProcessorMixin

        # Replace HF image processor with our local modified version to avoid bug in HF fast processor
        if hasattr(self._processor, "image_processor"):
            original_config = getattr(self._processor.image_processor, "__dict__", {})
            self._processor.image_processor = Eagle2_5_VLImageProcessorFast(
                **original_config
            )

        tokenizer: PreTrainedTokenizerBase = getattr(self._processor, "tokenizer")

        # Eagle2.5 uses numbered image placeholders like <image-1>, <image-2>
        # We need regex to match these patterns
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>",  # Generic representation
            image_token_id=getattr(hf_config, "image_token_index", 151667),
            image_token_regex=re.compile(
                r"<image-\d+>"
            ),  # Match <image-1>, <image-2>, etc.
            video_token="<video>",  # Generic representation
            video_token_id=getattr(hf_config, "video_token_id", 151670),
            video_token_regex=re.compile(
                r"<video-\d+>"
            ),  # Match <video-1>, <video-2>, etc.
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj: GenerateReqInput,
        **kwargs,
    ) -> dict[str, Any] | None:
        """
        Process multimodal data for Eagle2_5_VL using the base processor pipeline.

        Args:
            image_data: List of image data (URLs, bytes, PIL Images, ImageData objects)
            audio_data: Audio data (not supported by Eagle2.5)
            input_text: Text prompt with <image-N> placeholders
            request_obj: Request object containing additional data
            **kwargs: Additional processing arguments

        Returns:
            Dictionary containing:
            - input_ids: Tokenized input sequence
            - mm_items: List of MultimodalDataItem objects
            - im_token_id: Image token ID
            - video_token_id: Video token ID (optional)
        """
        # Load multimodal data using base processor (will replace with generic <image> tokens)
        base_output = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=request_obj.image_data,  # type: ignore
            video_data=request_obj.video_data,  # type: ignore
        )

        # Post-process: Convert generic <image> tokens back to numbered <image-1>, <image-2>, etc.
        base_output = self._postprocess_numbered_placeholders(base_output)

        # Preprocess videos for Eagle2.5 (adapted from Qwen-VL approach)
        video_metadata = None
        if base_output.videos:
            videos_processed = [
                await preprocess_eagle_video(video) for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output,
            self.mm_tokens,
            # images_kwargs={"resample": 3},  # Pass resample to HF processor
            # Alternative: {"images_kwargs": {"resample": 3}} if needed
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
        }

    def _postprocess_numbered_placeholders(self, base_output):
        """
        Post-process base_output to convert generic tokens back to numbered placeholders.

        The base processor replaces matched tokens with generic <image> tokens.
        This method converts them back to <image-1>, <image-2>, etc. for HF processor.

        Args:
            base_output: BaseMultiModalProcessorOutput from load_mm_data

        Returns:
            Modified base_output with numbered placeholders in input_text
        """
        input_text = base_output.input_text

        # Count actual images and videos loaded
        num_images = len(base_output.images) if base_output.images else 0
        num_videos = len(base_output.videos) if base_output.videos else 0

        # Early return if no multimodal content
        if num_images == 0 and num_videos == 0:
            return base_output

        # Use str.replace with count parameter for efficient single-pass replacement
        # Replace <image> tokens with numbered versions
        if num_images > 0:
            for i in range(1, num_images + 1):
                input_text = input_text.replace("<image>", f"<image-{i}>", 1)

        # Replace <video> tokens with numbered versions
        if num_videos > 0:
            for i in range(1, num_videos + 1):
                input_text = input_text.replace("<video>", f"<video-{i}>", 1)

        # Update the input_text in base_output (dataclass allows modification)
        base_output.input_text = input_text
        return base_output
