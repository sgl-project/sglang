"""
Eagle2.5-VL Processor for SGLang.

This processor wraps NVIDIA's Eagle2.5-VL HuggingFace processor
to work with SGLang's multimodal pipeline.

References:
- HuggingFace: https://huggingface.co/nvidia/Eagle2.5-8B/blob/main/processing_eagle2_5_vl.py
- Config: https://huggingface.co/nvidia/Eagle2.5-8B/blob/main/config.json
"""

import re
import time
from typing import Any

import numpy as np
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import smart_nframes
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import logger


# Eagle2.5 video preprocessing (adapted from Qwen-VL)
async def preprocess_eagle_video(
    vr, video_config: dict = {}
) -> tuple[torch.Tensor, dict]:
    """
    Preprocess video for Eagle2.5 model (adapted from Qwen-VL).

    Args:
        vr: VideoReader object from decord
        video_config: Configuration for video processing

    Returns:
        Tuple of (processed_video_tensor, metadata_dict)
    """
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )

    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    # Extract frames
    video_np = vr.get_batch(idx).asnumpy()
    video = torch.from_numpy(video_np).pin_memory()
    video = video.permute(0, 3, 1, 2)  # Convert to TCHW format

    # Convert frame indices to time in seconds
    timestamps = (idx / video_fps).tolist()

    # Create metadata
    video_metadata = {
        "fps": video_fps,
        "timestamps": timestamps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "video_backend": "torchvision",
    }

    get_batch_time = time.perf_counter()
    logger.debug(
        f"[preprocess_eagle_video Perf], "
        f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms"
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
        # maybe remove this if we never use it
        self.video_config = server_args.mm_process_config.get("video", {})

        # Use HF image processor directly (our local modifications are no longer needed)

        tokenizer: PreTrainedTokenizerBase = getattr(self._processor, "tokenizer")

        # Eagle2.5 uses numbered image placeholders like <image-1>, <image-2>
        # We need regex to match these patterns
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>",  # Generic representation that gets replaced by <image-1>, <image-2>, etc.
            image_token_id=getattr(hf_config, "image_token_index", 151667),
            image_token_regex=re.compile(
                r"<image-\d+>"
            ),  # Match <image-1>, <image-2>, etc.
            video_token="<video>",  # Generic representation that gets replaced by <video-1>, <video-2>, etc.
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
        video_kwargs = {}

        if base_output.videos:
            videos_processed = [
                await preprocess_eagle_video(video, video_config=self.video_config)
                for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

            # Eagle2.5 specific kwargs
            videos_kwargs = {
                "fps": [m["fps"] for m in video_metadata],
                "timestamps": [m["timestamps"] for m in video_metadata],
                "max_dynamic_tiles": 1,
                "min_dynamic_tiles": 1,
                "use_thumbnail": True,
            }

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output,
            self.mm_tokens,
            videos_kwargs=video_kwargs,
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
