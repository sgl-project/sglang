import math
import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.utils import logger

# Eagle2.5 specific constants
IMAGE_FACTOR = 14  # SigLIP patch size
MIN_PIXELS = 4 * 14 * 14
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)

if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        "Ignoring and using default."
    )

VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 14 * 14 * 0.9))
)
VIDEO_MIN_PIXELS = 128 * 14 * 14
VIDEO_MAX_PIXELS = 768 * 14 * 14
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


def resize_image(
    image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height), resample=RESIZE_RESAMPLE)
    return image


class Eagle2_5_VLProcessor(SGLangBaseProcessor):
    """Multimodal processor for Eagle2.5 Vision-Language Model."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)

        # Load the original HF processor
        self.hf_processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

        # Eagle2.5 specific attributes (NVIDIA's default is 151667)
        self.image_token_index = 151667  # From config
        self.downsample_ratio = 0.5
        self.dynamic_image_size = True
        self.force_image_size = 224
        self.max_dynamic_tiles = 12
        self.min_dynamic_tiles = 1
        self.use_thumbnail = True

        # Special tokens
        self.image_token = "<image>"
        self.video_token = "<video>"

    def _get_mm_special_tokens(self) -> MultimodalSpecialTokens:
        """Get multimodal special tokens."""
        return MultimodalSpecialTokens(
            image_start_token=self.image_token,
            image_end_token=self.image_token,  # Same token for start/end
            image_token=self.image_token,
            video_start_token=self.video_token,
            video_end_token=self.video_token,
            video_token=self.video_token,
        )

    def _process_images(self, images: List[Image.Image]) -> dict:
        """Process images for Eagle2.5."""
        processed_images = []

        for image in images:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize image according to Eagle2.5's smart resize
            image = resize_image(
                image,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                size_factor=IMAGE_FACTOR,
            )

            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

            processed_images.append(image_tensor)

        # Stack images
        if processed_images:
            pixel_values = torch.stack(processed_images, dim=0)
        else:
            pixel_values = torch.empty(0, 3, 224, 224)

        # Create image grid info (temporal, height, width)
        num_images = len(images)
        grid_thw = torch.zeros(num_images, 3, dtype=torch.int32)

        for i, img_tensor in enumerate(processed_images):
            _, h, w = img_tensor.shape
            # For static images, temporal dimension is 1
            grid_thw[i] = torch.tensor([1, h // IMAGE_FACTOR, w // IMAGE_FACTOR])

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }

    def _process_videos(self, videos: List[torch.Tensor]) -> dict:
        """Process videos for Eagle2.5."""
        processed_videos = []

        for video in videos:
            # Video should be in (T, H, W, C) format
            if video.dim() == 4:
                # Convert to (T, C, H, W)
                video = video.permute(0, 3, 1, 2)

            # Process each frame
            processed_frames = []
            for frame in video:
                # Convert to PIL for resizing
                frame_pil = Image.fromarray(
                    (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )
                frame_pil = resize_image(frame_pil)
                frame_tensor = torch.from_numpy(
                    np.array(frame_pil).astype(np.float32) / 255.0
                )
                frame_tensor = frame_tensor.permute(2, 0, 1)
                processed_frames.append(frame_tensor)

            video_tensor = torch.stack(processed_frames, dim=0)
            processed_videos.append(video_tensor)

        # Stack videos
        if processed_videos:
            pixel_values_videos = torch.stack(processed_videos, dim=0)
        else:
            pixel_values_videos = torch.empty(0, 1, 3, 224, 224)

        # Create video grid info
        num_videos = len(videos)
        grid_thw = torch.zeros(num_videos, 3, dtype=torch.int32)

        for i, vid_tensor in enumerate(processed_videos):
            t, _, h, w = vid_tensor.shape
            grid_thw[i] = torch.tensor([t, h // IMAGE_FACTOR, w // IMAGE_FACTOR])

        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": grid_thw,
        }

    def _get_image_tokens(self, image_grid_thw: torch.Tensor) -> List[str]:
        """Generate image tokens based on grid dimensions."""
        tokens = []
        for grid in image_grid_thw:
            t, h, w = grid.tolist()
            # Eagle2.5 uses a specific tokenization strategy
            # For now, use a simple approach - one token per patch
            num_patches = t * h * w
            tokens.extend([self.image_token] * num_patches)
        return tokens

    def _get_video_tokens(self, video_grid_thw: torch.Tensor) -> List[str]:
        """Generate video tokens based on grid dimensions."""
        tokens = []
        for grid in video_grid_thw:
            t, h, w = grid.tolist()
            # Similar to images but with temporal dimension
            num_patches = t * h * w
            tokens.extend([self.video_token] * num_patches)
        return tokens

    def process_mm_content(
        self,
        content: Union[str, List[dict]],
        images: List[Image.Image] = None,
        videos: List[torch.Tensor] = None,
    ) -> dict:
        """Process multimodal content for Eagle2.5."""

        # Process images if provided
        image_data = {}
        image_tokens = []
        if images:
            image_data = self._process_images(images)
            image_tokens = self._get_image_tokens(image_data["image_grid_thw"])

        # Process videos if provided
        video_data = {}
        video_tokens = []
        if videos:
            video_data = self._process_videos(videos)
            video_tokens = self._get_video_tokens(video_data["video_grid_thw"])

        # Combine all multimodal data
        mm_data = {**image_data, **video_data}

        # Process text content
        if isinstance(content, str):
            # Replace image/video placeholders with actual tokens
            text_content = content
            # Simple replacement - in practice this would be more sophisticated
            text_content = text_content.replace("<image>", " ".join(image_tokens))
            text_content = text_content.replace("<video>", " ".join(video_tokens))

            # Tokenize text
            inputs = self.hf_processor.tokenizer(
                text_content,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )

        elif isinstance(content, list):
            # Handle chat format
            messages = content
            # Use HF processor's chat template
            if hasattr(self.hf_processor, "apply_chat_template"):
                text_content = self.hf_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.hf_processor.tokenizer(
                    text_content,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
            else:
                # Fallback for processors without chat template
                inputs = self.hf_processor.tokenizer(
                    str(messages),
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )

        # Merge tokenized inputs with multimodal data
        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get(
                "attention_mask", torch.ones_like(inputs["input_ids"])
            ),
            **mm_data,
        }

        return result


# Register the processor
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


@register_customized_processor(Eagle2_5_VLProcessor)
class Eagle2_5_VLConfig:
    """Dummy class for processor registration."""

    model_type = "eagle_2_5_vl"
