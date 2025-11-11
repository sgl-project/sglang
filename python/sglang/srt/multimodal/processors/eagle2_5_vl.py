import math
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
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


async def resize_image_async(
    image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    size_factor: int = IMAGE_FACTOR,
):
    """Async wrapper for resize_image."""
    return resize_image(image, min_pixels, max_pixels, size_factor)


class Eagle2_5_VLProcessor(SGLangBaseProcessor):
    """Multimodal processor for Eagle2.5 Vision-Language Model."""

    models = [Eagle2_5_VLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Use the passed processor instead of loading our own
        self.hf_processor = _processor

        # Eagle2.5 specific attributes (from config)
        self.image_token_index = getattr(hf_config, "image_token_index", 151667)
        self.image_token_id = self.image_token_index  # Alias for consistency
        self.video_token_id = getattr(hf_config, "video_token_id", 151670)
        self.audio_token_id = None  # Eagle2.5 doesn't support audio yet

        self.downsample_ratio = getattr(hf_config, "downsample_ratio", 0.5)
        self.dynamic_image_size = getattr(hf_config, "dynamic_image_size", True)
        self.force_image_size = getattr(hf_config, "force_image_size", 224)
        self.max_dynamic_tiles = getattr(hf_config, "max_dynamic_tiles", 12)
        self.min_dynamic_tiles = getattr(hf_config, "min_dynamic_tiles", 1)
        self.use_thumbnail = getattr(hf_config, "use_thumbnail", True)

        # Special tokens - Eagle2.5 uses numbered image placeholders in prompts like <image-1>, <image-2>, etc.
        # The actual token in the tokenizer is <IMG_CONTEXT>, but the text contains <image-N> placeholders
        # We use regex to match all numbered placeholders since there can be multiple images per prompt

        # Initialize multimodal tokens with regex to match numbered placeholders
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>",  # Generic representation (not used due to custom regex)
            image_token_id=self.image_token_id,
            image_token_regex=re.compile(
                r"<image-\d+>"
            ),  # Match <image-1>, <image-2>, etc.
            video_token="<video>",  # Generic representation (not used due to custom regex)
            video_token_id=self.video_token_id,
            video_token_regex=re.compile(
                r"<video-\d+>"
            ),  # Match <video-1>, <video-2>, etc.
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    def _get_mm_special_tokens(self) -> MultimodalSpecialTokens:
        """Get multimodal special tokens."""
        # Note: This method appears unused, but kept for compatibility
        return MultimodalSpecialTokens(
            image_start_token="<image>",
            image_end_token="<image>",  # Same token for start/end
            image_token="<image>",
            video_start_token="<video>",
            video_end_token="<video>",
            video_token="<video>",
        )

    def __call__(
        self,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        videos: Optional[List[torch.Tensor]] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main method to prepare inputs for Eagle2.5-VL model.

        Args:
            images: PIL Image(s) to process
            text: Text string(s) to tokenize
            videos: Video tensor(s) to process (not fully supported yet)
            return_tensors: Format of returned tensors ('pt' for PyTorch)
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing:
            - input_ids: Tokenized text input
            - attention_mask: Attention mask for input
            - pixel_values: Processed image tensors (if images provided)
            - image_grid_thw: Image grid dimensions (if images provided)
            - pixel_values_videos: Processed video tensors (if videos provided)
            - video_grid_thw: Video grid dimensions (if videos provided)
        """
        # Initialize output dictionary
        output = {}

        # Process images if provided
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]

            # Resize images according to Eagle2.5's requirements
            processed_images = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = resize_image(
                    img,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                    size_factor=IMAGE_FACTOR,
                )
                processed_images.append(img)

            # Use HF processor's image processor
            if hasattr(self.hf_processor, "image_processor"):
                image_outputs = self.hf_processor.image_processor(
                    images=processed_images,
                    return_tensors=return_tensors,
                )
                output.update(image_outputs)

        # Process videos if provided
        if videos is not None:
            video_data = self._process_videos(videos)
            output.update(video_data)

        # Process text if provided
        if text is not None:
            if not isinstance(text, list):
                text = [text]

            # Tokenize text
            text_outputs = self.hf_processor.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=kwargs.get("padding", False),
                truncation=kwargs.get("truncation", False),
            )
            output.update(text_outputs)

        return output

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

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Process multimodal data for Eagle2_5_VL."""
        # Use base class to load multimodal data (handles various input formats)
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=getattr(request_obj, "video_data", None),
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Eagle2.5-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            import asyncio

            resize_tasks = [
                resize_image_async(
                    image,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                    size_factor=IMAGE_FACTOR,
                )
                for image in base_output.images
            ]
            base_output.images = await asyncio.gather(*resize_tasks)

        # Use base class to process and combine multimodal data with tokens
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # Return processed multimodal data
        if mm_items:
            input_ids = input_ids.flatten()
            return {
                "input_ids": input_ids.tolist(),
                "mm_items": mm_items,
                "im_start_id": self.image_token_id,
                "im_end_id": self.image_token_id,
                "im_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "audio_token_id": self.audio_token_id,
            }

        return None
