# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
OpenVLA multimodal processor for SGLang.

Handles image preprocessing for OpenVLA's dual vision backbone (DINOv2 + SigLIP).
"""

import asyncio
from typing import List, Union

import numpy as np
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.openvla import OpenVLAForActionPrediction
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import ImageData, load_image, logger
from sglang.utils import get_exception_traceback


class OpenVLAImageProcessor(BaseMultimodalProcessor):
    """Processor for OpenVLA image inputs.

    OpenVLA uses a 224x224 image size with dual vision encoders.
    The processor handles image loading and normalization.
    """

    models = [OpenVLAForActionPrediction]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_size = getattr(hf_config, "image_size", 224)

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes, ImageData, Image.Image],
        image_size: int = 224,
        processor=None,
    ):
        """Process a single image for OpenVLA.

        Args:
            image_data: Image URL, bytes, ImageData object, or PIL Image.
            image_size: Target image size (default 224 for OpenVLA).
            processor: HuggingFace processor with image_processor.

        Returns:
            Tuple of (pixel_values, image_hash, image_size).
        """
        try:
            # Handle PIL Image directly
            if isinstance(image_data, Image.Image):
                image = image_data
                image_hash = id(image)  # Use object id as hash for PIL images
            else:
                url = image_data.url if isinstance(image_data, ImageData) else image_data
                image, _ = load_image(url)
                image_hash = hash(url)

            # Precompute BOTH DINOv2 and SigLIP normalized pixels (6 channels total)
            # This matches HF's preprocessing and avoids bfloat16 precision loss
            # during runtime conversion.
            image = image.convert("RGB").resize(
                (image_size, image_size), Image.BILINEAR
            )
            raw = np.array(image, dtype=np.float32) / 255.0

            # DINOv2 normalization (HF's exact quantized ImageNet stats)
            dinov2_mean = np.array([0.484375, 0.455078125, 0.40625])
            dinov2_std = np.array([0.228515625, 0.2236328125, 0.224609375])
            dinov2_pixels = (raw - dinov2_mean) / dinov2_std
            dinov2_pixels = dinov2_pixels.transpose(2, 0, 1)  # HWC -> CHW

            # SigLIP normalization
            siglip_mean = np.array([0.5, 0.5, 0.5])
            siglip_std = np.array([0.5, 0.5, 0.5])
            siglip_pixels = (raw - siglip_mean) / siglip_std
            siglip_pixels = siglip_pixels.transpose(2, 0, 1)  # HWC -> CHW

            # Stack into 6-channel tensor: [DINOv2(3), SigLIP(3)]
            # Use float32 to preserve precision - will be converted to model dtype later
            pixel_values = np.concatenate([dinov2_pixels, siglip_pixels], axis=0).astype(np.float32)

            return pixel_values, image_hash, (image_size, image_size)

        except Exception:
            logger.error(
                "Exception in OpenVLA image processing:\n" + get_exception_traceback()
            )
            return None, None, None

    async def _process_single_image(
        self,
        image_data: Union[bytes, str, ImageData, Image.Image],
    ):
        """Async wrapper for image processing."""
        # PIL Images cannot be pickled properly for ProcessPoolExecutor
        # Process them directly in the current process
        if isinstance(image_data, Image.Image):
            return self._process_single_image_task(
                image_data,
                self.image_size,
                self._processor,
            )

        if self.cpu_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.cpu_executor,
                OpenVLAImageProcessor._process_single_image_task,
                image_data,
                self.image_size,
                self._processor,
            )
        else:
            return self._process_single_image_task(
                image_data,
                self.image_size,
                self._processor,
            )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, ImageData]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        """Process multimodal data asynchronously.

        Args:
            image_data: List of images to process.
            input_text: The input text/prompt.
            request_obj: Request object with metadata.

        Returns:
            Dictionary with mm_items containing processed images.
        """
        # Handle precomputed embeddings
        if (
            isinstance(image_data, list)
            and len(image_data) > 0
            and isinstance(image_data[0], dict)
        ):
            mm_items = []
            for item in image_data:
                if "image_sizes" not in item:
                    item["image_sizes"] = [(self.image_size, self.image_size)]
                mm_items.append(
                    MultimodalDataItem(
                        feature=item.get("feature"),
                        modality=Modality.IMAGE,
                        model_specific_data=item,
                    )
                )
            return {"mm_items": mm_items}

        if not image_data or len(image_data) == 0:
            return {"mm_items": []}

        # Process single image (OpenVLA typically uses single image)
        if len(image_data) == 1:
            pixel_values, image_hash, image_size = await self._process_single_image(
                image_data[0]
            )
            if pixel_values is None:
                raise ValueError("Failed to process image")

            return {
                "mm_items": [
                    MultimodalDataItem(
                        feature=pixel_values,
                        model_specific_data={
                            "image_sizes": [image_size],
                        },
                        modality=Modality.IMAGE,
                    )
                ],
            }

        # Process multiple images
        tasks = [self._process_single_image(img) for img in image_data]
        results = await asyncio.gather(*tasks)

        pixel_values_list = []
        image_sizes = []
        for pv, ih, isz in results:
            if pv is not None:
                pixel_values_list.append(pv)
                image_sizes.append(isz)

        if not pixel_values_list:
            raise ValueError("Failed to process any images")

        # Stack into batch
        pixel_values = np.stack(pixel_values_list, axis=0)

        return {
            "mm_items": [
                MultimodalDataItem(
                    feature=pixel_values,
                    model_specific_data={
                        "image_sizes": image_sizes,
                    },
                    modality=Modality.IMAGE,
                )
            ],
        }
