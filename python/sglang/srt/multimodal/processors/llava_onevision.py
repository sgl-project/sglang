# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Multimodal processor for LLaVA-OneVision.

Inherits ``LlavaImageProcessor``'s executor / IO plumbing, but overrides the
per-image encoding to call HuggingFace's ``LlavaOnevisionImageProcessor``
directly. HF's processor already produces the anyres patch batch as
``pixel_values[0]`` with shape ``(num_patches, C, H, W)``; SGLang's classic
``process_anyres_image`` helper (written for CLIP-based LLaVA) would double
that anyres processing and yield a 5-D tensor, so we bypass it.
"""

import asyncio
import os
from typing import List, Optional, Union

import numpy as np

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.llava_onevision import LlavaOnevisionForConditionalGeneration
from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
from sglang.srt.utils import ImageData, load_image


class LlavaOnevisionProcessor(LlavaImageProcessor):
    """Multimodal processor for ``LlavaOnevisionForConditionalGeneration``."""

    models = [LlavaOnevisionForConditionalGeneration]

    async def _process_single_image(
        self,
        image_data: Union[str, bytes, ImageData],
        aspect_ratio: Optional[str] = None,
        grid_pinpoints=None,
    ):
        """Load and preprocess one image via HF's LlavaOnevisionImageProcessor.

        The ``aspect_ratio`` / ``grid_pinpoints`` args are accepted for
        interface compatibility with the parent but are ignored — HF's own
        processor already applies OneVision's anyres_max recipe internally,
        keyed off ``hf_config.vision_aspect_ratio``.
        """
        if self.cpu_executor is not None:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(
                self.cpu_executor,
                LlavaOnevisionProcessor._encode_via_hf,
                image_data,
                self._processor,
            )
            timeout = int(os.environ.get("REQUEST_TIMEOUT", "10"))
            return await asyncio.wait_for(fut, timeout=timeout)
        return LlavaOnevisionProcessor._encode_via_hf(image_data, self._processor)

    @staticmethod
    def _encode_via_hf(image_data, processor):
        url = image_data.url if isinstance(image_data, ImageData) else image_data
        image, _size = load_image(url, False)
        image_hash = hash(url)
        out = processor.image_processor(image.convert("RGB"), return_tensors="np")
        # HF returns pixel_values of shape (1, num_patches, C, H, W); [0]
        # strips the leading batch dim, leaving the anyres patch stack.
        pixel_values = np.asarray(out["pixel_values"][0], dtype=np.float16)
        return pixel_values, image_hash, image.size  # image.size is (width, height)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, ImageData]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if (
            isinstance(image_data, list)
            and len(image_data) > 0
            and isinstance(image_data[0], dict)
        ):
            return self._process_precomputed_image_data(image_data)

        if not isinstance(image_data, list) or len(image_data) == 0:
            raise ValueError(f"Invalid image data: {image_data}")

        modalities = request_obj.modalities or ["image"]

        if "multi-images" in modalities or "video" in modalities:
            futures = [self._process_single_image(img) for img in image_data]
            results = await asyncio.gather(*futures)
            pixel_values, _hashes, image_sizes = map(list, zip(*results))
        else:
            pixel_v, _hash, image_s = await self._process_single_image(image_data[0])
            pixel_values, image_sizes = [pixel_v], [image_s]

        modality = Modality.IMAGE
        if isinstance(request_obj.modalities, list) and request_obj.modalities[0] == "video":
            modality = Modality.VIDEO

        mm_items = []
        for pixel_v, image_s in zip(pixel_values, image_sizes):
            mm_items.append(
                MultimodalDataItem(
                    feature=pixel_v,
                    model_specific_data={"image_sizes": [image_s]},
                    modality=modality,
                )
            )
        return MultimodalProcessorOutput(mm_items=mm_items)
