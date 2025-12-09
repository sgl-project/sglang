import asyncio
from typing import List, Optional, Union

import numpy as np
from transformers.models.auto.processing_auto import (
    PROCESSOR_MAPPING_NAMES as HF_MAPPING_NAMES,
)

import sglang.srt.managers.multimodal_processor as sgl_mm_processor_utils
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.llava import (
    LlavaForConditionalGeneration,
    LlavaLlamaForCausalLM,
    LlavaMistralForCausalLM,
    LlavaQwenForCausalLM,
)
from sglang.srt.models.llavavid import LlavaVidForCausalLM
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.multimodal.mm_utils import expand2square, process_anyres_image
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import ImageData, load_image, logger
from sglang.utils import get_exception_traceback


class LlavaImageProcessor(BaseMultimodalProcessor):
    models = [
        LlavaLlamaForCausalLM,
        LlavaVidForCausalLM,
        LlavaQwenForCausalLM,
        LlavaMistralForCausalLM,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes, ImageData],
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        processor=None,
    ):

        image_processor = processor.image_processor

        try:
            url = image_data.url if isinstance(image_data, ImageData) else image_data
            image, image_size = load_image(url)
            if image_size is not None:
                # It is a video with multiple images
                image_hash = hash(url)
                pixel_values = image_processor(image)["pixel_values"]
                for _ in range(len(pixel_values)):
                    pixel_values[_] = pixel_values[_].astype(np.float16)
                pixel_values = np.stack(pixel_values, axis=0)
                return pixel_values, image_hash, image_size
            else:
                # It is an image
                image_hash = hash(url)
                if image_aspect_ratio == "pad":
                    image = expand2square(
                        image,
                        tuple(int(x * 255) for x in image_processor.image_mean),
                    )
                    pixel_values = image_processor(image.convert("RGB"))[
                        "pixel_values"
                    ][0]
                elif image_aspect_ratio == "anyres" or (
                    image_aspect_ratio is not None
                    and "anyres_max" in image_aspect_ratio
                ):
                    pixel_values = process_anyres_image(
                        image, image_processor, image_grid_pinpoints
                    )
                else:
                    pixel_values = image_processor(image)["pixel_values"][0]

                if isinstance(pixel_values, np.ndarray):
                    pixel_values = pixel_values.astype(np.float16)

                return pixel_values, image_hash, image.size
        except Exception:
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())

    async def _process_single_image(
        self,
        image_data: Union[bytes, str, ImageData],
        aspect_ratio: str,
        grid_pinpoints: str,
    ):
        if self.cpu_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.cpu_executor,
                LlavaImageProcessor._process_single_image_task,
                image_data,
                aspect_ratio,
                grid_pinpoints,
                self._processor,
            )
        else:
            return self._process_single_image_task(
                image_data,
                aspect_ratio,
                grid_pinpoints,
                self._processor.image_processor,
            )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, ImageData]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        modalities = request_obj.modalities or ["image"]
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints")
            and "anyres" in aspect_ratio
            else None
        )

        if isinstance(image_data, list) and len(image_data) > 0:
            if "multi-images" in modalities or "video" in modalities:
                # Multiple images
                aspect_ratio = "pad"  # LLaVA OneVision Handling: more than one image --> interleaved image mode or video mode. We do not use anyres
                pixel_values, data_hashes, image_sizes = [], [], []
                res = []
                for img_data in image_data:
                    res.append(
                        self._process_single_image(
                            img_data, aspect_ratio, grid_pinpoints
                        )
                    )

                res = await asyncio.gather(*res)
                for pixel_v, image_h, image_s in res:
                    pixel_values.append(pixel_v)
                    data_hashes.append(image_h)
                    image_sizes.append(image_s)

                if isinstance(pixel_values[0], np.ndarray):
                    pixel_values = np.stack(pixel_values, axis=0)
            else:
                # A single image
                pixel_values, image_hash, image_size = await self._process_single_image(
                    image_data[0], aspect_ratio, grid_pinpoints
                )
                image_sizes = [image_size]
        else:
            raise ValueError(f"Invalid image data: {image_data}")
        modality = Modality.IMAGE
        if isinstance(request_obj.modalities, list):
            if request_obj.modalities[0] == "multi-images":
                modality = Modality.MULTI_IMAGES
            elif request_obj.modalities[0] == "video":
                modality = Modality.VIDEO

        return {
            "mm_items": [
                MultimodalDataItem(
                    feature=pixel_values,
                    model_specific_data={
                        "image_sizes": image_sizes,
                    },
                    modality=modality,
                )
            ],
        }


class LlavaMultimodalProcessor(BaseMultimodalProcessor):
    """
    This is a wrapper class used to identify the multimodal processor for Llava architectures' vision model.
    """

    models = [LlavaForConditionalGeneration, Mistral3ForConditionalGeneration]

    def _get_sgl_processor_cls(self, model_type: str):
        if hf_name := HF_MAPPING_NAMES.get(model_type):
            sgl_mm_processor_set = sgl_mm_processor_utils.PROCESSOR_MAPPING.values()
            sgl_processor_cls = list(
                filter(lambda p: p.__name__ == hf_name, sgl_mm_processor_set)
            )
            if sgl_processor_cls:
                return sgl_processor_cls[0]
        raise ValueError(
            f"Cannot find corresponding multimodal processor registered in sglang for model type `{model_type}`"
        )

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        assert hasattr(hf_config, "vision_config")
        assert hasattr(hf_config, "text_config")
        self.vision_config = hf_config.vision_config
        self.text_config = hf_config.text_config
        self.hf_config = hf_config

        if vision_type := getattr(self.vision_config, "model_type"):
            self.inner = self._get_sgl_processor_cls(vision_type)(
                hf_config, server_args, _processor, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Required `vision_config.model_type` is not found in hf_config: `{hf_config}`"
            )

    async def process_mm_data_async(self, *args, **kwargs):
        return await self.inner.process_mm_data_async(*args, **kwargs)
