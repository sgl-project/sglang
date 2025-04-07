import asyncio
from typing import List, Optional, Union

import numpy as np

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.models.llava import (
    LlavaLlamaForCausalLM,
    LlavaMistralForCausalLM,
    LlavaQwenForCausalLM,
)
from sglang.srt.models.llavavid import LlavaVidForCausalLM
from sglang.srt.utils import load_image, logger
from sglang.utils import get_exception_traceback


class LlavaImageProcessor(BaseMultimodalProcessor):
    models = [
        LlavaLlamaForCausalLM,
        LlavaVidForCausalLM,
        LlavaQwenForCausalLM,
        LlavaMistralForCausalLM,
    ]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        processor=None,
    ):

        image_processor = processor.image_processor

        try:
            image, image_size = load_image(image_data)
            if image_size is not None:
                # It is a video with multiple images
                image_hash = hash(image_data)
                pixel_values = image_processor(image)["pixel_values"]
                for _ in range(len(pixel_values)):
                    pixel_values[_] = pixel_values[_].astype(np.float16)
                pixel_values = np.stack(pixel_values, axis=0)
                return pixel_values, image_hash, image_size
            else:
                # It is an image
                image_hash = hash(image_data)
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
        self, image_data: Union[bytes, str], aspect_ratio: str, grid_pinpoints: str
    ):
        if self.cpu_executor is not None:
            loop = asyncio.get_event_loop()
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
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None

        modalities = request_obj.modalities or ["image"]
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints")
            and "anyres" in aspect_ratio
            else None
        )

        if isinstance(image_data, str):
            image_data = [image_data]

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
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    modality=modality,
                )
            ],
        }
