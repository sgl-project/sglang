import asyncio
from typing import List, Union

import numpy as np

from sglang.srt.managers.image_processor import BaseImageProcessor
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.utils import load_image
from sglang.utils import get_exception_traceback, logger


class Qwen2VLImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.hf_config = hf_config
        self._image_processor = _processor

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        image_processor=None,
    ):
        image_processor = image_processor or get_global_processor().image_processor

        try:
            image, image_size = load_image(image_data)
            if image_size is not None:
                # It is a video with multiple images
                image_hash = hash(image_data)
                process_result = image_processor(image)
                pixel_values, image_grid_thws = (
                    process_result["pixel_values"],
                    process_result["image_grid_thw"][0],
                )
                for _ in range(len(pixel_values)):
                    pixel_values[_] = pixel_values[_].astype(np.float16)
                pixel_values = np.stack(pixel_values, axis=0)
                image_grid_thws = np.stack(image_grid_thws, axis=0)
                return pixel_values, image_hash, image_size, image_grid_thws
            else:
                # It is an image
                image_hash = hash(image_data)
                process_result = image_processor(image)
                pixel_values, image_grid_thws = (
                    process_result["pixel_values"],
                    process_result["image_grid_thw"][0],
                )
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = pixel_values.astype(np.float16)

                return pixel_values, image_hash, image.size, image_grid_thws
        except Exception:
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())

    async def _process_single_image(self, image_data: Union[bytes, str]):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Qwen2VLImageProcessor._process_single_image_task,
                image_data,
            )
        else:
            return self._process_single_image_task(image_data)

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None

        if isinstance(image_data, list) and len(image_data) > 0:
            # Multiple images
            if len(image_data) > 1:
                pixel_values, image_hashes, image_sizes, image_grid_thws = (
                    [],
                    [],
                    [],
                    [],
                )
                res = []
                for img_data in image_data:
                    res.append(self._process_single_image(img_data))
                res = await asyncio.gather(*res)
                for pixel_v, image_h, image_s, image_thw in res:
                    pixel_values.append(pixel_v)
                    image_hashes.append(image_h)
                    image_sizes.append(image_s)
                    image_grid_thws.append(image_thw)

                if isinstance(pixel_values[0], np.ndarray):
                    pixel_values = np.concatenate(pixel_values, axis=0)
            else:
                # A single image
                pixel_values, image_hash, image_size, image_grid_thw = (
                    await self._process_single_image(image_data[0])
                )
                image_hashes = [image_hash]
                image_sizes = [image_size]
                image_grid_thws = [image_grid_thw]
        elif isinstance(image_data, str) or isinstance(image_data, bytes):
            # A single image
            pixel_values, image_hash, image_size, image_grid_thw = (
                await self._process_single_image(image_data)
            )
            image_hashes = [image_hash]
            image_sizes = [image_size]
            image_grid_thws = [image_grid_thw]
        else:

            raise ValueError(f"Invalid image data: {image_data}")
        return {
            "pixel_values": pixel_values,
            "image_hashes": image_hashes,
            "im_token_id": self.hf_config.image_token_id,
            "image_sizes": image_sizes,
            "modalities": request_obj.modalities or ["image"],
            "image_grid_thws": image_grid_thws,
        }


class Qwen2_5VLImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        print("aaaaaaa")

    @staticmethod
    def _process_images_task(images, input_text):
        result = get_global_processor().__call__(
            text=input_text, images=images, return_tensors="pt"
        )
        return {
            "input_ids": result.input_ids,
            "pixel_values": result.pixel_values,
            "image_grid_thws": result.image_grid_thw,
        }

    async def _process_images(self, images, input_text) -> dict:
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Qwen2_5VLImageProcessor._process_images_task,
                images,
                input_text,
            )
        else:
            return self._process_images_task(images, input_text)

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]

        image_token = self.IMAGE_TOKEN
        base_output = self.load_images(
            input_ids, image_data, image_token, max_req_input_len
        )

        ret = await self._process_images(base_output.all_frames, base_output.input_text)

        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "pixel_values": ret["pixel_values"],
            "image_hashes": base_output.image_hashes,
            "modalities": request_obj.modalities or ["image"],
            "image_grid_thws": ret["image_grid_thws"],
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
        }


ImageProcessorMapping = {
    Qwen2VLForConditionalGeneration: Qwen2VLImageProcessor,
    Qwen2_5_VLForConditionalGeneration: Qwen2_5VLImageProcessor,
}
