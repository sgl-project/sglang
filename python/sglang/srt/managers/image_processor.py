# TODO: also move pad_input_ids into this module
import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
import transformers

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import load_image
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

global global_processor


def init_global_processor(server_args: ServerArgs):
    """Init the global processor for multi modal models."""
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )


class BaseImageProcessor(ABC):
    @abstractmethod
    async def process_images_async(self, image_data, **kwargs):
        pass


class DummyImageProcessor(BaseImageProcessor):
    async def process_images_async(self, *args, **kwargs):
        return None


class LlavaImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _image_processor):
        self.hf_config = hf_config
        self._image_processor = _image_processor
        self.executor = concurrent.futures.ProcessPoolExecutor(
            initializer=init_global_processor,
            mp_context=mp.get_context("fork"),
            initargs=(server_args,),
            max_workers=os.environ.get("SGLANG_CPU_COUNT", os.cpu_count()),
        )

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        image_processor=None,
    ):
        image_processor = image_processor or global_processor.image_processor

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
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                LlavaImageProcessor._process_single_image_task,
                image_data,
                aspect_ratio,
                grid_pinpoints,
            )
        else:
            return self._process_single_image_task(
                image_data, aspect_ratio, grid_pinpoints
            )

    async def process_images_async(
        self, image_data: List[Union[str, bytes]], request_obj
    ):
        if not image_data:
            return None

        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints")
            and "anyres" in aspect_ratio
            else None
        )

        if isinstance(image_data, list) and len(image_data) > 0:
            # Multiple images
            if len(image_data) > 1:
                aspect_ratio = "pad"  # LLaVA OneVision Handling: more than one image --> interleaved image mode or video mode. We do not use anyres
                pixel_values, image_hashes, image_sizes = [], [], []
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
                    image_hashes.append(image_h)
                    image_sizes.append(image_s)

                if isinstance(pixel_values[0], np.ndarray):
                    pixel_values = np.stack(pixel_values, axis=0)
            else:
                # A single image
                pixel_values, image_hash, image_size = await self._process_single_image(
                    image_data[0], aspect_ratio, grid_pinpoints
                )
                image_hashes = [image_hash]
                image_sizes = [image_size]
        elif isinstance(image_data, str):
            # A single image
            pixel_values, image_hash, image_size = await self._process_single_image(
                image_data, aspect_ratio, grid_pinpoints
            )
            image_hashes = [image_hash]
            image_sizes = [image_size]
        else:
            raise ValueError(f"Invalid image data: {image_data}")

        return {
            "pixel_values": pixel_values,
            "image_hashes": image_hashes,
            "image_sizes": image_sizes,
            "modalities": request_obj.modalities,
        }


class MolmoImageProcessor(BaseImageProcessor):
    SPECIAL_TOKEN_TO_ID = {
        "<im_patch>": 152066,
        "<im_start>": 152064,
        "<im_end>": 152065,
        "<im_col>": 152067,
        "<|image|>": 152068,
    }

    def __init__(self, hf_config, server_args, _image_processor):
        self.hf_config = hf_config
        self._image_processor = _image_processor
        self.executor = concurrent.futures.ProcessPoolExecutor(
            initializer=init_global_processor,
            mp_context=mp.get_context("fork"),
            initargs=(server_args,),
            max_workers=os.environ.get("SGLANG_CPU_COUNT", os.cpu_count()),
        )
        self.image_patch_token_id = self.SPECIAL_TOKEN_TO_ID["<im_patch>"]
        self.image_start_token_id = self.SPECIAL_TOKEN_TO_ID["<im_start>"]
        self.image_end_token_id = self.SPECIAL_TOKEN_TO_ID["<im_end>"]
        self.image_col_token_id = self.SPECIAL_TOKEN_TO_ID["<im_col>"]
        self.image_prompt_token_id = self.SPECIAL_TOKEN_TO_ID["<|image|>"]

    @staticmethod
    def _process_image_task(
        image_data_list: List[Union[str, bytes]],
        input_ids: List[int],
        image_patch_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        image_col_token_id: int,
    ):
        global global_processor

        # Adapted from https://huggingface.co/allenai/Molmo-7B-D-0924/blob/main/preprocessing_molmo.py
        # Returns:
        #   input_ids
        #   image_input_idx
        #   images
        #   image_masks
        images = []
        image_sizes = []
        image_hashes = []
        for image_data in image_data_list:
            image, image_size = load_image(image_data)
            image = image.convert("RGB")
            image_hashes.append(hash(image_data))
            images.append(np.array(image))
            image_sizes.append(image_size)
        hf_dict = global_processor.image_processor.multimodal_preprocess(
            images=images,
            image_idx=[-1] * len(images),
            tokens=np.asarray(input_ids).astype(np.int32),
            sequence_length=len(input_ids),
            image_patch_token_id=image_patch_token_id,
            image_col_token_id=image_col_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
        )

        bos = (
            global_processor.tokenizer.bos_token_id
            or global_processor.tokenizer.eos_token_id
        )
        decoder_input_tokens = np.pad(
            hf_dict["input_ids"], [[1, 0]], constant_values=bos
        )
        hf_dict["input_ids"] = decoder_input_tokens
        if "image_input_idx" in hf_dict:
            # Shift patch mapping up by one since we added BOS
            image_input_idx = hf_dict["image_input_idx"]
            hf_dict["image_input_idx"] = np.where(
                image_input_idx < 0, image_input_idx, image_input_idx + 1
            )

        for k, v in hf_dict.items():
            hf_dict[k] = torch.from_numpy(v)

        hf_dict["image_hashes"] = image_hashes
        hf_dict["pixel_values"] = hf_dict["images"]
        hf_dict["image_sizes"] = image_sizes

        del hf_dict["images"]

        return hf_dict

    async def _process_image(
        self, image_data_list: List[Union[bytes, str]], input_ids: List[int]
    ):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                MolmoImageProcessor._process_image_task,
                image_data_list,
                input_ids,
                self.image_patch_token_id,
                self.image_start_token_id,
                self.image_end_token_id,
                self.image_col_token_id,
            )
        else:
            return self._process_image_task(
                image_data_list,
                input_ids,
                self.image_patch_token_id,
                self.image_start_token_id,
                self.image_end_token_id,
                self.image_col_token_id,
            )

    async def process_images_async(self, image_data, request_obj, **kwargs):
        if not image_data:
            return None

        input_ids = request_obj.input_ids
        res = {}
        if isinstance(image_data, list) and len(image_data) > 0:
            # Multiple images
            if len(image_data) > 1:
                res = await self._process_image(image_data, input_ids)
            else:
                res = await self._process_image(image_data[0:1], input_ids)
        elif isinstance(image_data, str):
            # A single image
            res = await self._process_image([image_data], input_ids)
        else:
            raise ValueError(f"Invalid image data: {image_data}")

        res["modalities"] = request_obj.modalities
        return res


def get_image_processor(
    hf_config, server_args: ServerArgs, _image_processor
) -> BaseImageProcessor:
    if "MolmoForCausalLM" in hf_config.architectures:
        return MolmoImageProcessor(hf_config, server_args, _processor)
    return LlavaImageProcessor(hf_config, server_args, _processor)


def get_dummy_image_processor():
    return DummyImageProcessor()
