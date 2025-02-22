# TODO: also move pad_input_ids into this module
import asyncio
import concurrent.futures
import dataclasses
import importlib
import logging
import multiprocessing as mp
import os
import pkgutil
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
import PIL
import transformers
from decord import VideoReader, cpu
from PIL import Image
from transformers import IMAGE_PROCESSOR_MAPPING

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.models.llava import LlavaMistralForCausalLM, LlavaQwenForCausalLM
from sglang.srt.models.llavavid import LlavaVidForCausalLM
from sglang.srt.models.minicpmv import MiniCPMV
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import load_image
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

global global_processor


def init_global_processor(sglang_image_processor, server_args: ServerArgs):
    """Init the global processor for multi-modal models."""
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = sglang_image_processor._build_processor(server_args=server_args)


def get_global_processor():
    global global_processor
    return global_processor


@dataclasses.dataclass
class BaseImageProcessorOutput:
    image_hashes: list[int]
    image_sizes: list[int]
    all_frames: [PIL.Image]
    # input_text, with each frame of video/image represented with a image_token
    input_text: str


class BaseImageProcessor(ABC):
    def __init__(self, hf_config, server_args, _processor):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.executor = concurrent.futures.ProcessPoolExecutor(
            initializer=init_global_processor,
            initargs=(
                self,
                server_args,
            ),
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_COUNT", os.cpu_count())),
        )

    def _build_processor(self, server_args):
        """Init the global processor for multi modal models."""
        return get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )

    @abstractmethod
    async def process_images_async(
        self, image_data, input_text, max_req_input_len, **kwargs
    ):
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input
        """
        # Before processing inputs
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                # Estimate frames for the video
                vr = VideoReader(path, ctx=cpu(0))
                num_frames = len(vr)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    def encode_video(self, video_path, frame_count_limit=None):
        if not os.path.exists(video_path):
            logger.error(f"Video {video_path} does not exist")
            return []

        if frame_count_limit == 0:
            return []

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if frame_count_limit is not None and len(frame_idx) > frame_count_limit:
            frame_idx = uniform_sample(frame_idx, frame_count_limit)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        return frames

    def load_images(
        self,
        input_ids: list,
        image_data,
        image_token: str,
        max_req_input_len: int,
        preserve_alpha_channel=False,
    ) -> BaseImageProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:

            preserve_alpha_channel: if True, preserve the alpha channel in the returned images

        """
        image_hashes, image_sizes = [], []
        all_frames = []
        new_text_parts = []

        if isinstance(input_ids, list):
            assert len(input_ids) and isinstance(input_ids[0], int)
            input_text = self._processor.tokenizer.decode(input_ids)
        else:
            input_text = input_ids

        text_parts = input_text.split(image_token)

        # roughly calculate the max number of frames under the max_req_input_len limit
        def calculate_max_num_frames() -> int:
            ret = (max_req_input_len - len(input_ids)) // self.NUM_TOKEN_PER_FRAME
            return min(ret, 100)

        MAX_NUM_FRAMES = calculate_max_num_frames()
        estimated_frames_list = self.get_estimated_frames_list(image_data=image_data)
        total_frame_count = sum(estimated_frames_list)
        # a heuristic value, suggesting the maximum fraction of frames to embed from all visual inputs.
        # e.g., 0.1 suggests that 1 frame out of 10 input frames should be used
        scaling_factor = min(1.0, MAX_NUM_FRAMES / total_frame_count)

        # Process each input with allocated frames
        for image_index, (image, estimated_frames) in enumerate(
            zip(image_data, estimated_frames_list)
        ):
            if len(all_frames) >= MAX_NUM_FRAMES:
                frames_to_process = 0
            else:
                frames_to_process = max(1, int(estimated_frames * scaling_factor))

            if frames_to_process == 0:
                frames = []
            else:
                try:
                    if isinstance(image, str) and image.startswith("video:"):
                        path = image[len("video:") :]
                        frames = self.encode_video(
                            path, frame_count_limit=frames_to_process
                        )
                    else:
                        raw_image, _size = load_image(image)
                        if preserve_alpha_channel == False:
                            raw_image = raw_image.convert("RGB")
                        frames = [raw_image]
                    if len(frames) == 0:
                        continue
                except FileNotFoundError as e:
                    print(e)
                    return None
                image_sizes += frames[0].size * len(frames)
                image_hashes += [hash(image)] * len(frames)
                all_frames += frames

            new_text_parts.append(text_parts[image_index])
            if frames_to_process != 0:
                new_text_parts.append(image_token * len(frames))
            assert frames_to_process == len(frames)

        new_text_parts.append(text_parts[-1])

        input_text = "".join(new_text_parts)
        return BaseImageProcessorOutput(
            image_hashes, image_sizes, all_frames, input_text
        )


class DummyImageProcessor(BaseImageProcessor):
    def __init__(self):
        pass

    async def process_images_async(self, *args, **kwargs):
        return None


class LlavaImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        image_aspect_ratio: Optional[str] = None,
        image_grid_pinpoints: Optional[str] = None,
        image_processor=None,
    ):
        processor = global_processor

        image_processor = image_processor or processor.image_processor

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
        else:
            raise ValueError(f"Invalid image data: {image_data}")

        return {
            "pixel_values": pixel_values,
            "image_hashes": image_hashes,
            "image_sizes": image_sizes,
            "modalities": request_obj.modalities or ["image"],
        }


class MllamaImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    @staticmethod
    def _process_single_image_task(images, input_text):
        # input_ids', 'attention_mask', 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask', 'cross_attention_mask'
        return global_processor(images, input_text, return_tensors="pt")

    async def _process_single_image(self, images, input_text):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                MllamaImageProcessor._process_single_image_task,
                images,
                input_text,
            )
        else:
            image_inputs = self._processor(images, input_text, return_tensors="pt")

        return image_inputs

    async def process_images_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        if not image_data:
            return None

        if isinstance(input_text, list):
            assert len(input_text) and isinstance(input_text[0], int)
            input_text = self._processor.tokenizer.decode(input_text)

        if not isinstance(image_data, list):
            image_data = [image_data]

        if len(image_data) > 0:
            images = [load_image(image)[0] for image in image_data]
        else:
            images = load_image(image_data[0])[0]

        image_inputs = await self._process_single_image(images, input_text)
        image_inputs["image_hashes"] = [hash(str(image_data))]
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()[0]

        return image_inputs


class MiniCPMVImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "(<image>./</image>)"

    @staticmethod
    def _process_images_task(images, input_text):
        processor = global_processor
        result = processor.__call__(text=input_text, images=images, return_tensors="pt")
        return {
            "input_ids": result.input_ids,
            "pixel_values": result.pixel_values,
            "tgt_sizes": result.tgt_sizes,
        }

    async def _process_images(self, images, input_text):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                MiniCPMVImageProcessor._process_images_task,
                images,
                input_text,
            )
        else:
            image_inputs = self._processor(
                images=images, text=input_text, return_tensors="pt"
            )

        return image_inputs

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
    ):
        if not image_data:
            return None
        if not isinstance(image_data, list):
            image_data = [image_data]

        base_output = self.load_images(
            input_ids, image_data, self.IMAGE_TOKEN, max_req_input_len
        )
        if base_output is None:
            return None

        if len(base_output.all_frames) == 0:
            return None
        res = await self._process_images(
            images=base_output.all_frames, input_text=base_output.input_text
        )

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        im_start_id = tokenizer.im_start_id
        im_end_id = tokenizer.im_end_id
        if tokenizer.slice_start_id:
            slice_start_id = tokenizer.slice_start_id
            slice_end_id = tokenizer.slice_end_id
        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            "pixel_values": res["pixel_values"],
            "tgt_sizes": res["tgt_sizes"],
            "image_hashes": base_output.image_hashes,
            "modalities": request_obj.modalities or ["image"],
            "im_start_id": im_start_id,
            "im_end_id": im_end_id,
            "slice_start_id": slice_start_id,
            "slice_end_id": slice_end_id,
        }


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
        image_processor = image_processor or global_processor.image_processor

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
        elif isinstance(image_data, str):
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

    @staticmethod
    def _process_images_task(images, input_text):
        result = global_processor.__call__(
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


IMAGE_PROCESSOR_MAPPING = {
    LlavaVidForCausalLM: LlavaImageProcessor,
    LlavaQwenForCausalLM: LlavaImageProcessor,
    LlavaMistralForCausalLM: LlavaImageProcessor,
    MllamaForConditionalGeneration: MllamaImageProcessor,
    Qwen2VLForConditionalGeneration: Qwen2VLImageProcessor,
    Qwen2_5_VLForConditionalGeneration: Qwen2_5VLImageProcessor,
    MiniCPMV: MiniCPMVImageProcessor,
}


def get_image_processor(
    hf_config, server_args: ServerArgs, processor
) -> BaseImageProcessor:
    for model_cls, processor_cls in IMAGE_PROCESSOR_MAPPING.items():
        if model_cls.__name__ in hf_config.architectures:
            return processor_cls(hf_config, server_args, processor)
    raise ValueError(
        f"No image processor found for architecture: {hf_config.architectures}"
    )


def get_dummy_image_processor():
    return DummyImageProcessor()


@lru_cache()
def import_image_processors():
    package_name = "sglang.srt.managers.image_processors"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}. " f"{e}")
                continue
            if hasattr(module, "ImageProcessorMapping"):
                entry = module.ImageProcessorMapping
                if isinstance(entry, dict):
                    for processor_name, cls in entry.items():
                        IMAGE_PROCESSOR_MAPPING[processor_name] = cls


import_image_processors()
