# TODO: also move pad_input_ids into this module
import asyncio
import concurrent.futures
import dataclasses
import logging
import math
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
import transformers
from decord import VideoReader, cpu
from PIL import Image

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
            mp_context=mp.get_context("fork"),
            initargs=(server_args,),
            max_workers=int(os.environ.get("SGLANG_CPU_COUNT", os.cpu_count())),
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

    @staticmethod
    def encode_video(video_path, frame_count_limit=None):
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
        frame_indices = [i for i in range(0, len(vr), sample_fps)]
        if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
            frame_indices = uniform_sample(frame_indices, frame_count_limit)

        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        return frames

    def load_images(
        self,
        input_ids: list,
        image_data,
        image_token: str,
        max_req_input_len: int,
        return_text: Optional[bool] = True,
    ) -> BaseImageProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token
        """
        image_hashes, image_sizes = [], []
        all_frames = []
        new_text_parts = []

        if isinstance(input_ids, list) and return_text:
            assert len(input_ids) and isinstance(input_ids[0], int)
            input_text = self._processor.tokenizer.decode(input_ids)
        else:
            input_text = input_ids

        if return_text:
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
                max_frames_to_process = 0
            else:
                max_frames_to_process = max(1, int(estimated_frames * scaling_factor))

            if max_frames_to_process == 0:
                frames = []
            else:
                try:
                    if isinstance(image, str) and image.startswith("video:"):
                        path = image[len("video:") :]
                        frames = BaseImageProcessor.encode_video(
                            path, frame_count_limit=max_frames_to_process
                        )
                    else:
                        raw_image, _size = load_image(image)
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

            if return_text:
                new_text_parts.append(text_parts[image_index])
            if max_frames_to_process != 0:
                new_text_parts.append(image_token * len(frames))
            assert max_frames_to_process >= len(frames)
        if return_text:
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
        self.IMAGE_TOKEN = "<image>"

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
                pixel_values = image_processor(image)["pixel_values"]
                for _ in range(len(pixel_values)):
                    pixel_values[_] = pixel_values[_].astype(np.float16)
                pixel_values = np.stack(pixel_values, axis=0)
                return {"pixel_values": pixel_values}
            else:
                # It is an image
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

                return {"pixel_values": pixel_values}
        except Exception:
            logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())

    async def _process_single_image(
        self, image_data: Union[bytes, str], aspect_ratio: str, grid_pinpoints: str
    ) -> dict:
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

        base_output = self.load_images(
            input_ids,
            image_data,
            self.IMAGE_TOKEN,
            max_req_input_len,
            return_text=False,
        )

        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints")
            and "anyres" in aspect_ratio
            else None
        )

        res = []
        for img_data in image_data:
            res.append(
                self._process_single_image(img_data, aspect_ratio, grid_pinpoints)
            )

        res = await asyncio.gather(*res)
        pixel_values = [r["pixel_values"] for r in res]
        pixel_values = np.stack(pixel_values, axis=0)
        return {
            "pixel_values": pixel_values,
            "image_hashes": base_output.image_hashes,
            "image_sizes": base_output.image_sizes,
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
        result = global_processor.__call__(
            text=input_text, images=images, return_tensors="pt"
        )
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
            input_ids,
            image_data,
            self.IMAGE_TOKEN,
            max_req_input_len,
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
        im_start_id = [tokenizer.im_start_id]
        im_end_id = [tokenizer.im_end_id]
        if tokenizer.slice_start_id:
            slice_start_id = [tokenizer.slice_start_id]
            slice_end_id = [tokenizer.slice_end_id]
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


# Compatible with Qwen2VL and Qwen2_5VL
class Qwen2_5VLImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200

    @staticmethod
    def _process_images_task(images, input_text, hf_config):

        result = global_processor.__call__(
            text=[input_text], images=images, padding=True, return_tensors="pt"
        )

        image_grid_thw = result.image_grid_thw if "image_grid_thw" in result else None
        pixel_values = result.pixel_values

        return {
            "input_ids": result.input_ids,
            "pixel_values": pixel_values,
            "image_grid_thws": image_grid_thw,
            "second_per_grid_ts": (
                result.second_per_grid_ts if "second_per_grid_ts" in result else None
            ),
            "video_grid_thws": (
                result.video_grid_thws if "video_grid_thws" in result else None
            ),
        }

    async def _process_images(self, images, input_text) -> dict:
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Qwen2_5VLImageProcessor._process_images_task,
                images,
                input_text,
                self.hf_config,
            )
        else:
            return self._process_images_task(images, input_text, self.hf_config)

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
            input_ids,
            image_data,
            image_token,
            max_req_input_len,
        )

        def resize_image(image, size_factor: int = self.IMAGE_FACTOR) -> Image.Image:
            width, height = image.size
            min_pixels = self.MIN_PIXELS
            max_pixels = self.MAX_PIXELS
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            return image

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
            factor: int = self.IMAGE_FACTOR,
            min_pixels: int = self.MIN_PIXELS,
            max_pixels: int = self.MAX_PIXELS,
        ) -> tuple[int, int]:
            """
            Rescales the image so that the following conditions are met:

            1. Both dimensions (height and width) are divisible by 'factor'.

            2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

            3. The aspect ratio of the image is maintained as closely as possible.
            """
            if max(height, width) / min(height, width) > self.MAX_RATIO:
                raise ValueError(
                    f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
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

        images = [resize_image(image) for image in base_output.all_frames]

        print(f"text:{base_output.input_text}")

        ret = await self._process_images(images, base_output.input_text)

        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "pixel_values": ret["pixel_values"],
            "image_hashes": base_output.image_hashes,
            "modalities": request_obj.modalities or ["image"],
            "image_grid_thws": ret["image_grid_thws"],
            "video_grid_thws": ret["video_grid_thws"],
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "second_per_grid_ts": ret["second_per_grid_ts"],
        }


def get_image_processor(
    hf_config, server_args: ServerArgs, processor
) -> BaseImageProcessor:
    if "MllamaForConditionalGeneration" in hf_config.architectures:
        return MllamaImageProcessor(hf_config, server_args, processor)
    elif "Qwen2VLForConditionalGeneration" in hf_config.architectures:
        return Qwen2_5VLImageProcessor(hf_config, server_args, processor)
    elif "Qwen2_5_VLForConditionalGeneration" in hf_config.architectures:
        return Qwen2_5VLImageProcessor(hf_config, server_args, processor)
    elif "MiniCPMV" in hf_config.architectures:
        return MiniCPMVImageProcessor(hf_config, server_args, processor)
    else:
        return LlavaImageProcessor(hf_config, server_args, processor.image_processor)


def get_dummy_image_processor():
    return DummyImageProcessor()
