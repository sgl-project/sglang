import asyncio
import math
import os
import re
from typing import List, Union

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import BaseImageProcessorFast

use_cv2 = True
try:
    import cv2
except:
    use_cv2 = False

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.mm_utils import FIFOTensorCache, reconstruct_tensor_from_infos
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.multimodal.mm_utils import (
    fast_image_hash,
    generate_reconstruct_cudatensor_infos,
    image_to_int,
    insert_input_ids,
    operate_substrings,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.utils import get_bool_env_var, get_int_env_var
from sglang.utils import logger

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
CACHED_IMAGE_MAX_MB_SIZE = 4096

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
    min_pixels = min_pixels
    max_pixels = max_pixels
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # default interpolation method of cv2 and PIL  both bilinear, but cv2 is much faster than pillow
    if height != resized_height or width != resized_width:
        if use_cv2:
            arr = np.array(image)  # convert PIL → NumPy
            resized = cv2.resize(
                arr, (resized_width, resized_height)
            )  # , interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(resized)
        else:
            image = image.resize((resized_width, resized_height))

    return image


def get_img_height_from_raw_img(img: Image, patch_pixel_size):
    resize_h, resize_w = smart_resize(
        img.size[0], img.size[1], IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS
    )
    ret = int((resize_h * resize_w) / patch_pixel_size)
    if ret < 1:
        raise ValueError("invalid image height")
    return ret, resize_h, resize_w


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


async def resize_image_async(
    image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    size_factor: int = IMAGE_FACTOR,
):
    return resize_image(image, min_pixels, max_pixels, size_factor)


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


# process video, qwen-specific
async def preprocess_video(
    vr,
    image_factor: int = IMAGE_FACTOR,
    # vr: VideoReader, image_factor: int = IMAGE_FACTOR
) -> torch.Tensor:
    ele = {}
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes({}, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


# Compatible with Qwen2VL and Qwen2_5VL
class Qwen2_5VLImageProcessor(SGLangBaseProcessor):
    models = [Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # The regex that matches expanded image tokens.
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200
        self.PATCH_SIZE = hf_config.vision_config.patch_size
        self.PATCH_PIXEL_NUMS = float(self.PATCH_SIZE) * self.PATCH_SIZE
        self.MERGE_PATCH_NUMS = (hf_config.vision_config.spatial_merge_size) ** 2
        self.IMG_PAD_TOKEN_ID = hf_config.image_token_id
        self.VISION_START_TOKEN_ID = hf_config.vision_start_token_id
        self.VISION_END_TOKEN_ID = hf_config.vision_end_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=hf_config.video_token_id,
        ).build(_processor)

        self.image_cache_table = FIFOTensorCache(CACHED_IMAGE_MAX_NUM)

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor
        """
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos

        processor = self._processor
        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessorFast)
            and not self.server_args.disable_fast_image_processor
        ):
            kwargs["device"] = "cuda"

        cache_mm_image_items = get_bool_env_var("SGL_CACHE_MM_IMAGE")
        is_qwen2_processor = isinstance(
            processor.image_processor,
            transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast,
        )

        if cache_mm_image_items and is_qwen2_processor:

            to_replace_str = "<|vision_start|><|image_pad|><|vision_end|>"
            repalce_str = "<|vision_end|>"
            v_start_token, img_pad_token, v_end_token = (
                self.VISION_START_TOKEN_ID,
                self.IMG_PAD_TOKEN_ID,
                self.VISION_END_TOKEN_ID,
            )

            img_hash_keys = kwargs.pop("img_hash_keys")
            img_heights = kwargs.pop("img_heights")
            new_processed_imgs = kwargs.pop("new_processed_imgs")
            new_processed_img_idxes = kwargs.pop("new_processed_img_idxes")
            img_token_nums = kwargs.pop("img_token_nums")
            remove_image_idx = kwargs.pop("remove_image_idx")
            image_grid_thw_lists = kwargs.pop("image_grid_thw_lists")
            processed_img_heights = []

            processed_text = operate_substrings(
                input_text, to_replace_str, remove_image_idx, repalce_str
            )
            kwargs["images"] = (
                new_processed_imgs if len(new_processed_imgs) != 0 else None
            )
            result = processor.__call__(
                text=[processed_text],
                padding=True,
                return_tensors="pt",
                **kwargs,
            )

            for feature_name in self.FEATURE_NAMES:
                if feature_name in result and isinstance(
                    result[feature_name], torch.Tensor
                ):
                    # not do D2H for pixel_values
                    if feature_name == "pixel_values":
                        continue
                    result[feature_name] = result[feature_name].to("cpu")

            start_height = 0
            end_height = 0
            tensor_lists = []
            used_hash_keys = set()
            for img_idx in range(len(images)):
                # cache Tensor
                if img_idx in new_processed_img_idxes:
                    img_height = img_heights[img_idx]
                    processed_img_heights.append(img_height)

                    start_height = end_height
                    end_height = start_height + img_height
                    to_cache_tensor = result["pixel_values"][start_height:end_height]
                    self.image_cache_table.add(
                        img_hash_keys[img_idx], to_cache_tensor
                    )

                    tensor_lists.append(to_cache_tensor)
                # add input ids and insert tensor
                else:
                    cached_tensor = self.image_cache_table.get(img_hash_keys[img_idx])
                    used_hash_keys.add(img_hash_keys[img_idx])
                    assert isinstance(
                        cached_tensor, torch.Tensor
                    ), "invalid cached_tensor"
                    tensor_lists.append(cached_tensor)
                    insert_cached_ids = (
                        [v_start_token]
                        + img_token_nums[img_idx] * [img_pad_token]
                        + [v_end_token]
                    )

                    result["input_ids"] = insert_input_ids(
                        result["input_ids"],
                        v_end_token,
                        img_pad_token,
                        insert_cached_ids,
                    )

            total_bytes = 0
            for ts in tensor_lists:
                total_bytes+= (ts.element_size() * ts.numel())
            
            total_MB = total_bytes // (1024 * 1024) + 1
            device_id = torch.cuda.current_device()
            device = torch.device(f"cuda:{device_id}")
            
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            
            #[NOTE]actually, torch reserved memory can also be used, here excluding torch reserved memory
            available_size_mb = (total - allocated - reserved) // (1024 * 1024)
            
            max_cache_image_size = CACHED_IMAGE_MAX_MB_SIZE
            if get_bool_env_var("SGL_TOKENIZER_CACHED_IMAGE_SIZE_MB"):
                max_cache_image_size = get_int_env_var("SGL_TOKENIZER_CACHED_IMAGE_SIZE_MB")
            else:
                logger.info("not set SGL_TOKENIZER_CACHED_IMAGE_SIZE_MB, use default value = {}".format(max_cache_image_size))
            
            if max_cache_image_size > available_size_mb:
                logger.info("max_cache_image_size {} mb over available size {} mb, set max cache size as {} mb".format(max_cache_image_size, available_size_mb, available_size_mb))
                max_cache_image_size = available_size_mb
            
            send_cudaipc_handle = True
            if max_cache_image_size < total_MB:
                logger.info("images data total size over max cache size, can not cache image datas for this request, send raw image instead of cudaipc-handle")
                send_cudaipc_handle = False
            
            if send_cudaipc_handle:
                proxy_pixel_values = generate_reconstruct_cudatensor_infos(tensor_lists)
                proxy_pixel_values["hash_keys"] = img_hash_keys
                proxy_pixel_values["cat_feature"] = True
                
                self.image_cache_table.pop_until(max_cache_image_size, used_hash_keys)
            else:
                proxy_pixel_values = torch.cat(tensor_lists).to("cpu")
            result["image_grid_thw"] = torch.Tensor(image_grid_thw_lists).to(
                torch.int64
            )

            result["pixel_values"] = proxy_pixel_values

        else:
            result = processor.__call__(
                text=[input_text],
                padding=True,
                return_tensors="pt",
                **kwargs,
            )
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:
                if feature_name in result and isinstance(
                    result[feature_name], torch.Tensor
                ):
                    result[feature_name] = result[feature_name].to("cpu")

        return result

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        cache_mm_image_items = get_bool_env_var("SGL_CACHE_MM_IMAGE")
        if cache_mm_image_items:
            images = base_output.images

            img_hash_keys = []
            img_heights = []
            new_processed_imgs = []
            new_processed_img_idxes = []
            img_token_nums = []
            remove_image_idx = []
            image_grid_thw_lists = []

            for img_idx in range(len(images)):
                hash_key = image_to_int(images[img_idx])
                img_hash_keys.append(hash_key)
                img_height, resize_h, resize_w = get_img_height_from_raw_img(
                    images[img_idx], self.PATCH_PIXEL_NUMS
                )
                img_token_num = int(img_height / self.MERGE_PATCH_NUMS)

                image_grid_thw_lists.append(
                    [
                        1,
                        int(resize_w // self.PATCH_SIZE),
                        int(resize_h // self.PATCH_SIZE),
                    ]
                )

                if img_token_num < 1:
                    raise ValueError("invalid img token num")

                if self.image_cache_table.get(hash_key) is None:
                    new_processed_img_idxes.append(img_idx)
                    new_processed_imgs.append(images[img_idx])
                else:
                    remove_image_idx.append(img_idx)

                img_heights.append(img_height)
                img_token_nums.append(img_token_num)

            # Qwen-specific: resize images if they are raw Image objects
            if len(new_processed_imgs) != 0 and isinstance(
                new_processed_imgs[0], Image.Image
            ):
                resize_tasks = [
                    resize_image_async(image) for image in new_processed_imgs
                ]
                new_processed_imgs = await asyncio.gather(*resize_tasks)

            if base_output.videos:
                base_output.videos = [
                    await preprocess_video(video) for video in base_output.videos
                ]

            args_dict = {
                "img_hash_keys": img_hash_keys,
                "img_heights": img_heights,
                "new_processed_imgs": new_processed_imgs,
                "new_processed_img_idxes": new_processed_img_idxes,
                "img_token_nums": img_token_nums,
                "remove_image_idx": remove_image_idx,
                "image_grid_thw_lists": image_grid_thw_lists,
            }

            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens, **args_dict
            )

        else:
            # Qwen-specific: resize images if they are raw Image objects
            if base_output.images and isinstance(base_output.images[0], Image.Image):
                resize_tasks = [
                    resize_image_async(image) for image in base_output.images
                ]
                base_output.images = await asyncio.gather(*resize_tasks)

            if base_output.videos:
                base_output.videos = [
                    await preprocess_video(video) for video in base_output.videos
                ]

            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.hf_config.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=getattr(ret, "second_per_grid_ts", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
