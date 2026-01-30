import math
import os
from typing import List, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import BaseImageProcessorFast

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.ernie45_vl import Ernie4_5_VLMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.utils import get_bool_env_var, is_npu, logger

_is_npu = is_npu()

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        f"Ignoring and using default."
    )
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 299 * 28 * 28
VIDEO_MAX_PIXELS = 1196 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 16
FPS_MAX_FRAMES = 180


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    if max(height, width) / min(height, width) > MAX_RATIO:
        if height > width:
            new_width = max(factor, round_by_factor(width, factor))
            new_height = floor_by_factor(new_width * MAX_RATIO, factor)
        else:
            new_height = max(factor, round_by_factor(height, factor))
            new_width = floor_by_factor(new_height * MAX_RATIO, factor)

        height = new_height
        width = new_width

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

    if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:
        raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")

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
    image = image.resize((resized_width, resized_height), resample=RESIZE_RESAMPLE)
    return image


def round_by_factor(number: int | float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int | float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int | float, factor: int) -> int:
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
) -> torch.Tensor:

    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes({}, total_frames=total_frames, video_fps=video_fps)
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)
    video_np = vr.get_batch(idx).asnumpy()
    video = torch.from_numpy(video_np).pin_memory()
    video = video.permute(0, 3, 1, 2)  # Convert to TCHW format
    nframes, _, height, width = video.shape
    min_pixels = VIDEO_MIN_PIXELS
    total_pixels = VIDEO_TOTAL_PIXELS
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )

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
        interpolation=InterpolationMode.BILINEAR,
    )

    video = video.permute(0, 2, 3, 1)
    video = video.pin_memory()
    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,
        "video_backend": "torchvision",
    }

    return video, video_metadata


# Compatible with Ernie-VL Series
class Ernie4_5_VLImageProcessor(SGLangBaseProcessor):
    models = [Ernie4_5_VLMoeForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.hf_config = hf_config
        self.model_type = hf_config.model_type
        self.image_start_token_id = hf_config.image_start_token_id
        self.image_end_token_id = hf_config.image_end_token_id
        self.video_start_token_id = hf_config.video_start_token_id
        self.video_end_token_id = hf_config.video_end_token_id

        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>",
            video_token="<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>",
            image_token_id=hf_config.im_patch_id,
            video_token_id=hf_config.im_patch_id,  # image and video use the same token_id
        ).build(_processor)

        self.tokenizer = self._processor.tokenizer
        self.image_processor = self._processor.image_processor

    def _pixel_values_norm(
        self,
        pixel_values: torch.Tensor,
        mm_kwargs: object,
    ) -> torch.Tensor:
        hf_config = self.hf_config
        vision_config = hf_config.vision_config
        image_processor = self.image_processor
        image_mean_tensor = torch.tensor(
            image_processor.image_mean, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        image_std_tensor = torch.tensor(
            image_processor.image_std, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        rescale_factor = torch.tensor(
            image_processor.rescale_factor, dtype=torch.float32
        )
        patch_size_squared = vision_config.patch_size**2

        image_mean_tensor = image_mean_tensor.squeeze([-2, -1]).repeat_interleave(
            patch_size_squared, -1
        )
        image_std_tensor = image_std_tensor.squeeze([-2, -1]).repeat_interleave(
            patch_size_squared, -1
        )

        if not image_mean_tensor.is_contiguous():
            image_mean_tensor = image_mean_tensor.contiguous()
        if not image_std_tensor.is_contiguous():
            image_std_tensor = image_std_tensor.contiguous()

        pixel_values = (
            rescale_factor * pixel_values.to(torch.float32) - image_mean_tensor
        ) / image_std_tensor
        pixel_values = pixel_values.to(hf_config.dtype)
        return pixel_values

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
            if not _is_npu:
                kwargs["device"] = "cuda"

        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        # Divide the processor_output into two modalities: image and video.
        if result is not None:
            pixel_values = result["images"]
            if pixel_values is not None:
                result["images"] = self._pixel_values_norm(pixel_values, kwargs)
            for key in list(result.keys()):
                if result[key] is None:
                    del result[key]
                    continue
                if key == "grid_thw":
                    grid_thw = result["grid_thw"]
                    pixel_values_all = result["images"]
                    # Identify elements where the first
                    # dimension is greater than 1 and
                    # treat them as the video modality
                    mask = grid_thw[:, 0] > 1
                    result["video_grid_thw"] = grid_thw[mask]
                    result["image_grid_thw"] = grid_thw[~mask]
                    image_patch_num = result["image_grid_thw"].prod(dim=1).sum()
                    result["pixel_values"] = pixel_values_all[:image_patch_num]
                    result["pixel_values_videos"] = pixel_values_all[image_patch_num:]
                    del result["images"]
                    del result["grid_thw"]

                    # del empty result
                    if result["image_grid_thw"].numel() == 0:
                        del result["image_grid_thw"]
                    if result["pixel_values"].numel() == 0:
                        del result["pixel_values"]
                    if result["video_grid_thw"].numel() == 0:
                        del result["video_grid_thw"]
                    if result["pixel_values_videos"].numel() == 0:
                        del result["pixel_values_videos"]

        if not self.server_args.keep_mm_feature_on_device:
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:
                if SGL_USE_CUDA_IPC:
                    pass
                else:
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
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        # resize images if they are raw Image objects
        resized_images = []
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            for image in base_output.images:
                resized_image = resize_image(image)
                resized_images.append(resized_image)
            base_output.images = resized_images

        if base_output.videos:
            videos_processed = [
                await preprocess_video(video) for video in base_output.videos
            ]
            base_output.videos, _ = map(list, zip(*videos_processed))

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_ernie45(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        assert (
            input_ids.shape[0] == mrope_positions.shape[-1]
        ), "input_ids and mrope_positions should have the same length"

        mm_inputs = {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.image_start_token_id,
            "im_end_id": self.image_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

        return mm_inputs
