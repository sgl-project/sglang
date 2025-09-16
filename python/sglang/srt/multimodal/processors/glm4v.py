import math
import re
from typing import List, Union

import torch
import transformers
from decord import VideoReader
from transformers import BaseImageProcessorFast
from transformers.video_utils import VideoMetadata

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.mm_utils import FIFOTensorCache
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration
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
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import get_bool_env_var, get_int_env_var, logger

CACHED_IMAGE_MAX_MB_SIZE = 4096


# code from transformers
def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
):
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Glm4vImageProcessor(SGLangBaseProcessor):
    models = [Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # GLM-4.1V and GLM-4.5V specific tokens
        self.IMAGE_TOKEN = "<|image|>"
        self.VIDEO_TOKEN = "<|video|>"
        self.IMAGE_START_TOKEN = "<|begin_of_image|>"
        self.IMAGE_END_TOKEN = "<|end_of_image|>"
        self.VIDEO_START_TOKEN = "<|begin_of_video|>"
        self.VIDEO_END_TOKEN = "<|end_of_video|>"

        # Token IDs
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id
        self.VIDEO_START_TOKEN_ID = hf_config.video_start_token_id
        self.VIDEO_END_TOKEN_ID = hf_config.video_end_token_id

        # Vision config
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 112 * 112
        self.MAX_PIXELS = 30000 * 28 * 28 * 2

        self.PATCH_SIZE = 14
        self.TEMP_FACTOR = 2
        self.MERGE_NUM = 2

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            video_token=self.VIDEO_TOKEN,
            # Note: For GLM4v videos, it uses the video token before tokenization but uses image token after tokenization
            video_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

        self.image_cache_table = FIFOTensorCache()

    # adapted from https://github.com/huggingface/transformers/blob/369c99d0cea403b77bd0aef818527106453fd9fc/src/transformers/video_utils.py#L312
    async def preprocess_video(self, vr: VideoReader):
        """
        Preprocess video using VideoReader from Decord backend.

        Args:
            vr (VideoReader): VideoReader object from decord

        Returns:
            tuple: A tuple containing processed frames and metadata
        """
        video_fps = vr.get_avg_fps()
        total_num_frames = len(vr)
        duration = total_num_frames / video_fps if video_fps else 0

        # Extract all frames
        indices = list(range(total_num_frames))
        frames = vr.get_batch(indices).asnumpy()

        metadata = {
            "total_num_frames": int(total_num_frames),
            "fps": float(video_fps),
            "duration": float(duration),
            "video_backend": "decord",
            "frames_indices": indices,
        }

        return frames, metadata

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
        is_glm4v_processor = isinstance(
            processor.image_processor,
            transformers.models.glm4v.image_processing_glm4v_fast.Glm4vImageProcessorFast,
        )

        if cache_mm_image_items and is_glm4v_processor:

            to_replace_str = (
                self.IMAGE_START_TOKEN + self.IMAGE_TOKEN + self.IMAGE_END_TOKEN
            )
            repalce_str = self.IMAGE_END_TOKEN
            v_start_token, img_pad_token, v_end_token = (
                self.IMAGE_START_TOKEN_ID,
                self.IM_TOKEN_ID,
                self.IMAGE_END_TOKEN_ID,
            )

            img_hash_keys = kwargs.pop("img_hash_keys")
            target_height, target_width = kwargs.pop("target_shape")
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

            # start_height = 0
            # end_height = 0
            tensor_lists = []
            used_hash_keys = set()
            for img_idx in range(len(images)):
                # cache Tensor
                if img_idx in new_processed_img_idxes:
                    idx_in_new_processed = new_processed_img_idxes.index(img_idx)

                    to_cache_tensor = result["pixel_values"][idx_in_new_processed].contiguous()
                    self.image_cache_table.add(
                        img_hash_keys[img_idx],
                        to_cache_tensor
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
            
            
            
            # send_cudaipc_handle
            # 1. generate cuda-ipc infos
            # 2. set cat/stack mark for tensort list
            # 3. add hash_keys(to generate new hash)
            # 4. update cache
            if send_cudaipc_handle:
                proxy_pixel_values = generate_reconstruct_cudatensor_infos(tensor_lists)
                proxy_pixel_values["cat_feature"] = False
                proxy_pixel_values["hash_keys"] = img_hash_keys
                
                self.image_cache_table.pop_until(max_cache_image_size, used_hash_keys)
            # send raw data , move to cpu    
            else:
                proxy_pixel_values = torch.stack(tensor_lists).to("cpu")
            
            
            result["image_grid_thw"] = torch.Tensor(image_grid_thw_lists).to(
                    torch.int64
                )

            result["attention_mask"] = torch.ones_like(result["input_ids"])
            result["token_type_ids"] = torch.zeros_like(result["input_ids"])
            result["pixel_values"] = proxy_pixel_values

        else:
            result = processor.__call__(
                text=[input_text],
                padding=True,
                return_tensors="pt",
                **kwargs,
            )

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
        """
        logic of cache and send cache data:
        1. whether to deal a new image or not (not only hash code of image, but also image size should be same with former image)
        2. height and width of image should be resize and max ->
        3. do not cat the image, but just make the list as a tensor(need a mark to show whether cat or not)
        4. insert input id , modify token type and attention according the length of tensor
        5. when hash data, not only hash the pixel_values but also  shape after resized
        """
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        cache_mm_image_items = get_bool_env_var("SGL_CACHE_MM_IMAGE")
        is_glm4_5_v_processor = isinstance(
            self._processor.image_processor,
            transformers.models.glm4v.image_processing_glm4v_fast.Glm4vImageProcessorFast,
        )

        if cache_mm_image_items and is_glm4_5_v_processor:
            images = base_output.images

            img_hash_keys = []
            img_heights = []
            new_processed_imgs = []
            new_processed_img_idxes = []
            img_token_nums = []
            remove_image_idx = []
            image_grid_thw_lists = []

            target_height, target_width = 0, 0
            for image in images:
                height, width = smart_resize(
                    self.TEMP_FACTOR, image.size[0], image.size[1]
                )
                target_height, target_width = max(height, target_height), max(
                    width, target_width
                )

            for img_idx in range(len(images)):
                hash_key = str(image_to_int(images[img_idx]))
                target_shape_str = "_" + str(target_height) + "_" + str(target_width)
                hash_key += target_shape_str

                img_hash_keys.append(hash_key)
                image_height = (target_width // self.PATCH_SIZE) * (
                    target_height // self.PATCH_SIZE
                )
                # ref: transformers
                img_token_num = image_height // (self.MERGE_NUM * self.MERGE_NUM)

                # all images should hold same image_grid_thw
                image_grid_thw_lists.append(
                    [
                        1,
                        int(target_width // self.PATCH_SIZE),
                        int(target_height // self.PATCH_SIZE),
                    ]
                )

                if img_token_num < 1:
                    raise ValueError("invalid img token num")

                cached_item = self.image_cache_table.get(hash_key)

                if cached_item is None:
                    new_processed_img_idxes.append(img_idx)
                    new_processed_imgs.append(images[img_idx])
                else:
                    remove_image_idx.append(img_idx)

                img_heights.append(image_height)
                img_token_nums.append(img_token_num)

            #
            args_dict = {
                "img_hash_keys": img_hash_keys,
                "target_shape": (target_height, target_width),
                "new_processed_imgs": new_processed_imgs,
                "new_processed_img_idxes": new_processed_img_idxes,
                "img_token_nums": img_token_nums,
                "remove_image_idx": remove_image_idx,
                "image_grid_thw_lists": image_grid_thw_lists,
            }
        else:
            args_dict = {}

        video_metadata = None

        if base_output.videos:
            videos_processed = [
                await self.preprocess_video(video) for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))
            # transformer requires the video inputs to be under this format
            base_output.videos = [base_output.videos]
            video_metadata = [video_metadata]

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, video_metadata=video_metadata, **args_dict
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            attention_mask=getattr(ret, "attention_mask", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_inputs = {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

        return mm_inputs
