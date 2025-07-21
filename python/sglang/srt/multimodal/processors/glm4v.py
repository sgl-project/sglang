import asyncio
import re
from typing import List, Union

from PIL import Image
from transformers.video_utils import VideoMetadata

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.qwen_vl import (
    preprocess_video,
    resize_image_async,
)


class Glm4vImageProcessor(SGLangBaseProcessor):
    models = [Glm4vForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

        # GLM-4V specific tokens
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
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28

        # Setup regex patterns
        self.IMAGE_TOKEN_REGEX = re.compile(
            r"<\|begin_of_image\|>(?:<\|image\|>)+<\|end_of_image\|>"
        )
        self.VIDEO_TOKEN_REGEX = re.compile(
            r"<\|begin_of_video\|>(?:<\|video\|>)+<\|end_of_video\|>"
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=self.IMAGE_TOKEN_REGEX,
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.VIDEO_TOKEN_ID,
            video_token_regex=self.VIDEO_TOKEN_REGEX,
        )

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
        print(base_output.videos)

        # # TODO(Xinyuan): GLM-4V specific: resize images, current methods are inherited from Qwen2.5VL
        # if base_output.images and isinstance(base_output.images[0], Image.Image):
        #     resize_tasks = [resize_image_async(image) for image in base_output.images]
        #     base_output.images = await asyncio.gather(*resize_tasks)

        # if base_output.videos:
        #     base_output.videos = [
        #         await preprocess_video(video) for video in base_output.videos
        #     ]

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=[
                self.IMAGE_START_TOKEN_ID,
                self.VIDEO_START_TOKEN_ID,
            ],
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

        mm_inputs = {
            "input_ids": input_ids.flatten().tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

        return mm_inputs
