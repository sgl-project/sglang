# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from sglang.srt.multimodal.internvl_utils import image_to_pixel_values
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.common import sample_video_frames

if TYPE_CHECKING:
    from decord import VideoReader

DEFAULT_NUM_TILES = 12
NUM_VIDEO_TILES = 1
DESIRED_FPS = 2  # TODO: allow desired fps/num frames to be configurable
MAX_FRAMES = 128


class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):
    models = [NemotronH_Nano_VL_V2]

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)
        Image.MAX_IMAGE_PIXELS = None
        self.image_size = hf_config.image_size
        self.VIDEO_CONTEXT_TOKEN = hf_config.video_context_token
        self.IMG_CONTEXT_TOKEN = hf_config.img_context_token
        self.IMG_START_TOKEN = hf_config.img_start_token
        self.IMG_END_TOKEN = hf_config.img_end_token
        self.num_image_token = int(
            (self.image_size // hf_config.patch_size) ** 2
            * (hf_config.downsample_ratio**2)
        )
        if hasattr(self._processor, "tokenizer"):
            tokenizer = self._processor.tokenizer
        else:
            tokenizer = self._processor
        self.tokenizer = tokenizer

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMG_CONTEXT_TOKEN,
            image_token_id=tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
            video_token=self.VIDEO_CONTEXT_TOKEN,
            video_token_id=tokenizer.convert_tokens_to_ids(self.VIDEO_CONTEXT_TOKEN),
        ).build(_image_processor)

        # Normalization config (mean/std) and tiling behavior
        self.norm_mean = hf_config.norm_mean
        self.norm_std = hf_config.norm_std
        self.use_thumbnail = hf_config.use_thumbnail

        self.PLACEHOLDER = self.tokenizer.unk_token
        assert isinstance(self.PLACEHOLDER, str)
        self.PLACEHOLDER_ID = tokenizer.convert_tokens_to_ids(self.PLACEHOLDER)
        assert isinstance(self.PLACEHOLDER_ID, int)

    def preprocess_image(
        self, image: Image.Image, *, max_num_tiles: int = DEFAULT_NUM_TILES
    ) -> torch.Tensor:
        return image_to_pixel_values(
            image,
            input_size=self.image_size,
            max_num_tiles=max_num_tiles,
            use_thumbnail=self.use_thumbnail,
            mean=self.norm_mean,
            std=self.norm_std,
        ).to(dtype=torch.bfloat16)

    def render_image(self, *, num_tiles: int):
        return f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * self.num_image_token * num_tiles}{self.IMG_END_TOKEN}"

    def render_frame(
        self, frame_index: int, *, timestamp: float, start_placeholder_token: str
    ):
        return f"Frame {frame_index + 1} sampled at {timestamp:.2f} seconds: {start_placeholder_token}{self.IMG_CONTEXT_TOKEN * self.num_image_token}{self.IMG_END_TOKEN}"

    @staticmethod
    def parse_video(video: "VideoReader") -> tuple[np.ndarray, list[float]]:
        frames = sample_video_frames(
            video, desired_fps=DESIRED_FPS, max_frames=MAX_FRAMES
        )
        video_array = video.get_batch(frames).asnumpy()
        # doing the `1000 /` and then `/ 1000` is to match vllm's timestamping *exactly*, for reference.
        frame_duration_ms = int(1000 / video.get_avg_fps())
        timestamps = [i * frame_duration_ms / 1000.0 for i in frames]
        return video_array, timestamps

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        prompt = input_text

        image_feature = None
        if base_output.images:
            preprocessed_images = [
                self.preprocess_image(image) for image in base_output.images
            ]
            rendered_images = [
                self.render_image(num_tiles=image.shape[0])
                for image in preprocessed_images
            ]
            prompt = prompt.replace(self.IMG_CONTEXT_TOKEN, "".join(rendered_images), 1)
            image_feature = torch.cat(preprocessed_images, dim=0)

        video_feature = None
        if base_output.videos:
            preprocessed_videos = []
            for video in base_output.videos:
                video_array, timestamps = self.parse_video(video)
                frames_tensors = [
                    self.preprocess_image(
                        Image.fromarray(frame, mode="RGB"),
                        max_num_tiles=NUM_VIDEO_TILES,
                    )
                    for frame in video_array
                ]
                preprocessed_video = torch.cat(frames_tensors, dim=0)
                preprocessed_videos.append(preprocessed_video)
                rendered_frames = [
                    self.render_frame(
                        i,
                        timestamp=timestamp,
                        start_placeholder_token=self.PLACEHOLDER,
                    )
                    for i, timestamp in enumerate(timestamps)
                ]
                prompt = prompt.replace(
                    self.VIDEO_CONTEXT_TOKEN, "".join(rendered_frames), 1
                )
            video_feature = torch.cat(preprocessed_videos, dim=0)

        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].flatten()
        offsets = self.get_mm_items_offset(prompt_ids, self.mm_tokens.image_token_id)
        img_offsets = [
            (start, end)
            for start, end in offsets
            if prompt_ids[start - 1] == self.img_start_token_id
        ]
        video_offsets = [
            (start, end)
            for start, end in offsets
            if prompt_ids[start - 1] == self.PLACEHOLDER_ID
        ]
        # Cleanup:
        prompt_ids[prompt_ids == self.PLACEHOLDER_ID] = self.img_start_token_id

        items = []
        if image_feature is not None:
            item = MultimodalDataItem(
                Modality.IMAGE, feature=image_feature, offsets=img_offsets
            )
            items.append(item)
        if video_feature is not None:
            item = MultimodalDataItem(
                Modality.VIDEO, feature=video_feature, offsets=video_offsets
            )
            items.append(item)

        return {
            "input_ids": prompt_ids.tolist(),
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.image_token_id,
        }
