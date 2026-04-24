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

import logging
from math import sqrt

import numpy as np
import torch
from PIL import Image

from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from sglang.srt.models.parakeet import ParakeetExtractor
from sglang.srt.multimodal.evs import EVSProcessor
from sglang.srt.multimodal.internvl_utils import image_to_pixel_values
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.common import sample_video_frames

logger = logging.getLogger(__name__)

DEFAULT_NUM_TILES = 12
NUM_VIDEO_TILES = 1
DESIRED_FPS = 2  # TODO: allow desired fps/num frames to be configurable
MAX_FRAMES = 128


class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):
    models = [NemotronH_Nano_VL_V2]
    gpu_image_decode = (
        False  # NanoNemotronVL processes loaded image as PIL image explicitly
    )

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)
        self.evs = EVSProcessor(
            hf_config, {NemotronH_Nano_VL_V2_Config: NemotronH_Nano_VL_V2}
        )
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

        # Audio support: initialize Parakeet extractor if sound_config is present
        self.audio_extractor: ParakeetExtractor | None = None
        self.AUDIO_CONTEXT_TOKEN = getattr(
            hf_config, "audio_context_token", "<so_embedding>"
        )
        self.AUDIO_START_TOKEN = getattr(hf_config, "audio_start_token", "<so_start>")
        self.AUDIO_END_TOKEN = getattr(hf_config, "audio_end_token", "<so_end>")

        audio_token_str = None
        audio_token_id = None
        if getattr(hf_config, "sound_config", None) is not None:
            self.audio_extractor = ParakeetExtractor(hf_config.sound_config)
            audio_token_str = self.AUDIO_CONTEXT_TOKEN
            audio_token_id = tokenizer.convert_tokens_to_ids(self.AUDIO_CONTEXT_TOKEN)
            self.audio_start_token_id = tokenizer.convert_tokens_to_ids(
                self.AUDIO_START_TOKEN
            )
            self.audio_end_token_id = tokenizer.convert_tokens_to_ids(
                self.AUDIO_END_TOKEN
            )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMG_CONTEXT_TOKEN,
            image_token_id=tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
            video_token=self.VIDEO_CONTEXT_TOKEN,
            video_token_id=tokenizer.convert_tokens_to_ids(self.VIDEO_CONTEXT_TOKEN),
            audio_token=audio_token_str,
            audio_token_id=audio_token_id,
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

    def render_frame(self, frame_index: int, *, timestamp: float, num_tokens: int):
        return f"Frame {frame_index + 1} sampled at {timestamp:.2f} seconds: {self.PLACEHOLDER}{self.IMG_CONTEXT_TOKEN * num_tokens}{self.IMG_END_TOKEN}"

    @staticmethod
    def parse_video(video) -> tuple[np.ndarray, list[float]]:
        frames = sample_video_frames(
            video, desired_fps=DESIRED_FPS, max_frames=MAX_FRAMES
        )
        video_array = video.get_frames_at(frames)
        avg_fps = video.avg_fps
        if avg_fps > 0:
            frame_duration_ms = int(1000 / avg_fps)
        else:
            frame_duration_ms = 0
        timestamps = [i * frame_duration_ms / 1000.0 for i in frames]
        return video_array, timestamps

    def render_audio(self, *, num_tokens: int):
        return (
            f"{self.AUDIO_START_TOKEN}"
            f"{self.AUDIO_CONTEXT_TOKEN * num_tokens}"
            f"{self.AUDIO_END_TOKEN}"
        )

    async def process_mm_data_async(
        self, image_data, audio_data, input_text, request_obj, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=audio_data if self.audio_extractor else None,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
            audio_sample_rate=(
                self.audio_extractor.sampling_rate if self.audio_extractor else None
            ),
        )

        videos = [self.parse_video(video) for video in base_output.videos]

        rows = cols = int(sqrt(self.num_image_token))
        create_data_items, tokens_per_frame = self.evs.static_size_data_items(
            frames_per_video=[len(frames) for frames, _ in videos],
            num_images=len(base_output.images),
            rows=rows,
            cols=cols,
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
            for (video_array, timestamps), tpf in zip(
                videos, tokens_per_frame, strict=True
            ):
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
                        num_tokens=num_tokens,
                    )
                    for i, (timestamp, num_tokens) in enumerate(
                        zip(timestamps, tpf, strict=True)
                    )
                ]
                prompt = prompt.replace(
                    self.VIDEO_CONTEXT_TOKEN, "".join(rendered_frames), 1
                )
            video_feature = torch.cat(preprocessed_videos, dim=0)

        # Process audio data through the Parakeet feature extractor
        audio_items: list[MultimodalDataItem] = []
        if base_output.audios and self.audio_extractor is not None:
            extractor = self.audio_extractor
            for audio in base_output.audios:
                num_tokens = extractor.audio_token_count(len(audio))
                rendered = self.render_audio(num_tokens=num_tokens)
                prompt = prompt.replace(self.AUDIO_CONTEXT_TOKEN, rendered, 1)

            all_audios = list(base_output.audios)
            extracted = extractor(
                all_audios,
                sampling_rate=extractor.sampling_rate,
                return_tensors="pt",
            )
            input_features = extracted.input_features
            attention_mask = extracted.attention_mask
            clip_counts = extracted.audio_num_clips

            clip_offset = 0
            for audio_idx, num_clips in enumerate(clip_counts):
                audio_features = input_features[clip_offset : clip_offset + num_clips]
                audio_mask = attention_mask[clip_offset : clip_offset + num_clips]
                clip_offset += num_clips
                audio_items.append(
                    MultimodalDataItem(
                        modality=Modality.AUDIO,
                        feature=audio_features,
                        model_specific_data={
                            "feature_attention_mask": audio_mask,
                            "audio_num_clips": num_clips,
                        },
                    )
                )

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

        # Compute audio offsets
        if audio_items:
            audio_token_id = self.mm_tokens.audio_token_id
            audio_offsets_list = self.get_mm_items_offset(prompt_ids, audio_token_id)
            for item, offset in zip(audio_items, audio_offsets_list):
                item.offsets = [offset]

        prompt_ids_list = prompt_ids.tolist()

        items = create_data_items(
            image=image_feature,
            image_offsets=img_offsets,
            video=video_feature,
            video_offsets=video_offsets,
            input_ids_list=prompt_ids_list,
        )
        items.extend(audio_items)

        return MultimodalProcessorOutput(
            input_ids=prompt_ids_list,
            mm_items=items,
            im_start_id=self.img_start_token_id,
            im_end_id=self.img_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.image_token_id,
            audio_token_id=self.mm_tokens.audio_token_id if audio_items else None,
            audio_start_id=(self.audio_start_token_id if audio_items else None),
            audio_end_id=(self.audio_end_token_id if audio_items else None),
        )
