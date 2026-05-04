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
import math
from math import sqrt

import numpy as np
import torch
from PIL import Image

from sglang.srt.configs.nano_nemotron_vl import (
    NemotronH_Nano_Omni_Reasoning_V3_Config,
    NemotronH_Nano_VL_V2_Config,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.nano_nemotron_vl import (
    NemotronH_Nano_Omni_Reasoning_V3,
    NemotronH_Nano_VL_V2,
)
from sglang.srt.models.parakeet import ParakeetExtractor
from sglang.srt.multimodal.audio_from_video import extract_audio_from_video_bytes
from sglang.srt.multimodal.evs import EVSProcessor
from sglang.srt.multimodal.internvl_utils import (
    compute_budgeted_image_sizes,
    get_video_target_size_and_feature_size,
    image_to_pixel_values,
    resize_image_to_pixels,
    video_to_pixel_values,
)
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
    models = [NemotronH_Nano_VL_V2, NemotronH_Nano_Omni_Reasoning_V3]
    gpu_image_decode = (
        False  # NanoNemotronVL processes loaded image as PIL image explicitly
    )

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)
        self.evs = EVSProcessor(
            hf_config,
            {
                NemotronH_Nano_VL_V2_Config: NemotronH_Nano_VL_V2,
                NemotronH_Nano_Omni_Reasoning_V3_Config: NemotronH_Nano_Omni_Reasoning_V3,
            },
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

        # Dynamic resolution config
        self.dynamic_resolution = getattr(hf_config, "dynamic_resolution", False)
        self.min_num_patches = getattr(hf_config, "min_num_patches", 0)
        self.max_num_patches = getattr(hf_config, "max_num_patches", 0)
        self.patch_size = hf_config.patch_size
        self.downsample_ratio = hf_config.downsample_ratio

        # Video temporal compression config
        self.video_temporal_patch_size = getattr(
            hf_config, "video_temporal_patch_size", 1
        )
        self.video_target_num_patches = getattr(
            hf_config, "video_target_num_patches", 0
        )
        self.video_maintain_aspect_ratio = getattr(
            hf_config, "video_maintain_aspect_ratio", True
        )

        self.max_model_len = getattr(server_args, "context_length", None) or 8192

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

    def render_image_dynamic(self, *, num_tokens: int):
        return f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_tokens}{self.IMG_END_TOKEN}"

    def render_tubelet(
        self,
        tubelet_index: int,
        frame_indices: list[int],
        timestamps: list[float],
        num_tokens: int,
    ):
        """Render a tubelet (group of T frames) for temporal compression."""
        if len(frame_indices) == 1:
            return self.render_frame(
                frame_indices[0], timestamp=timestamps[0], num_tokens=num_tokens
            )
        parts = " and ".join(
            f"frame {fi + 1} sampled at {ts:.2f} seconds"
            for fi, ts in zip(frame_indices, timestamps)
        )
        return f"{parts}: {self.PLACEHOLDER}{self.IMG_CONTEXT_TOKEN * num_tokens}{self.IMG_END_TOKEN}"

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

        T = self.video_temporal_patch_size

        if T > 1:
            tubelets_per_video = [math.ceil(len(frames) / T) for frames, _ in videos]
            if self.video_target_num_patches > 0 and videos:
                frame_h, frame_w = videos[0][0][0].shape[:2]
                target_w, target_h, tokens_per_tubelet = (
                    get_video_target_size_and_feature_size(
                        frame_w,
                        frame_h,
                        self.video_target_num_patches,
                        self.video_maintain_aspect_ratio,
                        self.patch_size,
                        self.downsample_ratio,
                    )
                )
                ds = int(1 / self.downsample_ratio)
                rows = target_h // self.patch_size // ds
                cols = target_w // self.patch_size // ds
            else:
                tokens_per_tubelet = self.num_image_token
                rows = cols = int(sqrt(tokens_per_tubelet))
            create_data_items, tokens_per_frame = self.evs.static_size_data_items(
                frames_per_video=tubelets_per_video,
                num_images=len(base_output.images),
                rows=rows,
                cols=cols,
            )
        else:
            rows = cols = int(sqrt(self.num_image_token))
            create_data_items, tokens_per_frame = self.evs.static_size_data_items(
                frames_per_video=[len(frames) for frames, _ in videos],
                num_images=len(base_output.images),
                rows=rows,
                cols=cols,
            )

        prompt = input_text
        image_is_dynamic = False
        num_tokens_per_image = []
        image_feature = None
        if base_output.images and self.dynamic_resolution:
            image_is_dynamic = True
            image_sizes = [(img.width, img.height) for img in base_output.images]
            text_only = input_text.replace(self.IMG_CONTEXT_TOKEN, "")
            text_tokens = len(
                self.tokenizer(text_only, add_special_tokens=False)["input_ids"]
            )
            total_token_budget = self.max_model_len - text_tokens
            budgeted_sizes = compute_budgeted_image_sizes(
                image_sizes,
                total_token_budget,
                self.patch_size,
                self.downsample_ratio,
                self.min_num_patches,
                self.max_num_patches,
            )
            preprocessed_images = []
            for image, (target_w, target_h, n_tokens) in zip(
                base_output.images, budgeted_sizes
            ):
                pv = resize_image_to_pixels(
                    image,
                    target_w,
                    target_h,
                    mean=self.norm_mean,
                    std=self.norm_std,
                )
                preprocessed_images.append(pv.to(dtype=torch.bfloat16))
                num_tokens_per_image.append(n_tokens)
            rendered_images = [
                self.render_image_dynamic(num_tokens=nt) for nt in num_tokens_per_image
            ]
            prompt = prompt.replace(self.IMG_CONTEXT_TOKEN, "".join(rendered_images), 1)
            image_feature = preprocessed_images
        elif base_output.images:
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
        T = self.video_temporal_patch_size
        if base_output.videos:
            preprocessed_videos = []
            for (video_array, timestamps), tpf in zip(
                videos, tokens_per_frame, strict=True
            ):
                if self.video_target_num_patches > 0:
                    frames_tensors = []
                    for frame in video_array:
                        pv, _ = video_to_pixel_values(
                            Image.fromarray(frame, mode="RGB"),
                            patch_size=self.patch_size,
                            downsample_ratio=self.downsample_ratio,
                            target_num_patches=self.video_target_num_patches,
                            maintain_aspect_ratio=self.video_maintain_aspect_ratio,
                            mean=self.norm_mean,
                            std=self.norm_std,
                        )
                        frames_tensors.append(pv.to(dtype=torch.bfloat16))
                else:
                    frames_tensors = [
                        self.preprocess_image(
                            Image.fromarray(frame, mode="RGB"),
                            max_num_tiles=NUM_VIDEO_TILES,
                        )
                        for frame in video_array
                    ]
                preprocessed_video = torch.cat(frames_tensors, dim=0)
                preprocessed_videos.append(preprocessed_video)

                if T > 1:
                    num_frames = len(video_array)
                    num_tubelets = math.ceil(num_frames / T)
                    rendered_parts = []
                    for ti in range(num_tubelets):
                        start_fi = ti * T
                        end_fi = min(start_fi + T, num_frames)
                        fi_list = list(range(start_fi, end_fi))
                        ts_list = [timestamps[fi] for fi in fi_list]
                        rendered_parts.append(
                            self.render_tubelet(
                                ti, fi_list, ts_list, num_tokens=tpf[ti]
                            )
                        )
                    prompt = prompt.replace(
                        self.VIDEO_CONTEXT_TOKEN, "\n".join(rendered_parts), 1
                    )
                else:
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

        # Extract audio from video if requested and no explicit audio provided
        use_audio_in_video = getattr(request_obj, "use_audio_in_video", False)
        extracted_audios: list[np.ndarray] = []
        if (
            use_audio_in_video
            and base_output.videos
            and not base_output.audios
            and self.audio_extractor is not None
        ):
            for video_wrapper in base_output.videos:
                video_bytes = video_wrapper.source_bytes
                if video_bytes is not None:
                    audio_array = extract_audio_from_video_bytes(
                        video_bytes,
                        target_sr=self.audio_extractor.sampling_rate,
                    )
                    if audio_array is not None:
                        extracted_audios.append(audio_array)

        all_audios: list[np.ndarray] = (
            list(base_output.audios) if base_output.audios else []
        )
        all_audios.extend(extracted_audios)

        # Process audio data through the Parakeet feature extractor
        audio_items: list[MultimodalDataItem] = []
        if all_audios and self.audio_extractor is not None:
            extractor = self.audio_extractor
            for audio in all_audios:
                num_tokens = extractor.audio_token_count(len(audio))
                rendered = self.render_audio(num_tokens=num_tokens)
                if self.AUDIO_CONTEXT_TOKEN in prompt:
                    prompt = prompt.replace(self.AUDIO_CONTEXT_TOKEN, rendered, 1)
                else:
                    prompt = prompt + rendered

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

        if image_is_dynamic and image_feature is not None:
            items = []
            for i, (pv, offset) in enumerate(zip(image_feature, img_offsets)):
                items.append(
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        feature=pv,
                        offsets=[offset],
                        model_specific_data={
                            "num_tokens": num_tokens_per_image[i],
                            "is_dynamic": True,
                        },
                    )
                )
            if video_feature is not None:
                items.append(
                    MultimodalDataItem(
                        modality=Modality.VIDEO,
                        feature=video_feature,
                        offsets=video_offsets,
                    )
                )
        else:
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
