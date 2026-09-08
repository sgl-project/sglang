# Copyright 2023-2025 SGLang Team
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
# ==============================================================================
"""Cosmos3-Edge multimodal processor.

The current supported Transformers release may not ship the Cosmos3-Edge
processor classes yet. This processor keeps serving unblocked by applying the
checkpoint's SigLIP2-style image/video patchification directly in SGLang.
"""

import math
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.cosmos3_edge import Cosmos3EdgeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

IMAGE_MIN_PIXELS = 256 * 256
IMAGE_MAX_PIXELS = 4096 * 4096
VIDEO_MIN_PIXELS = 64 * 64
VIDEO_TOTAL_PIXELS = 6144 * 4096
MAX_RATIO = 200
DEFAULT_SOURCE_VIDEO_FPS = 24.0
DEFAULT_TARGET_VIDEO_FPS = 2.0
DEFAULT_MIN_FRAMES = 4
DEFAULT_MAX_FRAMES = 768


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    num_frames: int = 1,
) -> tuple[int, int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            "absolute aspect ratio must be smaller than "
            f"{MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if num_frames * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif num_frames * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _as_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        image = image.numpy()

    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(
            f"Expected an image with 2 or 3 dimensions, got {image.shape}."
        )
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


class Cosmos3EdgeProcessor(SGLangBaseProcessor):
    models = [Cosmos3EdgeForConditionalGeneration]
    gpu_image_decode = False

    @staticmethod
    def _get_processor_output_value(output, key: str):
        if output is None:
            return None
        return output.get(key) if hasattr(output, "get") else getattr(output, key, None)

    @staticmethod
    def _as_grid_batch(value) -> Optional[torch.Tensor]:
        if value is None:
            return None
        grid = torch.as_tensor(value, dtype=torch.long)
        return grid.unsqueeze(0) if grid.ndim == 1 else grid

    @classmethod
    def _get_grid_from_output_or_items(
        cls,
        output,
        mm_items: list[MultimodalDataItem],
        key: str,
        modality: Modality,
    ) -> Optional[torch.Tensor]:
        grid = cls._as_grid_batch(cls._get_processor_output_value(output, key))
        if grid is not None:
            return grid

        grids = []
        for item in mm_items:
            if not item.is_modality(modality):
                continue
            item_grid = cls._as_grid_batch(item.model_specific_data.get(key))
            if item_grid is not None:
                grids.append(item_grid)
        return torch.cat(grids, dim=0) if grids else None

    def _get_precomputed_mrope(self, output):
        positions = self._get_processor_output_value(output, "mrope_positions")
        delta = self._get_processor_output_value(output, "mrope_position_delta")
        if positions is None or delta is None:
            return None

        positions = torch.as_tensor(positions)
        if positions.ndim == 3:
            if positions.shape[1] != 1:
                return None
            positions = positions.squeeze(1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            return None

        delta = torch.as_tensor(delta)
        if delta.ndim <= 1:
            delta = delta.reshape(-1, 1)
        return positions, delta

    def _make_processor_output(
        self,
        input_ids: Union[list[int], torch.Tensor],
        mm_items: list[MultimodalDataItem],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        processor_output=None,
    ) -> MultimodalProcessorOutput:
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).flatten()
        mrope_result = self._get_precomputed_mrope(processor_output)
        if mrope_result is None:
            has_images = any(item.is_image() for item in mm_items)
            has_videos = any(item.is_video() for item in mm_items)
            if has_images and image_grid_thw is None:
                raise ValueError(
                    "Cosmos3-Edge processed image input requires image_grid_thw "
                    "or precomputed MRoPE positions."
                )
            if has_videos and video_grid_thw is None:
                raise ValueError(
                    "Cosmos3-Edge processed video input requires video_grid_thw "
                    "or precomputed MRoPE positions."
                )
            mrope_result = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self.spatial_merge_size,
                image_token_id=self.mm_tokens.image_token_id,
                video_token_id=self.mm_tokens.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        mrope_positions, mrope_position_delta = mrope_result
        if mrope_positions.ndim == 3:
            mrope_positions = mrope_positions.squeeze(1)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IMAGE_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    async def _process_preprocessed_mm_data(self, base_output):
        (
            mm_items,
            input_ids,
            processor_output,
        ) = await self.process_and_combine_mm_data_async(base_output, self.mm_tokens)
        image_grid_thw = self._get_grid_from_output_or_items(
            processor_output,
            mm_items,
            "image_grid_thw",
            Modality.IMAGE,
        )
        video_grid_thw = self._get_grid_from_output_or_items(
            processor_output,
            mm_items,
            "video_grid_thw",
            Modality.VIDEO,
        )
        return self._make_processor_output(
            input_ids=input_ids,
            mm_items=mm_items,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            processor_output=processor_output,
        )

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_TOKEN_ID = hf_config.image_token_id
        self.IMAGE_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.model_type = hf_config.model_type

        self.patch_size = hf_config.vision_config.patch_size
        self._spatial_merge_size = hf_config.projector_config.spatial_merge_size
        self.temporal_patch_size = 1

        image_token = self._tokenizer.convert_ids_to_tokens([self.IMAGE_TOKEN_ID])[0]
        video_token = self._tokenizer.convert_ids_to_tokens([self.VIDEO_TOKEN_ID])[0]
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            video_token=video_token,
            image_token_id=self.IMAGE_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        ).build(self._processor)

        self.ATTR_NAME_TO_MODALITY["pixel_attention_mask"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["spatial_shapes"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["pixel_attention_mask_videos"] = Modality.VIDEO
        self.ATTR_NAME_TO_MODALITY["spatial_shapes_videos"] = Modality.VIDEO

    @property
    def spatial_merge_size(self):
        return self._spatial_merge_size

    def _tokenize_prompt(self, prompt: Union[str, list[int]]) -> list[int]:
        if isinstance(prompt, list):
            return list(prompt)
        add_special_tokens = True
        bos = getattr(self._tokenizer, "bos_token", None)
        if self._tokenizer_auto_adds_specials and bos and prompt.startswith(bos):
            add_special_tokens = False
        return self._tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    def _size_limits(
        self, config: dict, default_min_pixels: int, default_max_pixels: int
    ) -> tuple[int, int]:
        size_value = config.get("size", {})
        size = size_value if isinstance(size_value, dict) else {}
        min_pixels = config.get(
            "min_pixels",
            config.get("shortest_edge", size.get("shortest_edge", default_min_pixels)),
        )
        max_pixels = config.get(
            "max_pixels",
            config.get("longest_edge", size.get("longest_edge", default_max_pixels)),
        )
        return int(min_pixels), int(max_pixels)

    def _preprocess_pil_image(
        self,
        image: Image.Image,
        *,
        min_pixels: int,
        max_pixels: int,
        resized_size: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        factor = self.patch_size * self.spatial_merge_size
        if resized_size is None:
            resized_height, resized_width = _smart_resize(
                image.height,
                image.width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            resized_height, resized_width = resized_size

        if image.size != (resized_width, resized_height):
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.BICUBIC
            )

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5

        patch_size = self.patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        patches = array.reshape(grid_h, patch_size, grid_w, patch_size, 3)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(grid_h * grid_w, -1)
        return torch.from_numpy(patches), (grid_h, grid_w)

    def _preprocess_image_item(self, image: Any) -> MultimodalDataItem:
        min_pixels, max_pixels = self._size_limits(
            self.image_config, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS
        )
        patches, (grid_h, grid_w) = self._preprocess_pil_image(
            _as_pil_image(image), min_pixels=min_pixels, max_pixels=max_pixels
        )
        spatial_shapes = torch.tensor([[grid_h, grid_w]], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
        return MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=patches,
            model_specific_data={
                "spatial_shapes": spatial_shapes,
                "image_grid_thw": image_grid_thw,
            },
        )

    def _select_frame_indices(self, total_frames: int, video_fps: float) -> list[int]:
        if total_frames <= 0:
            raise ValueError("Video must contain at least one frame.")

        has_num_frames = "num_frames" in self.video_config
        has_nframes = "nframes" in self.video_config
        if has_num_frames and has_nframes:
            raise ValueError("Specify only one of num_frames and nframes")

        explicit_num_frames = self.video_config.get(
            "num_frames", self.video_config.get("nframes")
        )
        if explicit_num_frames is not None:
            if "fps" in self.video_config:
                raise ValueError("Specify only one of num_frames/nframes and fps")
            nframes = int(explicit_num_frames)
        else:
            fps = float(self.video_config.get("fps", DEFAULT_TARGET_VIDEO_FPS))
            source_fps = video_fps if video_fps > 0 else DEFAULT_SOURCE_VIDEO_FPS
            nframes = int(total_frames / source_fps * fps)

        min_frames = int(self.video_config.get("min_frames", DEFAULT_MIN_FRAMES))
        max_frames = int(self.video_config.get("max_frames", DEFAULT_MAX_FRAMES))
        if min_frames <= 0 or max_frames < min_frames:
            raise ValueError(
                "Video frame limits must satisfy 0 < min_frames <= max_frames"
            )
        nframes = max(min_frames, min(max_frames, nframes))

        nframes = max(1, min(total_frames, nframes))
        if nframes == total_frames:
            return list(range(total_frames))
        return (
            np.linspace(0, total_frames - 1, num=nframes)
            .round()
            .astype(np.int64)
            .tolist()
        )

    def _timestamps_from_indices(
        self, frame_indices: list[int], video_fps: float
    ) -> list[float]:
        if video_fps > 0:
            return [float(idx) / video_fps for idx in frame_indices]
        return [float(idx) for idx in range(len(frame_indices))]

    def _default_sampled_video_fps(self) -> float:
        fps = float(self.video_config.get("fps", DEFAULT_TARGET_VIDEO_FPS))
        return fps if fps > 0 else 0.0

    def _coerce_video_frames(self, video: Any) -> list[Image.Image]:
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu()
            if video.ndim == 3:
                video = video.unsqueeze(0)
            if video.ndim != 4:
                raise ValueError(
                    f"Expected video tensor with 4 dimensions, got {video.shape}."
                )
            if video.shape[-1] in (1, 3, 4):
                frames = [video[i] for i in range(video.shape[0])]
            elif video.shape[1] in (1, 3, 4):
                frames = [video[i].permute(1, 2, 0) for i in range(video.shape[0])]
            else:
                raise ValueError(
                    f"Cannot infer video channel dimension from {video.shape}."
                )
            return [_as_pil_image(frame) for frame in frames]

        if isinstance(video, np.ndarray):
            if video.ndim == 3:
                video = video[None, ...]
            if video.ndim != 4:
                raise ValueError(
                    f"Expected video array with 4 dimensions, got {video.shape}."
                )
            return [_as_pil_image(frame) for frame in video]

        if isinstance(video, (list, tuple)):
            return [_as_pil_image(frame) for frame in video]

        raise ValueError(f"Unsupported video input type: {type(video)!r}.")

    def _video_to_frames_and_timestamps(
        self, video: Any
    ) -> tuple[list[Image.Image], list[float]]:
        metadata = None
        if isinstance(video, tuple) and len(video) == 2 and isinstance(video[1], dict):
            video, metadata = video

        if isinstance(video, VideoDecoderWrapper):
            fps = float(video.avg_fps or DEFAULT_SOURCE_VIDEO_FPS)
            indices = self._select_frame_indices(len(video), fps)
            frames = video.get_frames_as_tensor(indices)
            frame_images = [_as_pil_image(frame) for frame in frames]
            timestamps = self._timestamps_from_indices(indices, fps)
            return frame_images, timestamps

        frames = self._coerce_video_frames(video)
        fps = self._default_sampled_video_fps()
        frame_indices = list(range(len(frames)))
        if metadata is not None:
            fps = float(metadata.get("fps", fps) or 0.0)
            metadata_indices = metadata.get("frames_indices")
            if metadata_indices is not None:
                metadata_indices = np.asarray(metadata_indices).reshape(-1).tolist()
                if len(metadata_indices) == len(frames):
                    frame_indices = [int(idx) for idx in metadata_indices]
        return frames, self._timestamps_from_indices(frame_indices, fps)

    def _preprocess_video_item(self, video: Any) -> MultimodalDataItem:
        frames, timestamps = self._video_to_frames_and_timestamps(video)
        total_min_pixels, total_max_pixels = self._size_limits(
            self.video_config, VIDEO_MIN_PIXELS, VIDEO_TOTAL_PIXELS
        )
        first_frame = frames[0]
        factor = self.patch_size * self.spatial_merge_size
        resized_size = _smart_resize(
            first_frame.height,
            first_frame.width,
            factor=factor,
            min_pixels=total_min_pixels,
            max_pixels=total_max_pixels,
            num_frames=len(frames),
        )

        frame_patches = []
        spatial_shapes = []
        for frame in frames:
            patches, (grid_h, grid_w) = self._preprocess_pil_image(
                frame,
                min_pixels=total_min_pixels,
                max_pixels=total_max_pixels,
                resized_size=resized_size,
            )
            frame_patches.append(patches)
            spatial_shapes.append([grid_h, grid_w])

        grid_h, grid_w = spatial_shapes[0]
        feature = torch.cat(frame_patches, dim=0)
        spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.long)
        video_grid_thw = torch.tensor([[len(frames), grid_h, grid_w]], dtype=torch.long)
        return MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=feature,
            model_specific_data={
                "spatial_shapes": spatial_shapes_tensor,
                "video_grid_thw": video_grid_thw,
                "timestamps": timestamps,
            },
        )

    def _timestamp_token_ids(self, timestamp: float) -> list[int]:
        return self._tokenizer.encode(
            f"<{timestamp:.1f} seconds>", add_special_tokens=False
        )

    def _build_input_ids(
        self,
        prompt: Union[str, list[int]],
        img_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        video_timestamps: Optional[list[list[float]]],
    ):
        if not isinstance(prompt, list):
            prompt = self._tokenize_prompt(prompt)

        input_ids = []
        offsets = []
        modality_list = []
        cur_idx = 0
        spatial_merge_size = self.spatial_merge_size

        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if prompt[i + 1] == self.IMAGE_TOKEN_ID:
                vision_start_indices.append((i, Modality.IMAGE))
            elif prompt[i + 1] == self.VIDEO_TOKEN_ID:
                vision_start_indices.append((i, Modality.VIDEO))

        img_idx = 0
        video_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            if modality == Modality.IMAGE:
                if img_grid_thw is None:
                    raise ValueError(
                        "Missing image grid metadata for image placeholder."
                    )
                mm_token_num = int(
                    img_grid_thw[img_idx].prod().item() // (spatial_merge_size**2)
                )
                assert cur_idx <= mm_start_idx
                input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
                mm_offset_start = len(input_ids)
                input_ids.extend([self.IMAGE_TOKEN_ID] * mm_token_num)
                offsets.append((mm_offset_start, len(input_ids) - 1))
                modality_list.append(Modality.IMAGE)
                cur_idx = mm_start_idx + 2
                img_idx += 1
                continue

            if video_grid_thw is None:
                raise ValueError("Missing video grid metadata for video placeholder.")
            num_frames = int(video_grid_thw[video_idx][0].item())
            tokens_per_frame = int(
                video_grid_thw[video_idx][1:].prod().item() // (spatial_merge_size**2)
            )
            timestamps = (
                video_timestamps[video_idx]
                if video_timestamps is not None
                else [float(i) for i in range(num_frames)]
            )
            if len(timestamps) != num_frames:
                raise ValueError(
                    "Cosmos3-Edge video timestamps must match video frame count: "
                    f"got {len(timestamps)} vs {num_frames}."
                )

            has_start = prompt[mm_start_idx] == self.IM_START_TOKEN_ID
            has_end = (
                mm_start_idx + 2 < len(prompt)
                and prompt[mm_start_idx + 2] == self.IM_END_TOKEN_ID
            )
            target_start = mm_start_idx if has_start else mm_start_idx + 1
            target_end = mm_start_idx + 2 if has_start and has_end else mm_start_idx + 1
            assert cur_idx <= target_start

            input_ids.extend(prompt[cur_idx:target_start])
            frame_offsets = []
            for timestamp in timestamps:
                input_ids.extend(self._timestamp_token_ids(float(timestamp)))
                input_ids.append(self.IM_START_TOKEN_ID)
                mm_offset_start = len(input_ids)
                input_ids.extend([self.VIDEO_TOKEN_ID] * tokens_per_frame)
                frame_offsets.append((mm_offset_start, len(input_ids) - 1))
                input_ids.append(self.IM_END_TOKEN_ID)

            offsets.append(frame_offsets)
            modality_list.append(Modality.VIDEO)
            cur_idx = target_end + 1
            video_idx += 1
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def _assign_offsets(
        self,
        modality_list: list[Modality],
        offsets: list,
        image_items: list[MultimodalDataItem],
        video_items: list[MultimodalDataItem],
    ) -> list[MultimodalDataItem]:
        image_idx = 0
        video_idx = 0
        mm_items = []
        for modality, offset in zip(modality_list, offsets):
            if modality == Modality.IMAGE:
                item = image_items[image_idx]
                image_idx += 1
            elif modality == Modality.VIDEO:
                item = video_items[video_idx]
                video_idx += 1
            else:
                continue
            item.offsets = offset if isinstance(offset, list) else [offset]
            mm_items.append(item)

        if image_idx != len(image_items) or video_idx != len(video_items):
            raise ValueError(
                "Cosmos3-Edge prompt media placeholders do not match provided media."
            )
        return mm_items

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text: str,
        request_obj,
        **kwargs,
    ):
        video_data = getattr(request_obj, "video_data", None)
        if video_data is not None and not isinstance(video_data, list):
            video_data = [video_data]

        if not image_data and not video_data:
            input_ids = self._tokenize_prompt(input_text)
            return MultimodalProcessorOutput(
                input_ids=input_ids,
                mm_items=[],
                im_start_id=self.IM_START_TOKEN_ID,
                im_end_id=self.IM_END_TOKEN_ID,
                im_token_id=self.IMAGE_TOKEN_ID,
                video_token_id=self.VIDEO_TOKEN_ID,
            )

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )

        if self._all_mm_data_is_preprocessed(base_output.images, base_output.videos):
            return await self._process_preprocessed_mm_data(base_output)

        if any(
            self._is_preprocessed_input(item)
            for item in [*base_output.images, *base_output.videos]
        ):
            raise ValueError(
                "Cosmos3-Edge does not support mixing raw and preprocessed media "
                "in the same request."
            )

        image_items = [
            self._preprocess_image_item(image) for image in base_output.images
        ]
        video_items = [
            self._preprocess_video_item(video) for video in base_output.videos
        ]

        image_grid_thw = (
            torch.cat([item.image_grid_thw for item in image_items], dim=0)
            if image_items
            else None
        )
        video_grid_thw = (
            torch.cat([item.video_grid_thw for item in video_items], dim=0)
            if video_items
            else None
        )

        prompt_ids = base_output.input_ids
        if prompt_ids is None:
            prompt_ids = self._tokenize_prompt(base_output.input_text)
        video_timestamps = (
            [item.timestamps for item in video_items] if video_items else None
        )
        input_ids, offsets, modality_list = self._build_input_ids(
            prompt_ids,
            img_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            video_timestamps=video_timestamps,
        )
        mm_items = self._assign_offsets(
            modality_list, offsets, image_items=image_items, video_items=video_items
        )

        return self._make_processor_output(
            input_ids=input_ids,
            mm_items=mm_items,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
