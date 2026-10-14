# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""HF processor classes live in sglang.srt.configs.minimax_vl_processor to avoid circular imports with model classes."""

import math
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.multimodal.cache import resolve_multimodal_item_hash
from sglang.srt.multimodal.media_artifacts import MediaArtifactCacheMixin
from sglang.srt.multimodal.media_artifacts.minimax_m3 import MiniMaxM3ImageArtifact
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import round_up


def get_hw_multiple_of(
    image_size: Tuple[int, int],
    multiple: int,
    max_size: Union[None, int, Tuple[int, int]] = None,
) -> Tuple[int, int]:
    w, h = image_size

    if isinstance(max_size, int):
        ratio = 1.0
        max_dim = max(w, h)
        if max_dim > max_size:
            ratio = max_size / max_dim
        new_w = round_up(round(w * ratio), multiple)
        new_h = round_up(round(h * ratio), multiple)
        return new_w, new_h

    new_w = round_up(w, multiple)
    new_h = round_up(h, multiple)

    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0

        if new_w > max_w or new_h > max_h:
            new_w_ = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h_ = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)
            new_w = new_w_
            new_h = new_h_

            new_w = (
                new_w
                if new_w % multiple == 0
                else new_w + (multiple - new_w % multiple)
            )
            new_h = (
                new_h
                if new_h % multiple == 0
                else new_h + (multiple - new_h % multiple)
            )

        assert new_w % multiple == 0 and new_h % multiple == 0
        assert new_w <= max_w and new_h <= max_h

    return new_w, new_h


def _compute_sampled_frame_indices(
    total_frames: int,
    video_fps: float,
    fps: float,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Frame indices must match SFT extract_frame.py constant-mode sampling (>=1/fps apart, always keep last) or eval diverges from training."""
    if total_frames <= 0 or video_fps <= 0 or fps <= 0:
        return [0] if total_frames > 0 else []

    read_time_interval = 1.0 / fps
    eps = 1e-4

    indices: List[int] = []
    prev_kept_ts = -float("inf")
    while True:
        if not indices:
            target_frame = 0
        else:
            target_ts = prev_kept_ts + read_time_interval - eps
            target_frame = math.ceil(target_ts * video_fps)
            target_frame = max(target_frame, indices[-1] + 1)
        if target_frame >= total_frames:
            break
        indices.append(target_frame)
        prev_kept_ts = target_frame / video_fps

    last_frame_idx = total_frames - 1
    last_ts = last_frame_idx / video_fps
    if indices and indices[-1] != last_frame_idx and last_ts - prev_kept_ts > eps:
        indices.append(last_frame_idx)

    if not indices:
        indices = [0]
    if max_frames is not None and len(indices) > max_frames > 0:
        last = indices[-1]
        if max_frames == 1:
            # max_frames == 1 would divide by (max_frames - 1) == 0 below; keep only the last frame.
            indices = [last]
        else:
            step = len(indices) / (max_frames - 1)
            indices = [indices[int(i * step)] for i in range(max_frames - 1)]
            indices.append(last)
    return indices


async def get_video_tensor(
    vr,
    image_factor: int,
    max_size: Tuple[int, int],
    fps: Optional[float] = None,
    frame_max_size: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    if fps is None:
        fps = 1.0
    if frame_max_size is None:
        frame_max_size = max_size[0]
    if fps <= 0:
        raise ValueError(f"video fps must be > 0, got {fps}")

    if isinstance(vr, torch.Tensor):
        video_tchw = vr
        _, _, height, width = video_tchw.shape
        resized_width, resized_height = get_hw_multiple_of(
            (width, height), image_factor, max_size
        )
        resized = torchvision.transforms.functional.resize(
            video_tchw,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
        )
        return resized, {
            "total_num_frames": resized.shape[0],
            "fps": None,
            "frames_indices": None,
        }

    total_frames = len(vr)
    video_fps = vr.avg_fps
    if video_fps <= 0 or total_frames <= 0:
        raise ValueError(
            f"Invalid video metadata: fps={video_fps}, frames={total_frames}"
        )
    indices = _compute_sampled_frame_indices(total_frames, video_fps, fps, max_frames)
    video_tchw = vr.get_frames_as_tensor(indices)
    video_tchw = video_tchw.permute(0, 3, 1, 2).float()

    _, _, height, width = video_tchw.shape
    resized_width, resized_height = get_hw_multiple_of(
        (width, height), image_factor, frame_max_size
    )
    resized = torchvision.transforms.functional.resize(
        video_tchw,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
    )
    return resized, {
        "total_num_frames": total_frames,
        "fps": video_fps,
        "frames_indices": indices,
    }


class MiniMaxM3VLProcessor(MediaArtifactCacheMixin, BaseMultimodalProcessor):
    artifact_modality = Modality.IMAGE

    models = [
        MiniMaxM3SparseForConditionalGeneration,
    ]

    gpu_image_decode = False

    # M3's tokenizer has no pad_token.
    tokenizer_padding = False

    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    IMAGE_START_TOKEN = "]<]start of image[>["
    IMAGE_END_TOKEN = "]<]end of image[>["

    @staticmethod
    def _token_id(tokenizer, token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None, f"token id for {token!r} not found"
        return token_id

    @property
    def spatial_merge_size(self):
        return self._processor.image_processor.merge_size

    def _video_resize_config(self):
        video_processor = self._processor.video_processor
        image_factor = video_processor.patch_size * video_processor.merge_size
        # Newer M3 video processors expose max_pixels (area) instead of max_size; derive an equivalent (max_w, max_h) cap.
        max_size = getattr(video_processor, "max_size", None)
        if max_size is None:
            max_pixels = getattr(video_processor, "max_pixels", None)
            if max_pixels is not None:
                side = int(math.isqrt(int(max_pixels)))
                side -= side % image_factor
                max_size = (side, side)
            else:
                max_size = video_processor._max_size_from_size(video_processor.size)
        assert max_size is not None, "video processor max_size is required"
        return image_factor, max_size

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        tokenizer = _processor.tokenizer
        assert tokenizer is not None, "tokenizer is required"

        self.IM_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_TOKEN)
        self.VIDEO_TOKEN_ID = self._token_id(tokenizer, self.VIDEO_TOKEN)
        self.IM_START_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_START_TOKEN)
        self.IM_END_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_END_TOKEN)
        self.video_fps = self.video_config.pop("fps", None)
        self.video_frame_max_size = self.video_config.pop("frame_max_size", None)
        self.video_max_frames = self.video_config.pop("max_frames", None)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=re.compile(
                r"<image>|<\|image\|>|<\|image_pad\|>|\]\<\]image\[\>\["
            ),
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.VIDEO_TOKEN_ID,
            video_token_regex=re.compile(r"<video>|<\|video\|>|\]\<\]video\[\>\["),
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["minimax_image_processor"] = self._processor.image_processor.to_dict()
        return payload

    def decode_media_snapshot(self, snapshot, modality):
        image = super().decode_media_snapshot(snapshot, modality)
        # Match BaseMultimodalProcessor._load_single_item even when callers
        # disable the HF processor's own RGB conversion.
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def prepare_artifact_batch(self, entries, *, processor=None):
        if not entries:
            return []
        processor, _ = self._resolve_processor(processor)
        result = processor.image_processor(
            images=[entry.media for entry in entries],
            **{**self.image_config, "return_tensors": "pt"},
        )
        grids = result["image_grid_thw"]
        features = result["pixel_values"]
        if features.device.type != "cpu":
            raise ValueError("MiniMax M3 image artifacts require CPU preprocessing")
        if len(grids) != len(entries):
            raise ValueError("Expected one MiniMax M3 grid per image")
        lengths = [int(grid.prod()) for grid in grids]
        if sum(lengths) != features.shape[0]:
            raise ValueError("MiniMax M3 pixel values do not match image grids")
        artifacts = []
        for entry, grid, feature in zip(entries, grids, features.split(lengths)):
            # Own each image's storage so eviction releases its memory independently.
            # A lone image usually owns the whole batch storage already.
            if (
                feature.untyped_storage().nbytes()
                != feature.numel() * feature.element_size()
            ):
                feature = feature.clone()
            artifacts.append(
                MiniMaxM3ImageArtifact(
                    content_digest=entry.content_digest,
                    artifact_key=entry.artifact_key,
                    feature_hash=resolve_multimodal_item_hash(
                        feature=feature, namespace=entry.artifact_key
                    ),
                    grid_thw=tuple(int(value) for value in grid),
                    feature=feature,
                )
            )
        return artifacts

    def compose_image_artifacts(self, input_text, artifacts, *, processor=None):
        processor, tokenizer = self._resolve_processor(processor)
        # Mirror HF MiniMaxM3VLProcessor.replace_image_token from cached grids,
        # so cache hits never call the image processor.
        merge_length = processor.image_processor.merge_size**2
        parts = input_text.split(self.IMAGE_TOKEN)
        if len(parts) != len(artifacts) + 1:
            raise ValueError("Expected one MiniMax M3 image placeholder per artifact")
        expanded = [parts[0]]
        for artifact, part in zip(artifacts, parts[1:]):
            num_tokens = math.prod(artifact.grid_thw) // merge_length
            expanded.append(
                self.IMAGE_START_TOKEN
                + self.IMAGE_TOKEN * num_tokens
                + self.IMAGE_END_TOKEN
                + part
            )
        input_text = "".join(expanded)
        # Same tokenizer call and BOS handling as process_mm_data.
        tokenizer_kwargs = {}
        if self._tokenizer_auto_adds_specials:
            bos = getattr(tokenizer, "bos_token", None)
            if bos and input_text.startswith(bos):
                tokenizer_kwargs["add_special_tokens"] = False
        ret = tokenizer(
            [input_text], padding=True, return_tensors="pt", **tokenizer_kwargs
        )
        input_ids = ret["input_ids"].flatten()
        offsets = self.get_mm_items_offset(input_ids, self.IM_TOKEN_ID)
        if len(offsets) != len(artifacts):
            raise ValueError("Expected one MiniMax M3 image span per artifact")
        items = []
        for artifact, offset in zip(artifacts, offsets):
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=artifact.feature,
                offsets=[offset],
                model_specific_data={
                    "image_grid_thw": torch.tensor(
                        [artifact.grid_thw], dtype=torch.long
                    )
                },
            )
            item.set_hash(artifact.feature_hash)
            items.append(item)
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=self._finalize_mm_items(items, images=None),
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )

    async def process_mm_data_async(
        self,
        image_data: Optional[List],
        audio_data: Optional[List],
        input_text: str,
        request_obj,
        **kwargs,
    ) -> Dict:
        # Opt-in through the existing preprocessing-cache budget. Keep mixed
        # modalities, processor-output inputs and tokenized prompts on their
        # established path; their reconstruction contracts differ.
        if (
            image_data
            and isinstance(input_text, str)
            and not audio_data
            and not request_obj.video_data
            and self.mm_preprocess_cache.enabled
            and not self.gpu_image_decode
            and not any(
                isinstance(item, torch.Tensor) and item.device.type != "cpu"
                for item in image_data
            )
            and self.image_config.get("device", "cpu") == "cpu"
            # The legacy path leaves alias placeholders such as <image> as
            # text, so only canonical prompts can take this path unchanged.
            and self.mm_tokens.image_token_regex.findall(input_text)
            == [self.IMAGE_TOKEN] * len(image_data)
            and not any(self._is_preprocessed_input(item) for item in image_data)
        ):
            artifacts = await self.prepare_media_artifacts(
                image_data,
                content_hashes=request_obj.mm_content_hashes,
            )
            if self.mm_processor_executor is None:
                return self.compose_image_artifacts(input_text, artifacts)
            return await self.mm_processor_executor.run(
                self.compose_image_artifacts, input_text, artifacts
            )

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        video_metadata = None
        if base_output.videos:
            image_factor, max_size = self._video_resize_config()
            videos_processed = [
                await get_video_tensor(
                    video,
                    image_factor=image_factor,
                    max_size=max_size,
                    fps=self.video_fps,
                    frame_max_size=self.video_frame_max_size,
                    max_frames=self.video_max_frames,
                )
                for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output=base_output,
            mm_tokens=self.mm_tokens,
            video_metadata=video_metadata,
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )
