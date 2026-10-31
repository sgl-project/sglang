# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""HF processor classes live in sglang.srt.configs.minimax_vl_processor to avoid circular imports with model classes."""

import itertools
import math
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import smart_resize
from sglang.srt.utils import ImageData, VideoData

# Per-request media limits; requests outside them are rejected with HTTP 400.
MAX_IMAGE_PIXELS = 12845056
MAX_VIDEO_PIXELS = 301056000
MIN_VIDEO_FPS = 0.2
MAX_VIDEO_FPS = 5.0
# Per the MiniMax-M3 size spec; an image without max_long_side_pixel uses the default.
MIN_SHORT_SIDE = 112
MIN_LONG_SIDE_PIXEL = 112
MAX_LONG_SIDE_PIXEL = 3584
MAX_ASPECT_RATIO = 200
DEFAULT_IMAGE_LONG_SIDE = 2016


def _compute_aligned_size(
    height: int, width: int, factor: int, max_long_side_pixel: int
) -> Tuple[int, int]:
    # Returns (height, width): shrink the long side to the cap (max_long_side_pixel
    # rounded down to factor), else enlarge a short side below MIN_SHORT_SIDE;
    # the cap wins when an extreme aspect ratio cannot satisfy both.
    ratio = max(height, width) / min(height, width)
    if ratio > MAX_ASPECT_RATIO:
        raise ValueError(
            f"media aspect ratio must be <= {MAX_ASPECT_RATIO}, got {ratio:.1f}"
        )
    cap = max_long_side_pixel - max_long_side_pixel % factor
    if max(height, width) > cap:
        beta = cap / max(height, width)
    elif min(height, width) < MIN_SHORT_SIDE:
        beta = min(MIN_SHORT_SIDE / min(height, width), cap / max(height, width))
    else:
        beta = 1.0
    return (
        max(factor, round(height * beta / factor) * factor),
        max(factor, round(width * beta / factor) * factor),
    )


def _check_max_long_side_pixel(value: Any, factor: int) -> Optional[int]:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not MIN_LONG_SIDE_PIXEL <= value <= MAX_LONG_SIDE_PIXEL
        or value % factor
    ):
        raise ValueError(
            f"max_long_side_pixel must be a multiple of {factor} in "
            f"[{MIN_LONG_SIDE_PIXEL}, {MAX_LONG_SIDE_PIXEL}], got {value!r}"
        )
    return value


def _check_video_fps(value: Any) -> Optional[float]:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not MIN_VIDEO_FPS <= value <= MAX_VIDEO_FPS
    ):
        raise ValueError(
            f"video fps must be in [{MIN_VIDEO_FPS}, {MAX_VIDEO_FPS}], got {value!r}"
        )
    return float(value)


def _max_long_side_pixel(item: Any) -> Any:
    if isinstance(item, (ImageData, VideoData)):
        return item.max_long_side_pixel
    if isinstance(item, dict):
        return item.get("max_long_side_pixel")
    return None


def _video_fps(item: Any) -> Any:
    # The chat template path carries per-video fps in preprocess_kwargs.
    if isinstance(item, VideoData):
        return (item.preprocess_kwargs or {}).get("fps")
    if isinstance(item, dict):
        fps = item.get("fps")
        return (
            fps if fps is not None else (item.get("preprocess_kwargs") or {}).get("fps")
        )
    return None


def _resize_image_to_long_side(
    image: Image.Image, factor: int, max_long_side_pixel: int
) -> Image.Image:
    width, height = image.size
    new_h, new_w = _compute_aligned_size(height, width, factor, max_long_side_pixel)
    if new_h * new_w > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"image pixels exceed {MAX_IMAGE_PIXELS}, got {new_h * new_w} "
            f"({new_w}x{new_h})"
        )
    if (new_w, new_h) == image.size:
        return image
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def _resize_image_to_max_pixels(
    image: Image.Image, factor: int, max_pixels: int
) -> Image.Image:
    pixels = tvF.pil_to_tensor(image)
    height, width = pixels.shape[-2:]
    new_height, new_width = smart_resize(
        height=height, width=width, factor=factor, max_pixels=max_pixels
    )
    if (new_height, new_width) == (height, width):
        return image
    # Matches the HF torchvision backend; transforms v1 resize gives other pixels.
    resized = tvF.resize(
        pixels.unsqueeze(0),
        [new_height, new_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return Image.fromarray(resized[0].permute(1, 2, 0).numpy())


def _check_video_pixels(num_frames: int, height: int, width: int) -> None:
    total = num_frames * height * width
    if total > MAX_VIDEO_PIXELS:
        raise ValueError(
            f"video pixels exceed {MAX_VIDEO_PIXELS}, got {total} "
            f"({num_frames} frames of {width}x{height})"
        )


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
    resize_fn=None,
    max_long_side_pixel: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    return sample_video_frames(
        vr,
        image_factor=image_factor,
        max_size=max_size,
        fps=fps,
        frame_max_size=frame_max_size,
        max_frames=max_frames,
        resize_fn=resize_fn,
        max_long_side_pixel=max_long_side_pixel,
    )


def sample_video_frames(
    vr,
    *,
    image_factor: int,
    max_size: Tuple[int, int],
    fps: Optional[float] = None,
    frame_max_size: Optional[int] = None,
    max_frames: Optional[int] = None,
    resize_fn=None,
    max_long_side_pixel: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    if fps is None:
        fps = 1.0
    if frame_max_size is None:
        frame_max_size = max_size[0]
    if fps <= 0:
        raise ValueError(f"video fps must be > 0, got {fps}")

    def resized_hw(height, width, default_long_side):
        return _compute_aligned_size(
            height, width, image_factor, max_long_side_pixel or default_long_side
        )

    def resize(video_tchw, size):
        if resize_fn is not None:
            return resize_fn(video_tchw, size)
        return torchvision.transforms.functional.resize(
            video_tchw, list(size), interpolation=InterpolationMode.BICUBIC
        )

    if isinstance(vr, torch.Tensor):
        video_tchw = vr
        num_frames, _, height, width = video_tchw.shape
        resized_size = resized_hw(height, width, max(max_size))
        _check_video_pixels(num_frames, *resized_size)
        resized = resize(video_tchw, resized_size)
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
    # Reject oversized videos before decoding the sampled frames.
    _check_video_pixels(len(indices), *resized_hw(*vr.frame_shape, frame_max_size))

    video_tchw = vr.get_frames_as_tensor(indices)
    video_tchw = video_tchw.permute(0, 3, 1, 2).float()
    # Container metadata may ignore rotation; size from the decoded frames.
    _, _, height, width = video_tchw.shape
    resized_size = resized_hw(height, width, frame_max_size)
    _check_video_pixels(len(indices), *resized_size)
    resized = resize(video_tchw, resized_size)
    return resized, {
        "total_num_frames": total_frames,
        "fps": video_fps,
        "frames_indices": indices,
    }


def video_resize_config(video_processor) -> Tuple[int, Tuple[int, int]]:
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


def resize_images_for_request(
    images: list, image_data: list, *, image_processor, max_pixels: Optional[int]
) -> list:
    # The HF image processor takes one resize setting per call, so every image
    # is resized here; one without the option uses DEFAULT_IMAGE_LONG_SIDE,
    # or the HF area cap when the server sets an image max_pixels.
    factor = image_processor.patch_size * image_processor.merge_size
    limits = [
        _check_max_long_side_pixel(_max_long_side_pixel(item), factor)
        for item in image_data or []
    ]
    if len(limits) != len(images):
        if any(limit is not None for limit in limits):
            raise ValueError(
                f"max_long_side_pixel needs one loaded image per image item, "
                f"got {len(images)} images for {len(limits)} items"
            )
        limits = [None] * len(images)
    resized = []
    for image, limit in zip(images, limits):
        if not isinstance(image, Image.Image):
            raise ValueError(
                f"max_long_side_pixel requires raw image input, got "
                f"{type(image).__name__}"
            )
        if limit is None and max_pixels is not None:
            resized.append(_resize_image_to_max_pixels(image, factor, max_pixels))
        else:
            resized.append(
                _resize_image_to_long_side(
                    image,
                    factor,
                    DEFAULT_IMAGE_LONG_SIDE if limit is None else limit,
                )
            )
    return resized


def expand_media_tokens(
    ids: List[int], replacements: Dict[int, List[List[int]]], *, im_start_id: int
) -> List[int]:
    output = []
    consumed = dict.fromkeys(replacements, 0)
    pos = 0
    while pos < len(ids):
        # Already expanded media must match its processed grid exactly.
        for token, chunks in replacements.items():
            index = consumed[token]
            if (
                index < len(chunks)
                and ids[pos : pos + len(chunks[index])] == chunks[index]
            ):
                chunk = chunks[index]
                output.extend(chunk)
                consumed[token] += 1
                pos += len(chunk)
                break
        else:
            token = ids[pos]
            if token == im_start_id:
                raise ValueError(
                    "Expanded media tokens do not match the processed media"
                )
            if token in replacements:
                index = consumed[token]
                if index >= len(replacements[token]):
                    raise ValueError("More media placeholders than supplied media")
                output.extend(replacements[token][index])
                consumed[token] += 1
                pos += 1
                continue
            output.append(token)
            pos += 1
    if any(consumed[token] != len(chunks) for token, chunks in replacements.items()):
        raise ValueError("Supplied media is missing a matching placeholder")
    return output


def video_frame_timestamps(
    metadata: dict, num_frames: int, temporal_patch_size: int
) -> Optional[List[float]]:
    fps = metadata.get("fps")
    indices = metadata.get("frames_indices")
    if fps is None or indices is None:
        return None
    return [
        indices[min(frame * temporal_patch_size, len(indices) - 1)] / fps
        for frame in range(num_frames)
    ]


def preprocess_for_encoder(
    items: list, modality: Modality, *, processor, mm_process_config: dict
) -> dict:
    if modality == Modality.IMAGE:
        return _preprocess_images_for_encoder(
            items, processor=processor, image_config=mm_process_config.get("image", {})
        )
    if modality == Modality.VIDEO:
        return _preprocess_videos_for_encoder(
            items, processor=processor, video_config=mm_process_config.get("video", {})
        )
    raise ValueError(f"MiniMax-M3 does not accept {modality.name.lower()} input")


def _preprocess_images_for_encoder(items: list, *, processor, image_config: dict):
    images = [item["image"] if isinstance(item, dict) else item for item in items]
    resized = resize_images_for_request(
        images,
        items,
        image_processor=processor.image_processor,
        max_pixels=image_config.get("max_pixels"),
    )
    return processor.image_processor(
        images=resized, return_tensors="pt", **{**image_config, "do_resize": False}
    )


def _preprocess_videos_for_encoder(items: list, *, processor, video_config: dict):
    video_config = dict(video_config)
    default_fps = video_config.pop("fps", None)
    frame_max_size = video_config.pop("frame_max_size", None)
    max_frames = video_config.pop("max_frames", None)
    video_processor = processor.video_processor
    image_factor, max_size = video_resize_config(video_processor)
    videos, metadata = [], []
    for item in items:
        fps = _check_video_fps(_video_fps(item))
        frames, frame_metadata = sample_video_frames(
            item["video"] if isinstance(item, dict) else item,
            image_factor=image_factor,
            max_size=max_size,
            fps=default_fps if fps is None else fps,
            frame_max_size=frame_max_size,
            max_frames=max_frames,
            resize_fn=getattr(video_processor, "resize_frames", None),
            max_long_side_pixel=_check_max_long_side_pixel(
                _max_long_side_pixel(item), image_factor
            ),
        )
        videos.append(frames)
        metadata.append(frame_metadata)
    result = video_processor(
        videos=videos, return_tensors="pt", **{"do_resize": False, **video_config}
    )
    result["video_timestamps"] = [
        video_frame_timestamps(
            frame_metadata, int(grid[0]), video_processor.temporal_patch_size
        )
        for frame_metadata, grid in zip(metadata, result["video_grid_thw"])
    ]
    return result


def _media_token_runs(input_ids: List[int], token_id: int) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for pos, token in enumerate(input_ids + [None]):
        if token == token_id and start is None:
            start = pos
        elif token != token_id and start is not None:
            runs.append((start, pos - 1))
            start = None
    return runs


class MiniMaxM3VLProcessor(BaseMultimodalProcessor):
    models = [MiniMaxM3SparseForConditionalGeneration]

    gpu_image_decode = False
    # The tokenizer process shares GPU 0 with the scheduler; long videos
    # preprocessed there run it out of memory.
    video_preprocessing_device = "cpu"
    retry_on_cpu_after_gpu_oom = True

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
        return video_resize_config(self._processor.video_processor)

    def _resize_images_for_request(self, images: list, image_data: list) -> list:
        return resize_images_for_request(
            images,
            image_data,
            image_processor=self._processor.image_processor,
            max_pixels=self.image_config.get("max_pixels"),
        )

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
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.VIDEO_TOKEN_ID,
        ).build(_processor)

    preserve_processor_input_ids = True

    @cached_property
    def feature_dtype(self) -> torch.dtype:
        return model_config_of(self.server_args).dtype

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        images: Optional[List[Any]],
    ) -> List[MultimodalDataItem]:
        for item in mm_items:
            if (
                isinstance(item.feature, torch.Tensor)
                and item.feature.is_floating_point()
            ):
                item.feature = item.feature.to(self.feature_dtype)
        return mm_items

    def process_mm_data(
        self,
        input_text,
        images=None,
        videos=None,
        audios=None,
        input_ids=None,
        video_metadata=None,
        images_resized=False,
        processor=None,
        **kwargs,
    ):
        processor, _ = self._resolve_processor(processor)
        image_config = (
            {**self.image_config, "do_resize": False}
            if images_resized
            else self.image_config
        )
        if input_ids is None:
            if images_resized:
                kwargs["images_kwargs"] = {"do_resize": False}
            return super().process_mm_data(
                input_text=input_text,
                images=images,
                videos=videos,
                audios=audios,
                processor=processor,
                video_metadata=video_metadata,
                **kwargs,
            )
        # Tokenized prompts must retain every text token. Decoding and encoding
        # the entire prompt can change BPE boundaries and special-token IDs.
        result = {}
        if images:
            result.update(
                processor.image_processor(
                    images=images, return_tensors="pt", **image_config
                )
            )
        if videos:
            result.update(
                processor.video_processor(
                    videos=videos,
                    return_tensors="pt",
                    **{"do_resize": False, **self.video_config},
                )
            )
            if video_metadata is not None:
                from types import SimpleNamespace

                result["video_metadata"] = [
                    SimpleNamespace(**m) if isinstance(m, dict) else m
                    for m in video_metadata
                ]
        ids = torch.as_tensor(input_ids, dtype=torch.long).flatten().tolist()
        image_grids = result.get("image_grid_thw")
        video_grids = result.get("video_grid_thw")
        video_timestamps = [
            (
                None
                if metadata is None
                else video_frame_timestamps(
                    vars(metadata),
                    int(grid[0]),
                    processor.video_processor.temporal_patch_size,
                )
            )
            for grid, metadata in zip(
                [] if video_grids is None else video_grids,
                result.get("video_metadata") or itertools.repeat(None),
            )
        ]
        # Checkpoint processors may define their own __call__ without the optional
        # replace_image_token/replace_video_token hooks, so build the text here.
        replacements = {
            self.IM_TOKEN_ID: [
                self._encode_media_text(self._image_placeholder_text(grid))
                for grid in ([] if image_grids is None else image_grids)
            ],
            self.VIDEO_TOKEN_ID: [
                self._encode_media_text(self._video_placeholder_text(grid, timestamps))
                for grid, timestamps in zip(
                    [] if video_grids is None else video_grids, video_timestamps
                )
            ],
        }
        output = expand_media_tokens(
            ids, replacements, im_start_id=self.IM_START_TOKEN_ID
        )
        result["input_ids"] = torch.tensor([output], dtype=torch.long)
        result["attention_mask"] = torch.ones_like(result["input_ids"])
        return result

    async def process_mm_data_async(
        self,
        image_data: Optional[list],
        audio_data: Optional[list],
        input_text: str | list[int],
        request_obj,
        **kwargs,
    ) -> dict:
        video_data = request_obj.video_data
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )

        images_resized = False
        # Precomputed inputs keep their own processing unless they ask for a resize.
        if base_output.images and (
            all(isinstance(image, Image.Image) for image in base_output.images)
            or any(_max_long_side_pixel(item) is not None for item in image_data or [])
        ):
            base_output.images = self._resize_images_for_request(
                base_output.images, image_data
            )
            images_resized = True

        video_metadata = None
        if base_output.videos:
            image_factor, max_size = self._video_resize_config()
            video_items = video_data or []
            if len(video_items) != len(base_output.videos):
                if any(
                    _video_fps(item) is not None
                    or _max_long_side_pixel(item) is not None
                    for item in video_items
                ):
                    raise ValueError(
                        f"per-video options need one loaded video per video item, "
                        f"got {len(base_output.videos)} videos for "
                        f"{len(video_items)} items"
                    )
                video_items = [None] * len(base_output.videos)
            videos_processed = []
            for video, item in zip(base_output.videos, video_items):
                fps = _check_video_fps(_video_fps(item))
                videos_processed.append(
                    await get_video_tensor(
                        video,
                        image_factor=image_factor,
                        max_size=max_size,
                        fps=self.video_fps if fps is None else fps,
                        frame_max_size=self.video_frame_max_size,
                        max_frames=self.video_max_frames,
                        resize_fn=getattr(
                            self._processor.video_processor, "resize_frames", None
                        ),
                        max_long_side_pixel=_check_max_long_side_pixel(
                            _max_long_side_pixel(item), image_factor
                        ),
                    )
                )
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output=base_output,
            mm_tokens=self.mm_tokens,
            video_metadata=video_metadata,
            **({"images_resized": True} if images_resized else {}),
            **({"input_ids": input_text} if isinstance(input_text, list) else {}),
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        image_grids = kwargs.get("img_grid_thw")
        video_grids = kwargs.get("video_grid_thw")
        replacements = {
            self.IM_TOKEN_ID: [
                self._encode_media_text(self._image_placeholder_text(grid))
                for grid in ([] if image_grids is None else image_grids)
            ],
            self.VIDEO_TOKEN_ID: [
                self._encode_media_text(self._video_placeholder_text(grid, timestamps))
                for grid, timestamps in zip(
                    [] if video_grids is None else video_grids,
                    kwargs.get("video_timestamps") or itertools.repeat(None),
                )
            ],
        }
        ids = (
            list(prompt)
            if isinstance(prompt, list)
            else self._tokenizer.encode(prompt, add_special_tokens=False)
        )
        input_ids = expand_media_tokens(
            ids, replacements, im_start_id=self.IM_START_TOKEN_ID
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=self._precomputed_media_items(
                input_ids=input_ids,
                embeddings=embeddings,
                frames_per_video=(
                    [] if video_grids is None else [int(g[0]) for g in video_grids]
                ),
            ),
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )

    def _encode_media_text(self, text: str) -> List[int]:
        return self._processor.tokenizer.encode(text, add_special_tokens=False)

    def _image_placeholder_text(self, grid) -> str:
        merge_size = self._processor.image_processor.merge_size
        return (
            self.IMAGE_START_TOKEN
            + self.IMAGE_TOKEN * (int(grid[0] * grid[1] * grid[2]) // merge_size**2)
            + self.IMAGE_END_TOKEN
        )

    def _video_placeholder_text(self, grid, timestamps: Optional[List[float]]) -> str:
        merge_size = self._processor.video_processor.merge_size
        frame = (
            self.IMAGE_START_TOKEN
            + self.VIDEO_TOKEN * (int(grid[1] * grid[2]) // merge_size**2)
            + self.IMAGE_END_TOKEN
        )
        if timestamps is None:
            return frame * int(grid[0])
        return "".join(f"]<]{ts:.1f} seconds[>[" + frame for ts in timestamps)

    def _precomputed_media_items(
        self, *, input_ids: List[int], embeddings: dict, frames_per_video: List[int]
    ) -> List[MultimodalDataItem]:
        image_runs = _media_token_runs(input_ids, self.IM_TOKEN_ID)
        runs_per_item = {
            Modality.IMAGE: (image_runs, [1] * len(image_runs)),
            Modality.VIDEO: (
                _media_token_runs(input_ids, self.VIDEO_TOKEN_ID),
                frames_per_video,
            ),
        }
        items = []
        for modality, (runs, item_run_counts) in runs_per_item.items():
            run_start = row_start = 0
            for run_count in item_run_counts:
                offsets = runs[run_start : run_start + run_count]
                num_rows = sum(end - start + 1 for start, end in offsets)
                items.append(
                    MultimodalDataItem(
                        modality=modality,
                        offsets=offsets,
                        precomputed_embeddings=embeddings[modality][
                            row_start : row_start + num_rows
                        ],
                    )
                )
                run_start += run_count
                row_start += num_rows
        return items
