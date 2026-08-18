"""Kimi K3 multimodal processor.

GPU image preprocessing dedicated to K3: unlike the K2.5 wrapper it keeps
the alpha channel through the bicubic resize and then composites RGBA
images onto the checkpoint-configured background
(``transparent_bg_config`` with ``transparent_bg_fill_stage ==
"after_resize"`` in preprocessor_config.json), instead of dropping alpha
at load time.
"""

import functools
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.multimodal.cache import resolve_multimodal_item_hash
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    KimiK3DeferredPreprocessing,
)
from sglang.srt.multimodal.kimi_k3_image_processing import (
    fill_transparent_bg as _fill_transparent_bg,
)
from sglang.srt.multimodal.kimi_k3_image_processing import (
    to_chw_uint8,
)
from sglang.srt.multimodal.media_artifacts import (
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)
from sglang.srt.multimodal.media_artifacts.kimi_k3 import (
    KimiK3ImagePreprocessArtifact,
    KimiK3PreprocessConfig,
    KimiK3ResizeConfig,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    _get_image_dimensions,
    _gpu_preprocess_images,
    _grid_thw_from_resize_config,
    navit_resize_config,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.srt.utils import is_cuda


def _encode_k3_special_tokens(tokenizer, text: str) -> list[int]:
    """Encode K3 control tokens without allowing them to be BPE-split."""
    try:
        return list(tokenizer.encode(text, allowed_special="all"))
    except TypeError:
        # Keep the helper usable with lightweight tokenizer stubs in CPU tests.
        return list(tokenizer.encode(text))


@dataclass(frozen=True)
class _KimiK3VisualItem:
    """One MoonViT input grid derived from an image or a video chunk."""

    media_type: Literal["image", "video"]
    frames: tuple[Any, ...]
    source_index: int
    in_patch_limit: int
    timestamp_text: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return _get_image_dimensions(self.frames[0])


def _format_k3_timestamp(timestamp: float, mode: str) -> str:
    total_seconds = max(0, int(timestamp))
    milliseconds = int((max(0.0, timestamp) % 1) * 1000)
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    if mode == "hh:mm:ss.fff":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    if mode == "mm:ss.fff":
        return f"{(total_seconds // 60) % 60:02d}:{seconds:02d}.{milliseconds:03d}"
    if mode == "mm:ss":
        return f"{(total_seconds // 60) % 60:02d}:{seconds:02d}"
    raise ValueError(f"Invalid Kimi-K3 timestamp mode: {mode}")


def _video_frame_to_chw(frame):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
    if isinstance(frame, torch.Tensor):
        if frame.ndim != 3:
            raise ValueError(f"Expected a 3D video frame, got shape {frame.shape}.")
        if frame.shape[-1] in (1, 3, 4) and frame.shape[0] not in (1, 3, 4):
            frame = frame.permute(2, 0, 1)
        return frame.contiguous()
    if isinstance(frame, Image.Image):
        return frame
    raise TypeError(f"Unsupported Kimi-K3 video frame type: {type(frame)}")


def _split_k3_video(
    video,
    media_proc_cfg: dict,
    source_index: int,
) -> list[_KimiK3VisualItem]:
    total_frames = len(video)
    if total_frames <= 0:
        raise ValueError("Kimi-K3 video input must contain at least one frame.")

    avg_fps = float(getattr(video, "avg_fps", 0) or 0)
    if avg_fps <= 0:
        raise ValueError("Kimi-K3 video input must report a positive frame rate.")

    sample_fps = min(float(media_proc_cfg.get("sample_fps", 8.0)), avg_fps)
    sampled_nframes = max(round(total_frames * sample_fps / avg_fps), 1)
    max_frames = media_proc_cfg.get("max_num_frames_each_video")
    if max_frames is not None:
        sampled_nframes = min(sampled_nframes, max(1, int(max_frames)))
    sampled_nframes = min(sampled_nframes, total_frames)

    frame_indices = (
        np.linspace(0, total_frames - 1, sampled_nframes).round().astype(int).tolist()
    )
    if hasattr(video, "get_frames_as_tensor"):
        decoded_frames = video.get_frames_as_tensor(frame_indices)
    elif hasattr(video, "get_frames_at"):
        decoded_frames = video.get_frames_at(frame_indices)
    else:
        decoded_frames = [video[index] for index in frame_indices]
    frames = [_video_frame_to_chw(frame) for frame in decoded_frames]

    temporal_merge = int(media_proc_cfg.get("temporal_merge_kernel_size", 4))
    if temporal_merge < 1 or temporal_merge > 4:
        raise ValueError("Kimi-K3 temporal_merge_kernel_size must be between 1 and 4.")

    per_frame_limit = int(
        media_proc_cfg.get("in_patch_limit_each_frame")
        or media_proc_cfg["in_patch_limit"]
    )
    total_patch_limit = media_proc_cfg.get("in_patch_limit_video")
    if total_patch_limit is not None:
        per_frame_limit = min(
            per_frame_limit,
            max(1, round(int(total_patch_limit) / sampled_nframes)),
        )

    timestamp_mode = media_proc_cfg.get("timestamp_mode", "hh:mm:ss.fff")
    chunks = []
    for start in range(0, sampled_nframes, temporal_merge):
        chunks.append(
            _KimiK3VisualItem(
                media_type="video",
                frames=tuple(frames[start : start + temporal_merge]),
                source_index=source_index,
                in_patch_limit=per_frame_limit,
                timestamp_text=_format_k3_timestamp(
                    frame_indices[start] / avg_fps, timestamp_mode
                ),
            )
        )
    return chunks


def _expand_k3_image_prompt_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
    tokenizer,
) -> torch.Tensor:
    """Expand K3 image placeholders into the checkpoint's media contract.

    K3 requires each image feature span to be enclosed by its original uploaded
    dimensions.  The chat template deliberately emits one ``media_pad`` per
    image; after decode, insert the surrounding control tokens and expand that
    one placeholder to the NaViT feature count.
    """
    if len(image_token_counts) != len(image_sizes):
        raise ValueError("Expected one original size for each K3 image.")

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    placeholder_count = np.count_nonzero(input_ids == image_token_id)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder token(s), "
            f"found {placeholder_count}."
        )

    output = []
    image_index = 0
    for token_id in input_ids:
        if token_id != image_token_id:
            output.append(int(token_id))
            continue

        width, height = image_sizes[image_index]
        output.extend(
            _encode_k3_special_tokens(
                tokenizer,
                f"<|media_begin|>image {width}x{height}<|media_content|>",
            )
        )
        output.extend([image_token_id] * image_token_counts[image_index])
        output.extend(_encode_k3_special_tokens(tokenizer, "<|media_end|>"))
        image_index += 1

    return torch.tensor(output, dtype=torch.long).unsqueeze(0)


def _expand_k3_visual_prompt_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    visual_items: list[_KimiK3VisualItem],
    resize_configs: list[dict],
    tokenizer,
) -> torch.Tensor:
    """Replace each source-media placeholder with its MoonViT grid(s)."""
    if len(visual_items) != len(resize_configs):
        raise ValueError("Expected one resize config for each Kimi-K3 visual item.")

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    groups: list[list[tuple[_KimiK3VisualItem, dict]]] = []
    for item, config in zip(visual_items, resize_configs):
        if item.source_index == len(groups):
            groups.append([])
        elif item.source_index != len(groups) - 1:
            raise ValueError("Kimi-K3 visual items are not ordered by source media.")
        groups[item.source_index].append((item, config))

    placeholder_count = int(np.count_nonzero(input_ids == image_token_id))
    if placeholder_count != len(groups):
        raise ValueError(
            f"Expected {len(groups)} visual placeholder token(s), "
            f"found {placeholder_count}."
        )

    output = []
    source_index = 0
    for token_id in input_ids:
        if token_id != image_token_id:
            output.append(int(token_id))
            continue

        group = groups[source_index]
        if group[0][0].media_type == "image" and len(group) != 1:
            raise ValueError("A Kimi-K3 image source must produce exactly one grid.")
        for item, config in group:
            if item.media_type == "image":
                width, height = item.size
                media_prefix = f"<|media_begin|>image {width}x{height}<|media_content|>"
            else:
                media_prefix = (
                    f"{item.timestamp_text}<|media_begin|>video<|media_content|>"
                )
            output.extend(_encode_k3_special_tokens(tokenizer, media_prefix))
            output.extend([image_token_id] * int(config["num_tokens"]))
            output.extend(_encode_k3_special_tokens(tokenizer, "<|media_end|>"))
        source_index += 1

    return torch.tensor(output, dtype=torch.long).unsqueeze(0)


def _expand_k3_image_prompt_text(
    input_text: str,
    image_token: str,
    image_token_counts: List[int],
    image_sizes: List[tuple[int, int]],
) -> str:
    """Render the K3 media framing for the CPU HF-processor fallback."""
    parts = input_text.split(image_token)
    if len(parts) - 1 != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder(s), "
            f"found {len(parts) - 1}."
        )

    output = [parts[0]]
    for image_token_count, (width, height), suffix in zip(
        image_token_counts, image_sizes, parts[1:]
    ):
        output.extend(
            (
                f"<|media_begin|>image {width}x{height}<|media_content|>",
                image_token * image_token_count,
                "<|media_end|>",
                suffix,
            )
        )
    return "".join(output)


def _k3_to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return to_chw_uint8(image, device="cuda")

    image = image.cuda()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def _gpu_preprocess_k3_visual_items(
    visual_items: list[_KimiK3VisualItem],
    resize_configs: list[dict],
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    transparent_bg_config: Optional[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess image/video frames and restore each item's temporal grid."""
    if len(visual_items) != len(resize_configs):
        raise ValueError("Expected one resize config for each Kimi-K3 visual item.")

    frames = []
    frame_configs = []
    for item, config in zip(visual_items, resize_configs):
        frames.extend(item.frames)
        frame_configs.extend([config] * len(item.frames))

    pixel_values, frame_grids = _gpu_preprocess_images(
        frames,
        frame_configs,
        image_scale,
        image_bias,
        patch_size,
        to_chw=_k3_to_cuda_chw,
        post_resize=lambda x: _fill_transparent_bg(x, transparent_bg_config),
    )

    grids = []
    frame_index = 0
    for item in visual_items:
        item_grids = frame_grids[frame_index : frame_index + len(item.frames)]
        frame_index += len(item.frames)
        if len(item_grids) == 0 or not torch.all(item_grids[:, 0] == 1):
            raise ValueError("Expected one spatial MoonViT grid per video frame.")
        if not torch.all(item_grids[:, 1:] == item_grids[0, 1:]):
            raise ValueError("Kimi-K3 video frames must share one spatial grid.")
        grids.append((len(item.frames), *item_grids[0, 1:].tolist()))

    return pixel_values, torch.tensor(grids, dtype=torch.int64)


class KimiK3GPUProcessorWrapper(KimiGPUProcessorWrapper):
    def __init__(self, hf_processor, image_token, image_token_id, config):
        self.preprocess_config = config
        super().__init__(
            hf_processor,
            image_token=image_token,
            image_token_id=image_token_id,
            patch_size=config.patch_size,
            merge_kernel_size=config.merge_kernel_size,
            in_patch_limit=config.in_patch_limit,
            patch_limit_on_one_side=config.patch_limit_on_one_side,
            fixed_output_tokens=config.fixed_output_tokens,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )
        self._transparent_bg_config = config.transparent_bg_config

    def preprocess_fingerprint_payload(self):
        return self.preprocess_config

    def _normalize_visual_items(self, images) -> list[_KimiK3VisualItem]:
        return [
            (
                image
                if isinstance(image, _KimiK3VisualItem)
                else _KimiK3VisualItem(
                    media_type="image",
                    frames=(image,),
                    source_index=index,
                    in_patch_limit=self._in_patch_limit,
                )
            )
            for index, image in enumerate(images or [])
        ]

    def _prepare_input_ids(
        self, input_text, resize_configs, original_input_ids, image_sizes
    ):
        image_token_counts = [config["num_tokens"] for config in resize_configs]
        if original_input_ids is None:
            original_input_ids = _encode_k3_special_tokens(
                self._hf_processor.tokenizer, input_text
            )
        return _expand_k3_image_prompt_token_ids(
            original_input_ids,
            self._image_token_id,
            image_token_counts,
            image_sizes,
            self._hf_processor.tokenizer,
        )

    def __call__(self, text=None, images=None, **kwargs):
        images = images or kwargs.pop("images", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)
        if images and torch.cuda.is_available():
            return self._gpu_call(text, images, original_input_ids)
        if images and any(
            isinstance(image, _KimiK3VisualItem) and image.media_type == "video"
            for image in images
        ):
            raise RuntimeError("Kimi-K3 video preprocessing requires CUDA.")
        return self._cpu_call(text, images, original_input_ids, **kwargs)

    def _gpu_call(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text

        if any(isinstance(image, _KimiK3VisualItem) for image in images):
            visual_items = self._normalize_visual_items(images)
            resize_configs = []
            for item in visual_items:
                width, height = item.size
                resize_configs.append(
                    navit_resize_config(
                        width,
                        height,
                        self._patch_size,
                        self._merge_kernel_size,
                        item.in_patch_limit,
                        self._patch_limit_on_one_side,
                        self._fixed_output_tokens,
                    )
                )

            if original_input_ids is None:
                original_input_ids = _encode_k3_special_tokens(
                    self._hf_processor.tokenizer, input_text
                )
            input_ids = _expand_k3_visual_prompt_token_ids(
                original_input_ids,
                self._image_token_id,
                visual_items,
                resize_configs,
                self._hf_processor.tokenizer,
            )
            image_scale, image_bias = self._get_gpu_norm_tensors()
            pixel_values, grid_thws = _gpu_preprocess_k3_visual_items(
                visual_items,
                resize_configs,
                image_scale,
                image_bias,
                self._patch_size,
                self._transparent_bg_config,
            )
            return {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thws,
            }

        resize_configs = []
        image_sizes = []
        for image in images:
            w, h = _get_image_dimensions(image)
            image_sizes.append((w, h))
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )

        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids, image_sizes
        )

        image_scale, image_bias = self._get_gpu_norm_tensors()
        # Shared source-compatible batched pipeline (same as K2.5): RGBA
        # inputs land in their own source-shape groups, and the
        # transparent-background compositing runs on each resized batch
        # before patchify -- identical order to the previous per-image path.
        pixel_values, grid_thws = _gpu_preprocess_images(
            images,
            resize_configs,
            image_scale,
            image_bias,
            self._patch_size,
            to_chw=_k3_to_cuda_chw,
            post_resize=lambda x: _fill_transparent_bg(x, self._transparent_bg_config),
        )

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thws,
        }

    def _cpu_call(self, text, images, original_input_ids=None, **kwargs):
        """HF fallback with the same K3 media framing as the GPU path."""
        input_text = text[0] if isinstance(text, list) else text
        if not images:
            return self._hf_processor(text=[input_text], **kwargs)

        image_sizes = [_get_image_dimensions(image) for image in images]
        image_token_counts = [
            self._hf_processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            for image in images
        ]
        expanded_text = _expand_k3_image_prompt_text(
            input_text,
            self._image_token,
            image_token_counts,
            image_sizes,
        )
        kwargs["medias"] = [{"type": "image", "image": image} for image in images]
        out = self._hf_processor(text=[expanded_text], **kwargs)
        out["input_ids"] = self._prepare_input_ids(
            input_text,
            [{"num_tokens": count} for count in image_token_counts],
            original_input_ids,
            image_sizes,
        )
        grid_thws = out.pop("grid_thws", None)
        if grid_thws is not None:
            out["image_grid_thw"] = grid_thws
        return out

    def prepare_deferred(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text
        image_sizes = [_get_image_dimensions(image) for image in images]
        resize_configs = [
            navit_resize_config(
                width,
                height,
                self._patch_size,
                self._merge_kernel_size,
                self._in_patch_limit,
                self._patch_limit_on_one_side,
                self._fixed_output_tokens,
            )
            for width, height in image_sizes
        ]
        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids, image_sizes
        )
        # This path only ever defers GPU preprocessing: the caller gates on
        # `_should_defer_gpu_preprocessing` and stages CHW uint8 features.
        deferred_preprocessing = functools.partial(
            KimiK3DeferredPreprocessing,
            backend="gpu",
            image_mean=list(self._image_mean),
            image_std=list(self._image_std),
            transparent_bg_config=self._transparent_bg_config,
        )
        return input_ids, resize_configs, deferred_preprocessing

    def prepare_image_features(self, images):
        """Prepare prompt-independent, per-image features in one processor call."""
        image_sizes = [_get_image_dimensions(image) for image in images]
        resize_configs = [
            navit_resize_config(
                width,
                height,
                self._patch_size,
                self._merge_kernel_size,
                self._in_patch_limit,
                self._patch_limit_on_one_side,
                self._fixed_output_tokens,
            )
            for width, height in image_sizes
        ]

        if images and torch.cuda.is_available():
            image_scale, image_bias = self._get_gpu_norm_tensors()
            pixel_values, grid_thws = _gpu_preprocess_images(
                images,
                resize_configs,
                image_scale,
                image_bias,
                self._patch_size,
                to_chw=_k3_to_cuda_chw,
                post_resize=lambda x: _fill_transparent_bg(
                    x, self._transparent_bg_config
                ),
            )
        else:
            # The checkpoint CPU processor couples prompt composition with media
            # preprocessing. A synthetic prompt keeps that API but is discarded;
            # image features and grids are independent of its text.
            output = self._cpu_call(self._image_token * len(images), images)
            pixel_values = output["pixel_values"]
            grid_thws = output["image_grid_thw"]

        grids = [tuple(int(value) for value in grid) for grid in grid_thws.tolist()]
        patch_counts = [math.prod(grid) for grid in grids]
        if sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError(
                "Kimi-K3 processor feature length does not match image grids: "
                f"{pixel_values.shape[0]} != {sum(patch_counts)}"
            )
        return (
            list(pixel_values.split(patch_counts)),
            image_sizes,
            resize_configs,
            grids,
        )


class KimiK3ImageProcessor(
    KimiGridMMDataMixin,
    MediaArtifactCacheMixin,
    SGLangBaseProcessor,
):
    models = [KimiK3ForConditionalGeneration]
    artifact_modality = Modality.IMAGE
    # K3 accuracy is sensitive to the chroma upsampling used for common 4:2:0
    # JPEG inputs. This mode uses interpolated nvJPEG upsampling when the K3
    # image dependency is installed and otherwise falls back to PIL.
    gpu_image_decode = "nvjpeg_fancy"
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    auto_mm_preprocess_cache_size_mb = 256
    supports_mm_processor_concurrency = True
    preserve_processor_input_ids = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        preprocess_config = KimiK3PreprocessConfig.from_media_processor(
            _processor.media_processor
        )

        processor = KimiK3GPUProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            config=preprocess_config,
        )
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens
        self.media_proc_cfg = dict(_processor.media_processor.media_proc_cfg)

    @staticmethod
    def _resolve_visual_modalities(request_obj, image_count, video_count):
        modalities = []
        for modality in getattr(request_obj, "modalities", None) or []:
            if isinstance(modality, list):
                modalities.extend(modality)
            else:
                modalities.append(modality)

        if not modalities:
            return ["image"] * image_count + ["video"] * video_count
        if (
            any(modality not in ("image", "video") for modality in modalities)
            or modalities.count("image") != image_count
            or modalities.count("video") != video_count
        ):
            raise ValueError(
                "Kimi-K3 media order does not match the supplied image/video data."
            )
        return modalities

    def _should_defer_gpu_preprocessing(self, images) -> bool:
        """
        when raw_bytes <= processed_bytes, preprocess first would introduce larger payload, so deferring gpu preprocessing would benefit
        """
        if (
            not images
            or self.mm_feature_transport != "cpu"
            or not is_cuda()
            or not all(
                isinstance(image, Image.Image)
                or (isinstance(image, torch.Tensor) and image.dtype == torch.uint8)
                for image in images
            )
        ):
            return False

        raw_bytes = 0
        processed_bytes = 0
        config = self._processor.preprocess_config
        patch_size = config.patch_size
        for image in images:
            width, height = _get_image_dimensions(image)
            resize_config = navit_resize_config(
                width,
                height,
                patch_size,
                config.merge_kernel_size,
                config.in_patch_limit,
                config.patch_limit_on_one_side,
                config.fixed_output_tokens,
            )
            if isinstance(image, torch.Tensor):
                channels = (
                    3 if image.dim() == 2 or image.shape[0] == 1 else image.shape[0]
                )
            else:
                channels = (
                    4
                    if image.mode != "RGB"
                    and ("A" in image.getbands() or "transparency" in image.info)
                    else 3
                )
            raw_bytes += channels * width * height
            padded_width = resize_config["new_width"] + resize_config["pad_width"]
            padded_height = resize_config["new_height"] + resize_config["pad_height"]
            processed_bytes += 3 * padded_width * padded_height * torch.float32.itemsize

        return raw_bytes <= processed_bytes

    def _build_deferred_output(self, base_output):
        (
            input_ids,
            resize_configs,
            deferred_preprocessing,
        ) = self._processor.prepare_deferred(
            base_output.input_text,
            base_output.images,
            base_output.input_ids,
        )
        offsets = self.get_mm_items_offset(
            input_ids.flatten(), self.mm_tokens.image_token_id
        )
        if len(offsets) != len(base_output.images):
            raise ValueError("Expected one Kimi-K3 image span for each image")

        items = []
        for image, resize_config, offset in zip(
            base_output.images, resize_configs, offsets
        ):
            grid_thw = _grid_thw_from_resize_config(
                resize_config, self._processor.preprocess_config.patch_size
            )
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=to_chw_uint8(image),
                offsets=[offset],
                model_specific_data={
                    "image_grid_thw": torch.tensor([grid_thw], dtype=torch.int64),
                    DEFERRED_PREPROCESSING_KEY: deferred_preprocessing(
                        resize_config=resize_config
                    ),
                },
            )
            items.append(item)

        self._precompute_hashes_before_cpu_transfer(items)
        return MultimodalProcessorOutput(
            input_ids=input_ids.flatten().tolist(),
            mm_items=items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def _make_artifact(
        self,
        *,
        content_digest: str,
        artifact_key: str,
        original_size: tuple[int, int],
        resize_config: dict,
        grid_thw: tuple[int, int, int],
        feature: torch.Tensor,
        deferred: Optional[KimiK3DeferredPreprocessing] = None,
    ) -> KimiK3ImagePreprocessArtifact:
        """Freeze one image's prompt-independent preprocessing result."""
        # Use the same feature-hash contract as MultimodalDataItem.
        feature_hash = resolve_multimodal_item_hash(
            feature=feature, namespace=artifact_key
        )
        if not self.keep_mm_features_on_device and feature.device.type != "cpu":
            feature = feature.cpu()
        return KimiK3ImagePreprocessArtifact(
            content_digest=content_digest,
            artifact_key=artifact_key,
            feature_hash=feature_hash,
            original_size=original_size,
            resize_config=KimiK3ResizeConfig.from_dict(resize_config),
            grid_thw=grid_thw,
            feature=feature,
            deferred=deferred,
        )

    def prepare_artifact_batch(
        self,
        entries: list[MediaArtifactInput],
        *,
        processor=None,
    ) -> list[KimiK3ImagePreprocessArtifact]:
        """Preprocess raw cache misses into reusable per-image cache items.

        Each entry is a confirmed cache miss. It is either processed now or
        stored with the metadata needed for deferred GPU preprocessing.
        """
        processor = processor or self._processor
        artifacts: list[Optional[KimiK3ImagePreprocessArtifact]] = [None] * len(entries)
        # 1. collect inputs that must be preprocessed now instead of deferred
        eager_entry_indices = []
        eager_images = []

        config = processor.preprocess_config
        for index, entry in enumerate(entries):
            image = entry.media
            if not self._should_defer_gpu_preprocessing([image]):
                eager_entry_indices.append(index)
                eager_images.append(image)
                continue

            width, height = _get_image_dimensions(image)
            resize_config = navit_resize_config(
                width,
                height,
                config.patch_size,
                config.merge_kernel_size,
                config.in_patch_limit,
                config.patch_limit_on_one_side,
                config.fixed_output_tokens,
            )
            grid_thw = _grid_thw_from_resize_config(resize_config, config.patch_size)
            feature = to_chw_uint8(image).cpu().contiguous()
            artifacts[index] = self._make_artifact(
                content_digest=entry.content_digest,
                artifact_key=entry.artifact_key,
                original_size=(width, height),
                resize_config=resize_config,
                grid_thw=grid_thw,
                feature=feature,
                deferred=KimiK3DeferredPreprocessing(
                    backend="gpu",
                    image_mean=list(config.image_mean),
                    image_std=list(config.image_std),
                    transparent_bg_config=config.transparent_bg_config,
                    resize_config=resize_config,
                ),
            )

        # 2. preprocess CPU eager inputs as one batch
        if eager_images:
            features, sizes, configs, grids = processor.prepare_image_features(
                eager_images
            )
            for index, feature, size, resize_config, grid in zip(
                eager_entry_indices, features, sizes, configs, grids
            ):
                entry = entries[index]
                artifacts[index] = self._make_artifact(
                    content_digest=entry.content_digest,
                    artifact_key=entry.artifact_key,
                    original_size=size,
                    resize_config=resize_config,
                    grid_thw=grid,
                    feature=feature,
                )

        # 3. return artifacts in the original processor-input order
        if any(artifact is None for artifact in artifacts):
            raise RuntimeError("Kimi-K3 artifact batch did not produce every image")
        return [artifact for artifact in artifacts if artifact is not None]

    def compose_request(
        self,
        input_text,
        artifacts: list[KimiK3ImagePreprocessArtifact],
    ) -> MultimodalProcessorOutput:
        """Compose the current request from its prompt and ordered artifacts.

        ``prepare_media_artifacts`` has already returned one artifact for each
        processor input, either from the preprocess cache or from fresh
        preprocessing. This method expands the current prompt's image tokens
        and converts each artifact into its request-specific
        ``MultimodalDataItem`` with offsets, grid metadata, feature, and feature
        hash. It does not read raw media or access the preprocess cache.
        """
        # 1. rebuild prompt-specific tokens and offsets
        original_ids = (
            input_text
            if isinstance(input_text, (list, torch.Tensor))
            else _encode_k3_special_tokens(self._tokenizer, input_text)
        )
        input_ids = _expand_k3_image_prompt_token_ids(
            original_ids,
            self.mm_tokens.image_token_id,
            [artifact.resize_config.num_tokens for artifact in artifacts],
            [artifact.original_size for artifact in artifacts],
            self._tokenizer,
        ).flatten()
        offsets = self.get_mm_items_offset(input_ids, self.mm_tokens.image_token_id)
        if len(offsets) != len(artifacts):
            raise ValueError("Expected one Kimi-K3 image span for each image")

        # 2. build request-owned items from prompt-independent artifacts
        items = []
        for artifact, offset in zip(artifacts, offsets):
            model_specific_data = {
                "image_grid_thw": torch.tensor([artifact.grid_thw], dtype=torch.int64)
            }
            if artifact.deferred is not None:
                model_specific_data[DEFERRED_PREPROCESSING_KEY] = artifact.deferred
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=artifact.feature,
                offsets=[offset],
                model_specific_data=model_specific_data,
            )
            item.set_hash(artifact.feature_hash)
            if self.use_cuda_ipc and isinstance(item.feature, torch.Tensor):
                item.feature = self._wrap_tensor_for_cuda_ipc(item.feature)
            if self.keep_mm_features_on_device and item.feature is not None:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
            items.append(item)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    async def _process_mm_data_uncached(
        self, image_data, input_text, request_obj, **kwargs
    ):
        """Compatibility path for precomputed inputs and lightweight test stubs."""
        image_data = list(image_data or [])
        video_data = list(getattr(request_obj, "video_data", None) or [])
        expected_image_count = len(image_data or [])
        placeholder_count = self.count_image_placeholders(
            input_text, self.mm_tokens.image_token_id
        )
        if video_data:
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                video_data=video_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
                input_ids=input_text if placeholder_count is not None else None,
            )
        elif placeholder_count is not None:
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
                input_ids=input_text,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
            )
        loaded_images = list(getattr(base_output, "images", None) or [])
        loaded_videos = list(getattr(base_output, "videos", None) or [])
        if len(loaded_images) != expected_image_count or len(loaded_videos) != len(
            video_data
        ):
            raise ValueError(
                "Kimi visual placeholders must map one-to-one to source media: "
                f"expected {expected_image_count} image(s) and {len(video_data)} "
                f"video(s), loaded {len(loaded_images)} image(s) and "
                f"{len(loaded_videos)} video(s)"
            )

        if video_data:
            modalities = self._resolve_visual_modalities(
                request_obj, expected_image_count, len(video_data)
            )
            image_iter = iter(loaded_images)
            video_iter = iter(loaded_videos)
            visual_items = []
            for source_index, modality in enumerate(modalities):
                if modality == "image":
                    visual_items.append(
                        _KimiK3VisualItem(
                            media_type="image",
                            frames=(next(image_iter),),
                            source_index=source_index,
                            in_patch_limit=int(self.media_proc_cfg["in_patch_limit"]),
                        )
                    )
                    continue

                video = next(video_iter)
                try:
                    visual_items.extend(
                        _split_k3_video(video, self.media_proc_cfg, source_index)
                    )
                finally:
                    close = getattr(video, "close", None)
                    if close is not None:
                        close()

            # K3 uses the image placeholder id for every MoonViT grid. The T
            # dimension distinguishes temporal video chunks from still images.
            base_output.images = visual_items
            base_output.videos = []

        if self._should_defer_gpu_preprocessing(base_output.images):
            return self._build_deferred_output(base_output)
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output,
            self.mm_tokens,
            sglang_original_input_ids=base_output.input_ids,
        )
        if self.keep_mm_features_on_device:
            for item in mm_items:
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        audio_data = kwargs.get("audio_data") or getattr(
            request_obj, "audio_data", None
        )
        if audio_data:
            raise ValueError(
                "Kimi-K3 supports image and video input (without audio track) only"
            )

        image_data = list(image_data or [])
        video_data = list(getattr(request_obj, "video_data", None) or [])
        expected_image_count = len(image_data or [])
        expected_media_count = expected_image_count + len(video_data)
        placeholder_count = self.count_image_placeholders(
            input_text, self.mm_tokens.image_token_id
        )
        if placeholder_count is not None:
            if placeholder_count != expected_media_count:
                raise ValueError(
                    "Kimi visual placeholders must map one-to-one to source media: "
                    f"expected {expected_media_count}, found {placeholder_count} "
                    "token(s)"
                )
        elif video_data:
            placeholder_count = input_text.count(self.mm_tokens.image_token)
            if placeholder_count != expected_media_count:
                raise ValueError(
                    "Kimi visual placeholders must map one-to-one to source media: "
                    f"expected {expected_media_count}, found {placeholder_count}."
                )

        if video_data:
            self._resolve_visual_modalities(
                request_obj, expected_image_count, len(video_data)
            )
        if (
            video_data
            or any(self._is_preprocessed_input(item) for item in image_data)
            or not self.mm_preprocess_cache.enabled
        ):
            # 1. keep video, preprocessed inputs, and cache-off requests on the
            # uncached path. Image artifacts remain image-only.
            return await self._process_mm_data_uncached(
                image_data, input_text, request_obj, **kwargs
            )

        # 2. resolve per-image artifacts before composing the current prompt
        artifacts = await self.prepare_media_artifacts(
            image_data,
            content_hashes=request_obj.mm_content_hashes,
        )
        return self.compose_request(input_text, artifacts)

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        output = self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
        image_sizes = kwargs.get("original_image_sizes")
        if image_sizes is None:
            return output

        counts = [self._num_image_tokens_from_grid(grid) for grid in img_grid_thw]
        if len(image_sizes) != len(counts):
            raise ValueError(
                "Expected one original image size for each K3 encoder grid."
            )
        output.input_ids = (
            _expand_k3_image_prompt_token_ids(
                prompt,
                self.mm_tokens.image_token_id,
                counts,
                [tuple(size) for size in image_sizes],
                self._tokenizer,
            )
            .flatten()
            .tolist()
        )

        search_start = 0
        for item, count in zip(output.mm_items, counts):
            start = output.input_ids.index(self.mm_tokens.image_token_id, search_start)
            item.offsets = [(start, start + count - 1)]
            search_start = start + count
        return output
