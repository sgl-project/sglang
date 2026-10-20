"""Native image, video, and waveform preprocessing for Muse 1.2 Omni."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import re
from bisect import bisect_left, bisect_right
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers.utils.hub import cached_file

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.muse_spark_omni import MuseSparkForConditionalGeneration
from sglang.srt.multimodal.cache import resolve_multimodal_item_hash
from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput
from sglang.srt.multimodal.media_artifacts.base import (
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_mm, get_model
from sglang.srt.utils.hf_transformers.common import resolve_runai_obj_uri
from sglang.srt.utils.hf_transformers_utils import resolve_image_processor_backend
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 24_000
AUDIO_SAMPLES_PER_TOKEN = 1_920

# Checkpoint processor_config.json video_processor key -> PreprocessConfig field.
_VIDEO_PROCESSOR_FIELDS = {
    "num_frames": "num_frames_per_video",
    "max_video_frame_tokens": "max_tokens_per_frame",
    "frame_timestamp_format": "timestamp_format",
    "fps": "sampling_fps",
}


@dataclass
class PreprocessConfig:
    patch_size: int = 14
    patch_temporal: int = 2
    downsample_factor: int = 2
    min_image_tokens: int = 1
    max_image_tokens: int = 4096
    max_video_tokens: int = 4096
    use_per_video_token_budget: bool = False
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    num_frames_per_video: int = 96
    sampling_fps: Optional[float] = 2.0
    frame_factor: int = 2
    max_tokens_per_frame: int = 144
    timestamp_format: str = "Time: {s:.1f}s"
    use_rust_resize: bool = True
    use_gpu_preprocess: bool = False

    @property
    def token_patch_size(self) -> int:
        return self.patch_size * self.downsample_factor

    def format_timestamp(self, seconds: float) -> str:
        # Checkpoint templates name the value {timestamp}; the default uses {s}.
        return self.timestamp_format.format(s=seconds, timestamp=seconds)


@dataclass
class VideoFeatures:
    tensor: torch.Tensor
    patches_per_clip: int
    timestamps: list[float]


def _least_distortion_fit(
    ideal_h: float,
    ideal_w: float,
    image_h: int,
    image_w: int,
    patch_size: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    candidates = set(
        itertools.product(
            (math.floor(ideal_h), math.ceil(ideal_h)),
            (math.floor(ideal_w), math.ceil(ideal_w)),
        )
    )
    candidates = [
        (height, width)
        for height, width in candidates
        if height > 0 and width > 0 and minimum <= height * width <= maximum
    ]
    if not candidates:
        raise ValueError("no rounded resize target satisfies the token budget")
    aspect = image_h / image_w
    return min(candidates, key=lambda item: abs(item[0] / item[1] - aspect))


def _least_distance_fit(
    ideal_h: float, ideal_w: float, minimum: int, maximum: int
) -> tuple[int, int]:
    best = None
    for height in range(1, maximum + 1):
        lower = math.ceil(minimum / height)
        upper = math.floor(maximum / height)
        if lower > upper:
            continue
        width = min(max(round(ideal_w), lower), upper)
        distance = (height - ideal_h) ** 2 + (width - ideal_w) ** 2
        if best is None or distance < best[0]:
            best = (distance, height, width)
    if best is None:
        raise ValueError("no resize target satisfies the token budget")
    return best[1], best[2]


def find_resize_target(
    image_h: int,
    image_w: int,
    patch_size: int,
    minimum_tokens: int,
    maximum_tokens: int,
) -> tuple[int, int, int]:
    ideal_h = image_h / patch_size
    ideal_w = image_w / patch_size
    ideal_count = ideal_h * ideal_w
    aspect = ideal_w / ideal_h
    if ideal_count > maximum_tokens:
        ideal_h = math.sqrt(maximum_tokens / aspect)
        ideal_w = ideal_h * aspect
    elif ideal_count < minimum_tokens:
        ideal_h = math.sqrt(minimum_tokens / aspect)
        ideal_w = ideal_h * aspect
    try:
        patches_h, patches_w = _least_distortion_fit(
            ideal_h,
            ideal_w,
            image_h,
            image_w,
            patch_size,
            minimum_tokens,
            maximum_tokens,
        )
    except ValueError:
        patches_h, patches_w = _least_distance_fit(
            ideal_h, ideal_w, minimum_tokens, maximum_tokens
        )
    return patches_h * patch_size, patches_w * patch_size, patches_h * patches_w


def _image_to_uint8(
    image: Any, *, keep_device: bool = False, channel_last: bool = False
) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1)
    if isinstance(image, np.ndarray):
        value = torch.from_numpy(np.array(image, copy=True))
        channel_last = True
    else:
        value = torch.as_tensor(image)
        if not keep_device:
            value = value.cpu()
    if value.ndim != 3:
        raise ValueError(f"unexpected image shape {tuple(value.shape)}")
    if channel_last or value.shape[0] not in (1, 3, 4):
        if value.shape[-1] not in (1, 3, 4):
            raise ValueError(f"unexpected image channel shape {tuple(value.shape)}")
        value = value[..., :3].permute(2, 0, 1)
    else:
        value = value[:3]
    if value.dtype != torch.uint8:
        if value.is_floating_point() and value.max().item() <= 1.0:
            value = value * 255
        value = value.clamp(0, 255).to(torch.uint8)
    if value.shape[0] == 1:
        value = value.expand(3, -1, -1)
    return value.contiguous()


@lru_cache(maxsize=32)
def _normalization_lut(mean: tuple, std: tuple) -> np.ndarray:
    # Preserve the reference float32 operation order for all 256 pixel values.
    values = torch.arange(256, dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0)
    return (
        (values[None, :] - torch.tensor(mean)[:, None]) / torch.tensor(std)[:, None]
    ).numpy()


@lru_cache(maxsize=1)
def _rust_resize():
    from sglang.srt.rust_extensions import load_rust_extension

    try:
        return load_rust_extension(
            "sglang.srt.rust_extensions._multimodal", mode="never"
        ).common.resize_rgb
    except (ImportError, OSError, AttributeError, RuntimeError) as exc:
        # RuntimeError covers an unusable Rust toolchain: the loader wraps the
        # OSError from probing cargo, so mode="never" still raises here.
        logger.debug("Muse Rust resize unavailable; using Pillow: %s", exc)
        return None


def _resize_and_normalize(
    image: Any,
    target_h: int,
    target_w: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    use_rust_resize: bool = True,
    use_gpu_preprocess: bool = False,
) -> torch.Tensor:
    if use_gpu_preprocess:
        from sglang.srt.multimodal.muse_image_processing_gpu import resize_normalize_gpu

        return resize_normalize_gpu(
            _image_to_uint8(image, keep_device=True), target_h, target_w, mean, std
        )
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        value = _image_to_uint8(image)
        pil_image = Image.fromarray(value.permute(1, 2, 0).numpy())
    rust_resize = _rust_resize() if use_rust_resize else None
    if rust_resize is not None:
        array = np.asarray(pil_image)
        resized = rust_resize(array, target_w, target_h, "pil_lanczos")
        array = resized.reshape(target_h, target_w, 3)
    else:
        array = np.asarray(
            pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        )
    lut = _normalization_lut(tuple(mean), tuple(std))
    # A lookup avoids full-image float conversion and three arithmetic passes.
    normalized = np.empty((3, target_h, target_w), dtype=np.float32)
    for channel in range(3):
        normalized[channel] = lut[channel, array[:, :, channel]]
    return torch.from_numpy(normalized)


def preprocess_image(image: Any, config: PreprocessConfig) -> tuple[torch.Tensor, int]:
    if isinstance(image, Image.Image):
        width, height = image.size
        value = image
    else:
        value = _image_to_uint8(image, keep_device=config.use_gpu_preprocess)
        height, width = value.shape[-2:]
    target_h, target_w, token_count = find_resize_target(
        height,
        width,
        config.token_patch_size,
        config.min_image_tokens,
        config.max_image_tokens,
    )
    return (
        _resize_and_normalize(
            value,
            target_h,
            target_w,
            config.image_mean,
            config.image_std,
            config.use_rust_resize,
            config.use_gpu_preprocess,
        ),
        token_count,
    )


def _video_frames(
    video: Any, config: PreprocessConfig
) -> tuple[list[Any], list[float]]:
    if isinstance(video, tuple) and len(video) == 2:
        frames, metadata = video
        fps = float(metadata.get("fps", 1.0))
        source_indices = metadata.get("frames_indices", list(range(len(frames))))
        frames = list(frames)
    elif isinstance(video, torch.Tensor) and video.ndim == 4:
        frames = list(video)
        fps = 1.0
        source_indices = list(range(len(frames)))
    else:
        # Decoder-like input (e.g. torchcodec): len() and avg_fps are
        # metadata, so select the sampled indices first and decode only those
        # frames instead of decoding every frame.
        total = len(video)
        if total == 0:
            raise ValueError("video input has no frames")
        try:
            fps = float(video.avg_fps)
        except Exception:
            fps = 1.0
        if not math.isfinite(fps) or fps <= 0:
            logger.warning("Invalid video FPS %r; falling back to 1.0", fps)
            fps = 1.0
        factor = max(config.frame_factor, config.patch_temporal)
        if config.sampling_fps and fps > 0:
            requested = int(total * config.sampling_fps / fps)
        else:
            requested = total
        requested = min(max(requested, factor), config.num_frames_per_video, total)
        requested = max(factor, requested // factor * factor)
        if total > requested:
            selected = torch.linspace(0, total - 1, requested).long().tolist()
        else:
            selected = list(range(total))
        frames = [video[index] for index in selected]
        # VideoDecoderWrapper returns HWC even for CUDA tensors. Preserve that
        # contract instead of guessing from heights that can be 1, 3, or 4.
        if isinstance(video, VideoDecoderWrapper):
            frames = [
                _image_to_uint8(
                    frame, keep_device=config.use_gpu_preprocess, channel_last=True
                )
                for frame in frames
            ]
        source_indices = list(selected)
        while len(frames) < factor:
            frames.append(frames[-1])
            source_indices.append(source_indices[-1])
        if not frames:
            raise ValueError("video input has no frames")
        return frames, [float(index) / fps for index in source_indices]
    if not frames:
        raise ValueError("video input has no frames")
    if len(source_indices) != len(frames):
        raise ValueError("frames_indices must align with the supplied video frames")
    if not math.isfinite(fps) or fps <= 0:
        logger.warning("Invalid video FPS %r; falling back to 1.0", fps)
        fps = 1.0

    factor = max(config.frame_factor, config.patch_temporal)
    if config.sampling_fps and fps > 0:
        # Indices refer to the original video, not to the sampled frame list.
        source_span = source_indices[-1] - source_indices[0] + 1
        if source_span <= 0 or any(
            right < left for left, right in zip(source_indices, source_indices[1:])
        ):
            raise ValueError("frames_indices must be nondecreasing")
        requested = int(source_span * config.sampling_fps / fps)
    else:
        requested = len(frames)
    requested = min(max(requested, factor), config.num_frames_per_video, len(frames))
    requested = max(factor, requested // factor * factor)
    if len(frames) > requested:
        selected = torch.linspace(0, len(frames) - 1, requested).long().tolist()
        frames = [frames[index] for index in selected]
        source_indices = [source_indices[index] for index in selected]
    while len(frames) < factor:
        frames.append(frames[-1])
        source_indices.append(source_indices[-1])
    aligned = len(frames) // factor * factor
    frames = frames[:aligned]
    source_indices = source_indices[:aligned]
    return frames, [float(index) / fps for index in source_indices]


def preprocess_video(video: Any, config: PreprocessConfig) -> VideoFeatures:
    frames, timestamps = _video_frames(video, config)
    first = _image_to_uint8(frames[0], keep_device=config.use_gpu_preprocess)
    height, width = first.shape[-2:]
    if config.use_per_video_token_budget:
        num_clips = max(1, len(frames) // config.patch_temporal)
        per_video_maximum = config.max_video_tokens // num_clips
        if per_video_maximum < config.min_image_tokens:
            raise ValueError(
                "max_video_tokens cannot satisfy min_image_tokens for every clip"
            )
        maximum = per_video_maximum
        if config.max_tokens_per_frame > 0:
            maximum = min(maximum, config.max_tokens_per_frame)
    elif config.max_tokens_per_frame > 0:
        maximum = config.max_tokens_per_frame
    else:
        maximum = config.max_video_tokens
    target_h, target_w, patches_per_clip = find_resize_target(
        height,
        width,
        config.token_patch_size,
        config.min_image_tokens,
        maximum,
    )
    tensor = torch.stack(
        [
            _resize_and_normalize(
                frame,
                target_h,
                target_w,
                config.image_mean,
                config.image_std,
                config.use_rust_resize,
                config.use_gpu_preprocess,
            )
            for frame in frames
        ]
    )
    clip_timestamps = timestamps[:: config.patch_temporal]
    return VideoFeatures(tensor, patches_per_clip, clip_timestamps)


def preprocess_audio(
    audio: Any, samples_per_token: int = AUDIO_SAMPLES_PER_TOKEN
) -> tuple[torch.Tensor, int]:
    value = torch.as_tensor(audio, dtype=torch.float32)
    if value.ndim > 1:
        value = value.mean(dim=0)
    value = value.flatten()
    return value, math.ceil(value.numel() / samples_per_token)


def _load_video_processor_config(
    model_path: str, revision: Optional[str]
) -> dict[str, Any]:
    config_file = cached_file(
        resolve_runai_obj_uri(model_path),
        "processor_config.json",
        revision=revision,
        # None when the checkpoint has no processor_config.json.
        _raise_exceptions_for_missing_entries=False,
    )
    if config_file is None:
        return {}
    with open(config_file) as f:
        video = json.load(f).get("video_processor", {})
    return {
        field: video[key]
        for key, field in _VIDEO_PROCESSOR_FIELDS.items()
        if key in video
    }


def make_preprocess_config(vision) -> PreprocessConfig:
    return PreprocessConfig(
        use_rust_resize=resolve_image_processor_backend(get_mm()) != "pil",
        patch_size=vision.patch_size,
        patch_temporal=vision.patch_temporal,
        downsample_factor=vision.merge_size,
        **_load_video_processor_config(
            model_path=get_model().model_path, revision=get_model().revision
        ),
    )


def build_encoder_inputs(mm_data, modality, config, samples_per_token):
    """Use colocated preprocessing and send output grids alongside embeddings."""
    items, grids, features, timestamps = [], [], [], []
    for value in mm_data:
        if modality == Modality.IMAGE:
            feature, count = preprocess_image(value, config)
            grid = [
                1,
                feature.shape[-2] // config.token_patch_size,
                feature.shape[-1] // config.token_patch_size,
            ]
        elif modality == Modality.VIDEO:
            video = preprocess_video(value, config)
            feature = video.tensor
            grid = [
                len(video.timestamps),
                feature.shape[-2] // config.token_patch_size,
                feature.shape[-1] // config.token_patch_size,
            ]
            timestamps.append(video.timestamps)
        elif modality == Modality.AUDIO:
            feature, grid = preprocess_audio(value, samples_per_token)
        else:
            raise ValueError(f"Unsupported Muse encoder modality: {modality}")
        features.append(feature)
        grids.append(grid)
        items.append(MultimodalDataItem(modality=modality, feature=feature))
    grid_key, feature_key = {
        Modality.IMAGE: ("image_grid_thw", "pixel_values"),
        Modality.VIDEO: ("video_grid_thw", "pixel_values_videos"),
        Modality.AUDIO: ("audio_feature_lens_raw", "input_features"),
    }[modality]
    values = {grid_key: torch.tensor(grids, dtype=torch.long), feature_key: features}
    if modality == Modality.VIDEO:
        values["video_timestamps"] = timestamps
    return EncoderPreprocessOutput(values, mm_items=items)


def _token_id(tokenizer, token: str) -> int:
    token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
    encoding = getattr(tokenizer, "tokenizer", tokenizer)
    special_tokens = getattr(encoding, "_special_tokens", {})
    token_id = special_tokens.get(token, token_id)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        encoded = _encode(tokenizer, token)
        if len(encoded) != 1:
            raise ValueError(f"tokenizer does not define required token {token!r}")
        token_id = encoded[0]
    return int(token_id)


def _optional_token_id(tokenizer, token: str) -> Optional[int]:
    """Resolve a special token by name, or None when the tokenizer lacks it.

    Deliberately name-driven with no id constants: the delimiter ids are a
    function of the tokenizer's special-token ordering, so hardcoding them
    would silently produce wrong ids on any checkpoint whose vocab differs.
    Returning None keeps the decision at the use site, which rejects audio
    with a clear error rather than encoding tokens the model cannot consume.
    """
    try:
        return _token_id(tokenizer, token)
    except ValueError:
        return None


def _encode(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text, allowed_special="all"))


def _image_replacement(token_count: int) -> str:
    return "<|image_start|>" + "<|patch|>" * token_count + "<|image_end|>"


def _replace_once(text: str, placeholder: str, replacement: str) -> str:
    if placeholder not in text:
        raise ValueError(f"missing {placeholder!r} placeholder for supplied media")
    return text.replace(placeholder, replacement, 1)


def _expand_raw_placeholders(tokenizer, ids, placeholder, pattern, replacements):
    """Replace raw markers once, preserving token IDs outside their boundaries."""
    text = tokenizer.decode(ids)
    count = text.count(placeholder)
    if count > len(replacements):
        raise ValueError(f"more {placeholder!r} placeholders than supplied media")
    if not count:
        return ids, 0

    # Most templates encode each marker as its own token sequence.
    starts = [
        i
        for i in range(len(ids) - len(pattern) + 1)
        if ids[i : i + len(pattern)] == pattern
    ]
    if len(starts) == count:
        output, cursor = [], 0
        for start, replacement in zip(starts, replacements):
            output.extend(ids[cursor:start])
            output.extend(replacement)
            cursor = start + len(pattern)
        output.extend(ids[cursor:])
        return output, count

    # BPE may merge a marker with adjacent text. Decode each token only once,
    # then re-encode only boundary fragments; never scan generated patch tokens.
    parts = [tokenizer.decode([token_id]) for token_id in ids]
    text = "".join(parts)
    offsets = [0]
    for part in parts:
        offsets.append(offsets[-1] + len(part))

    def append_text(start, end):
        if start == end:
            return
        first = bisect_right(offsets, start) - 1
        last = bisect_left(offsets, end) - 1
        if first == last:
            if start == offsets[first] and end == offsets[first + 1]:
                output.append(ids[first])
            else:
                output.extend(_encode(tokenizer, text[start:end]))
            return
        if start != offsets[first]:
            output.extend(_encode(tokenizer, text[start : offsets[first + 1]]))
            first += 1
        output.extend(ids[first:last])
        if end == offsets[last + 1]:
            output.append(ids[last])
        else:
            output.extend(_encode(tokenizer, text[offsets[last] : end]))

    output, cursor, consumed = [], 0, 0
    for match in re.finditer(re.escape(placeholder), text):
        if consumed == len(replacements):
            raise ValueError(f"more {placeholder!r} placeholders than supplied media")
        append_text(cursor, match.start())
        output.extend(replacements[consumed])
        consumed += 1
        cursor = match.end()
    append_text(cursor, len(text))
    return output, consumed


def _find_runs(
    input_ids: list[int], token_id: int, expected_lengths: list[list[int]]
) -> list[list[tuple[int, int]]]:
    runs = []
    index = 0
    while index < len(input_ids):
        if input_ids[index] != token_id:
            index += 1
            continue
        end = index + 1
        while end < len(input_ids) and input_ids[end] == token_id:
            end += 1
        runs.append((index, end - 1))
        index = end
    flattened_expected = [length for group in expected_lengths for length in group]
    actual_lengths = [end - start + 1 for start, end in runs]
    if actual_lengths != flattened_expected:
        raise ValueError(
            f"multimodal token runs {actual_lengths} do not match features "
            f"{flattened_expected}"
        )
    grouped = []
    cursor = 0
    for lengths in expected_lengths:
        grouped.append(runs[cursor : cursor + len(lengths)])
        cursor += len(lengths)
    return grouped


def _replace_unexpanded_hf_markers(parts, replacements, boundaries):
    """Normalize HF template markers while preserving already expanded media."""
    result = []
    end = None
    for part in parts:
        if end is not None:
            result.append(part)
            if part == end:
                end = None
        elif part in boundaries:
            end = boundaries[part]
            result.append(part)
        else:
            result.extend(replacements.get(part, [part]))
    return result


@dataclass(frozen=True)
class MuseImagePreprocessArtifact:
    content_digest: str
    artifact_key: str
    feature_hash: int
    feature: torch.Tensor
    token_count: int
    cpu_feature: Optional[torch.Tensor] = None

    @property
    def has_feature(self) -> bool:
        return True

    def cache_value(self):
        if self.feature.device.type == "cpu":
            return self
        # The IO worker prepares this copy. Never synchronize D2H on the
        # tokenizer event loop, or make a miss request upload the copy again.
        assert self.cpu_feature is not None and self.cpu_feature.device.type == "cpu"
        return replace(self, feature=self.cpu_feature, cpu_feature=None)

    def cache_size_items(self):
        return (
            self.content_digest,
            self.artifact_key,
            self.feature_hash,
            self.feature,
            self.token_count,
        )


class MuseSpark1xOmniMultimodalProcessor(
    MediaArtifactCacheMixin, BaseMultimodalProcessor
):
    models = [MuseSparkForConditionalGeneration]
    gpu_image_decode = False
    # Decode on CPU even when optional GPU resizing is enabled. Decoder worker
    # threads otherwise create a context on the default GPU before the explicit
    # base_gpu_id preprocessing context is entered.
    video_preprocessing_device = "cpu"
    artifact_modality = Modality.IMAGE
    auto_mm_preprocess_cache_size_mb = 256

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs) -> None:
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.audio_samples_per_token = math.prod(hf_config.audio_config.patch_sizes)
        vision = hf_config.vision_config
        self.preprocess_config = make_preprocess_config(vision)

        tokenizer = self._tokenizer
        self.image_start_id = _token_id(tokenizer, "<|image_start|>")
        self.image_end_id = _token_id(tokenizer, "<|image_end|>")
        self.patch_id = _token_id(tokenizer, "<|patch|>")
        self.video_start_id = _token_id(tokenizer, "<|vid_start|>")
        self.video_end_id = _token_id(tokenizer, "<|vid_end|>")
        self.video_separator_id = _token_id(tokenizer, "<|vid_frame_separator|>")
        self.video_id = _token_id(tokenizer, "<|video|>")
        self.audio_id = _token_id(tokenizer, "<|audio|>")
        # The published Muse-Spark-1.2-OSS tokenizer keeps <|audio|> but leaves the
        # soft-audio delimiters as reserved placeholders. Treat them as optional so
        # a text/image/video server still starts; audio input fails loudly instead.
        self.audio_start_id = _optional_token_id(tokenizer, "<|soft_audio_start|>")
        self.audio_end_id = _optional_token_id(tokenizer, "<|soft_audio_end|>")
        self.audio_supported = (
            self.audio_start_id is not None and self.audio_end_id is not None
        )
        if not self.audio_supported:
            logger.warning(
                "Tokenizer does not define <|soft_audio_start|>/<|soft_audio_end|>; "
                "audio input is disabled for this checkpoint. In the published "
                "release these two delimiters are still named as reserved "
                "placeholders (ids 201463 and 201464). To enable audio, serve a "
                "copy of the tokenizer with those two entries renamed, passed "
                "via --tokenizer-path."
            )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>",
            video_token="<video>",
            audio_token="<audio>",
        ).build(_processor)
        self.image_placeholder_ids = _encode(self._tokenizer, "<image>")
        self.video_placeholder_ids = _encode(self._tokenizer, "<video>")
        self.audio_placeholder_ids = _encode(self._tokenizer, "<audio>")

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        # Called by the base constructor before preprocess_config exists.
        vision = getattr(self.hf_config, "vision_config", {}) or {}
        payload["muse"] = {
            "defaults": asdict(PreprocessConfig()),
            "vision_config": vision if isinstance(vision, dict) else vars(vision),
        }
        return payload

    async def _run_preprocess_and_build_artifact_batch(self, entries):
        # Match the uncached path: GPU/CPU preprocessing must not block the
        # tokenizer event loop. The shared cache coordinates concurrent misses.
        return await asyncio.get_running_loop().run_in_executor(
            self.io_executor, self.prepare_artifact_batch, entries
        )

    def decode_media_snapshot(self, snapshot, modality):
        # Keep NumPy's HWC layout explicit through the cache path.
        if isinstance(snapshot.data, np.ndarray):
            return snapshot.data
        return super().decode_media_snapshot(snapshot, modality)

    def prepare_artifact_batch(self, entries: list[MediaArtifactInput]):
        context = (
            torch.cuda.device(self.server_args.base_gpu_id)
            if self.preprocess_config.use_gpu_preprocess
            else nullcontext()
        )
        artifacts = []
        with context:
            for entry in entries:
                feature, token_count = preprocess_image(
                    entry.media, self.preprocess_config
                )
                cpu_feature = feature.cpu()
                keep_device = self.keep_mm_features_on_device and feature.is_cuda
                artifacts.append(
                    MuseImagePreprocessArtifact(
                        content_digest=entry.content_digest,
                        artifact_key=entry.artifact_key,
                        feature_hash=resolve_multimodal_item_hash(
                            existing_hash=0, namespace=entry.artifact_key
                        ),
                        feature=feature if keep_device else cpu_feature,
                        token_count=token_count,
                        cpu_feature=cpu_feature if keep_device else None,
                    )
                )
        return artifacts

    async def _process_cached_images(self, image_data, input_text, request_obj):
        self.validate_mm_data(image_data=image_data)
        artifacts = await self.prepare_media_artifacts(
            image_data,
            content_hashes=getattr(request_obj, "mm_content_hashes", None),
        )
        images = [(artifact.feature, artifact.token_count) for artifact in artifacts]
        input_ids = getattr(request_obj, "input_ids", None)
        if input_ids is None and "<|image_start|>" in input_text:
            input_ids = _encode(self._tokenizer, input_text)
        if input_ids is None:
            text = input_text
            for artifact in artifacts:
                text = _replace_once(
                    text,
                    "<image>",
                    _image_replacement(artifact.token_count),
                )
            input_ids = _encode(self._tokenizer, text)
        else:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            input_ids = _replace_unexpanded_hf_markers(
                input_ids,
                {
                    self.patch_id: self.image_placeholder_ids,
                    self.video_id: self.video_placeholder_ids,
                    self.audio_id: self.audio_placeholder_ids,
                },
                {
                    self.image_start_id: self.image_end_id,
                    self.video_start_id: self.video_end_id,
                    self.audio_start_id: self.audio_end_id,
                },
            )
            input_ids = self._expand_mm_placeholders(input_ids, images, [], [])
        output = self._build_output(input_ids, images, [], [])
        for item, artifact in zip(output.mm_items, artifacts):
            item.set_hash(artifact.feature_hash)
        return output

    async def process_mm_data_async(
        self,
        image_data=None,
        audio_data=None,
        input_text="",
        request_obj=None,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        video_data = kwargs.get("video_data")
        if video_data is None and request_obj is not None:
            video_data = getattr(request_obj, "video_data", None)
        if any(
            self._is_preprocessed_input(item)
            for data in (image_data, video_data, audio_data)
            for item in (data or [])
        ):
            raise ValueError(
                "Muse does not yet support processor_output or precomputed_embedding "
                "inputs; provide raw media with text or input_ids instead"
            )
        if isinstance(input_text, str):
            input_text = "".join(
                _replace_unexpanded_hf_markers(
                    re.split(r"(<\|[^|]*\|>)", input_text),
                    {
                        "<|patch|>": ["<image>"],
                        "<|video|>": ["<video>"],
                        "<|audio|>": ["<audio>"],
                    },
                    {
                        "<|image_start|>": "<|image_end|>",
                        "<|vid_start|>": "<|vid_end|>",
                        "<|soft_audio_start|>": "<|soft_audio_end|>",
                    },
                )
            )
        if (
            image_data
            and not video_data
            and not audio_data
            and self.mm_preprocess_cache.enabled
            and not any(self._is_preprocessed_input(item) for item in image_data)
        ):
            return await self._process_cached_images(
                image_data, input_text, request_obj
            )
        pretokenized_ids = (
            getattr(request_obj, "input_ids", None) if request_obj is not None else None
        )
        if (
            pretokenized_ids is None
            and isinstance(input_text, str)
            and any(
                marker in input_text
                for marker in (
                    "<|image_start|>",
                    "<|vid_start|>",
                    "<|soft_audio_start|>",
                )
            )
        ):
            # Load every supplied item; the legacy text loader counts only raw
            # placeholders and would drop media for already expanded spans.
            pretokenized_ids = _encode(self._tokenizer, input_text)
        if pretokenized_ids is not None:
            if not (image_data or video_data or audio_data):
                return None
            return await self._process_pretokenized(
                pretokenized_ids, image_data, video_data, audio_data
            )
        loaded = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
        )

        images, videos, audios = await self._preprocess_media_async(loaded)

        return self._build_text_output(loaded.input_text, images, videos, audios)

    def _build_text_output(self, text, images, videos, audios, *, transport=True):
        for _, token_count in images:
            text = _replace_once(text, "<image>", _image_replacement(token_count))
        for video in videos:
            pieces = ["<|vid_start|>"]
            for index, timestamp in enumerate(video.timestamps):
                if index:
                    pieces.append("<|vid_frame_separator|>")
                pieces.append(self.preprocess_config.format_timestamp(timestamp))
                pieces.append("<|video|>" * video.patches_per_clip)
            pieces.append("<|vid_end|>")
            text = _replace_once(text, "<video>", "".join(pieces))
        if audios and not self.audio_supported:
            raise ValueError(
                "This checkpoint's tokenizer does not define "
                "<|soft_audio_start|>/<|soft_audio_end|>, so audio input is not "
                "supported. Serve a tokenizer copy with those two delimiters "
                "named (--tokenizer-path), or send text, image, or video "
                "instead."
            )
        for _, token_count in audios:
            replacement = (
                "<|soft_audio_start|>"
                + "<|audio|>" * token_count
                + "<|soft_audio_end|>"
            )
            text = _replace_once(text, "<audio>", replacement)

        input_ids = _encode(self._tokenizer, text)
        return self._build_output(
            input_ids, images, videos, audios, transport=transport
        )

    async def _process_pretokenized(
        self, input_ids, image_data, video_data, audio_data
    ) -> Optional[MultimodalProcessorOutput]:
        self.validate_mm_data(
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
        )
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        else:
            input_ids = list(input_ids)
        input_ids = _replace_unexpanded_hf_markers(
            input_ids,
            {
                self.patch_id: self.image_placeholder_ids,
                self.video_id: self.video_placeholder_ids,
                self.audio_id: self.audio_placeholder_ids,
            },
            {
                self.image_start_id: self.image_end_id,
                self.video_start_id: self.video_end_id,
                self.audio_start_id: self.audio_end_id,
            },
        )
        loaded = await self.fast_load_mm_data(
            prompt="",
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            input_ids=input_ids,
        )
        images, videos, audios = await self._preprocess_media_async(loaded)
        if audios and not self.audio_supported:
            raise ValueError(
                "This checkpoint's tokenizer does not define "
                "<|soft_audio_start|>/<|soft_audio_end|>, so audio input is not "
                "supported. Serve a tokenizer copy with those two delimiters "
                "named (--tokenizer-path), or send text, image, or video "
                "instead."
            )
        input_ids = self._expand_mm_placeholders(input_ids, images, videos, audios)
        return self._build_output(input_ids, images, videos, audios)

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """Rebuild Muse media spans without loading or encoding media on P/D."""

        def grids(name):
            value = kwargs.get(name)
            return [] if value is None else value

        images = [
            (None, int(grid[0] * grid[1] * grid[2])) for grid in grids("img_grid_thw")
        ]
        video_grids = grids("video_grid_thw")
        timestamps = kwargs.get("video_timestamps") or []
        if len(timestamps) != len(video_grids):
            raise ValueError("Muse encoder video grids/timestamps do not match")
        videos = []
        for grid, times in zip(video_grids, timestamps):
            if len(times) != int(grid[0]):
                raise ValueError("Muse encoder video clip/timestamp count mismatch")
            videos.append(VideoFeatures(None, int(grid[1] * grid[2]), times))
        audios = [(None, int(count)) for count in grids("audio_feature_lens")]
        if audios and not self.audio_supported:
            raise ValueError("This checkpoint's tokenizer does not support audio input")
        if isinstance(prompt, str):
            text = "".join(
                _replace_unexpanded_hf_markers(
                    re.split(r"(<\|[^|]*\|>)", prompt),
                    {
                        "<|patch|>": ["<image>"],
                        "<|video|>": ["<video>"],
                        "<|audio|>": ["<audio>"],
                    },
                    {
                        "<|image_start|>": "<|image_end|>",
                        "<|vid_start|>": "<|vid_end|>",
                        "<|soft_audio_start|>": "<|soft_audio_end|>",
                    },
                )
            )
            output = self._build_text_output(
                text, images, videos, audios, transport=False
            )
        else:
            ids = prompt.tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
            ids = _replace_unexpanded_hf_markers(
                ids,
                {
                    self.patch_id: self.image_placeholder_ids,
                    self.video_id: self.video_placeholder_ids,
                    self.audio_id: self.audio_placeholder_ids,
                },
                {
                    self.image_start_id: self.image_end_id,
                    self.video_start_id: self.video_end_id,
                    self.audio_start_id: self.audio_end_id,
                },
            )
            ids = self._expand_mm_placeholders(ids, images, videos, audios)
            output = self._build_output(ids, images, videos, audios, transport=False)
        consumed = {}
        for item in output.mm_items:
            start = consumed.get(item.modality, 0)
            count = sum(end - begin + 1 for begin, end in item.offsets)
            item.precomputed_embeddings = embeddings[item.modality][
                start : start + count
            ]
            consumed[item.modality] = start + count
        return output

    def _preprocessing_competes_with_the_scheduler(self) -> bool:
        return False

    async def _preprocess_media_async(self, loaded):
        # Video/image decoding and resizing is CPU-heavy; run it in the IO
        # worker pool so the tokenizer manager's event loop stays responsive
        # to health checks, cancellations and other requests.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.io_executor, self._preprocess_media, loaded
        )

    def _preprocess_media(self, loaded):
        context = (
            torch.cuda.device(self.server_args.base_gpu_id)
            if self.preprocess_config.use_gpu_preprocess
            else nullcontext()
        )
        with context:
            images = [
                preprocess_image(value, self.preprocess_config)
                for value in (loaded.images or [])
            ]
            videos = [
                preprocess_video(value, self.preprocess_config)
                for value in (loaded.videos or [])
            ]
            audios = [
                preprocess_audio(value, self.audio_samples_per_token)
                for value in (loaded.audios or [])
            ]
            return images, videos, audios

    def _expand_modality(self, ids, placeholder, pattern, replacements):
        # Original expanded spans consume media too. Split before inserting any
        # replacements so newly generated spans cannot be counted a second time.
        boundaries = {
            "<image>": (self.image_start_id, self.image_end_id),
            "<video>": (self.video_start_id, self.video_end_id),
            "<audio>": (self.audio_start_id, self.audio_end_id),
        }
        start_id, end_id = boundaries[placeholder]
        replacements = list(replacements)
        if not replacements:
            return ids
        consumed = 0

        def expand_raw(segment):
            nonlocal consumed
            expanded, count = _expand_raw_placeholders(
                tokenizer=self._tokenizer,
                ids=segment,
                placeholder=placeholder,
                pattern=pattern,
                replacements=replacements[consumed:],
            )
            consumed += count
            return expanded

        output = []
        cursor = 0
        while start_id is not None and start_id in ids[cursor:]:
            start = ids.index(start_id, cursor)
            output.extend(expand_raw(ids[cursor:start]))
            try:
                end = ids.index(end_id, start + 1)
            except ValueError:
                raise ValueError(
                    f"unterminated expanded {placeholder!r} span"
                ) from None
            if consumed == len(replacements):
                raise ValueError(f"more {placeholder!r} spans than supplied media")
            output.extend(ids[start : end + 1])
            consumed += 1
            cursor = end + 1
        output.extend(expand_raw(ids[cursor:]))
        if consumed and consumed != len(replacements):
            raise ValueError(f"fewer {placeholder!r} placeholders than supplied media")
        # Legacy bare patch runs are checked by _find_runs downstream.
        return output

    def _expand_mm_placeholders(self, input_ids, images, videos, audios):
        ids = list(input_ids)
        ids = self._expand_modality(
            ids,
            "<image>",
            self.image_placeholder_ids,
            (
                [self.image_start_id]
                + [self.patch_id] * token_count
                + [self.image_end_id]
                for _, token_count in images
            ),
        )
        ids = self._expand_modality(
            ids,
            "<video>",
            self.video_placeholder_ids,
            (self._video_replacement(video) for video in videos),
        )
        ids = self._expand_modality(
            ids,
            "<audio>",
            self.audio_placeholder_ids,
            (
                [self.audio_start_id]
                + [self.audio_id] * token_count
                + [self.audio_end_id]
                for _, token_count in audios
            ),
        )
        return ids

    def _video_replacement(self, video):
        pieces = [self.video_start_id]
        for index, timestamp in enumerate(video.timestamps):
            if index:
                pieces.append(self.video_separator_id)
            pieces.extend(
                _encode(
                    self._tokenizer, self.preprocess_config.format_timestamp(timestamp)
                )
            )
            pieces.extend([self.video_id] * video.patches_per_clip)
        pieces.append(self.video_end_id)
        return pieces

    def _build_output(
        self, input_ids, images, videos, audios, *, transport=True
    ) -> MultimodalProcessorOutput:
        image_offsets = _find_runs(
            input_ids, self.patch_id, [[count] for _, count in images]
        )
        video_offsets = _find_runs(
            input_ids,
            self.video_id,
            [[video.patches_per_clip] * len(video.timestamps) for video in videos],
        )
        audio_offsets = _find_runs(
            input_ids, self.audio_id, [[count] for _, count in audios]
        )

        items = []
        for (feature, _), offsets in zip(images, image_offsets):
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE, feature=feature, offsets=offsets
                )
            )
        for video, offsets in zip(videos, video_offsets):
            items.append(
                MultimodalDataItem(
                    modality=Modality.VIDEO, feature=video.tensor, offsets=offsets
                )
            )
        for (feature, _), offsets in zip(audios, audio_offsets):
            items.append(
                MultimodalDataItem(
                    modality=Modality.AUDIO, feature=feature, offsets=offsets
                )
            )

        if transport and self.preprocess_config.use_gpu_preprocess:
            if not self.keep_mm_features_on_device:
                for item in items:
                    item.feature = self._move_feature_to_cpu(item.feature)
            items = self._prepare_mm_items_for_transport(items)

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=items,
            im_token_id=self.patch_id,
            im_start_id=self.image_start_id,
            im_end_id=self.image_end_id,
            video_token_id=self.video_id,
            audio_token_id=self.audio_id,
            audio_start_id=self.audio_start_id,
            audio_end_id=self.audio_end_id,
        )


__all__ = [
    "MuseSpark1xOmniMultimodalProcessor",
    "PreprocessConfig",
    "find_resize_target",
    "preprocess_audio",
    "preprocess_image",
    "preprocess_video",
]
