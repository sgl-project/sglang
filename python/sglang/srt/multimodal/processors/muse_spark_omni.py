"""Native image, video, and waveform preprocessing for Muse 1.2 Omni."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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
from sglang.srt.multimodal.media_artifacts.base import (
    MediaArtifactCacheMixin,
    MediaArtifactInput,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_model
from sglang.srt.utils.hf_transformers.common import resolve_runai_obj_uri

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


def _image_to_uint8(image: Any, *, keep_device: bool = False) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1)
    if isinstance(image, np.ndarray):
        value = torch.from_numpy(np.array(image, copy=True))
    else:
        value = torch.as_tensor(image)
        if not keep_device:
            value = value.cpu()
    if value.ndim != 3:
        raise ValueError(f"unexpected image shape {tuple(value.shape)}")
    if value.shape[0] in (1, 3, 4):
        value = value[:3]
    else:
        value = value[..., :3].permute(2, 0, 1)
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
        source_indices = list(selected)
        while len(frames) < factor:
            frames.append(frames[-1])
            source_indices.append(source_indices[-1])
        if not frames:
            raise ValueError("video input has no frames")
        return frames, [float(index) / fps for index in source_indices]
    if not frames:
        raise ValueError("video input has no frames")
    if not math.isfinite(fps) or fps <= 0:
        logger.warning("Invalid video FPS %r; falling back to 1.0", fps)
        fps = 1.0

    factor = max(config.frame_factor, config.patch_temporal)
    if config.sampling_fps and fps > 0:
        requested = int(len(frames) * config.sampling_fps / fps)
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


def _contains_sequence(input_ids: list[int], pattern: list[int]) -> bool:
    end = len(input_ids) - len(pattern) + 1
    return any(input_ids[i : i + len(pattern)] == pattern for i in range(max(end, 0)))


def _replace_sequence_once(
    input_ids: list[int], pattern: list[int], replacement: list[int], name: str
) -> list[int]:
    end = len(input_ids) - len(pattern) + 1
    for start in range(max(end, 0)):
        if input_ids[start : start + len(pattern)] == pattern:
            return input_ids[:start] + replacement + input_ids[start + len(pattern) :]
    raise ValueError(f"missing {name!r} placeholder token sequence for supplied media")


def _splice_placeholder_once(
    tokenizer, input_ids: list[int], placeholder: str, replacement: list[int]
) -> Optional[list[int]]:
    # BPE may merge a placeholder with adjacent text (e.g. " <image>\n" becomes
    # [" <", "image", ">\n"]). Decode token-wise (lossless), locate the
    # placeholder's character span, split the boundary tokens, and re-encode
    # the residual characters so every surrounding character is preserved.
    parts = [tokenizer.decode([token_id]) for token_id in input_ids]
    start_char = "".join(parts).find(placeholder)
    if start_char < 0:
        return None
    end_char = start_char + len(placeholder)
    offsets = [0]
    for piece in parts:
        offsets.append(offsets[-1] + len(piece))
    first = next(i for i in range(len(parts)) if offsets[i + 1] > start_char)
    last = next(i for i in range(len(parts)) if offsets[i + 1] >= end_char)
    residual_pre = parts[first][: start_char - offsets[first]]
    residual_post = parts[last][end_char - offsets[last] :]
    return (
        input_ids[:first]
        + (_encode(tokenizer, residual_pre) if residual_pre else [])
        + replacement
        + (_encode(tokenizer, residual_post) if residual_post else [])
        + input_ids[last + 1 :]
    )


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

    @property
    def has_feature(self) -> bool:
        return True

    def cache_value(self):
        # Preprocessing may run on GPU, but retained features belong to CPU.
        assert self.feature.device.type == "cpu"
        return self

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
    artifact_modality = Modality.IMAGE
    auto_mm_preprocess_cache_size_mb = 256

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs) -> None:
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.audio_samples_per_token = math.prod(hf_config.audio_config.patch_sizes)
        vision = hf_config.vision_config
        self.preprocess_config = PreprocessConfig(
            use_rust_resize=not self.disable_fast_image_processor,
            patch_size=vision.patch_size,
            patch_temporal=vision.patch_temporal,
            downsample_factor=vision.merge_size,
            **_load_video_processor_config(
                model_path=get_model().model_path, revision=get_model().revision
            ),
        )

        tokenizer = self._tokenizer
        self.image_start_id = _token_id(tokenizer, "<|image_start|>")
        self.image_end_id = _token_id(tokenizer, "<|image_end|>")
        self.patch_id = _token_id(tokenizer, "<|patch|>")
        self.video_start_id = _token_id(tokenizer, "<|vid_start|>")
        self.video_end_id = _token_id(tokenizer, "<|vid_end|>")
        self.video_separator_id = _token_id(tokenizer, "<|vid_frame_separator|>")
        self.video_id = _token_id(tokenizer, "<|video|>")
        self.audio_id = _token_id(tokenizer, "<|audio|>")
        # The published Muse Spark Open tokenizer keeps <|audio|> but leaves the
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
                artifacts.append(
                    MuseImagePreprocessArtifact(
                        content_digest=entry.content_digest,
                        artifact_key=entry.artifact_key,
                        feature_hash=resolve_multimodal_item_hash(
                            existing_hash=0, namespace=entry.artifact_key
                        ),
                        feature=feature.cpu(),
                        token_count=token_count,
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

        text = loaded.input_text
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
        return self._build_output(input_ids, images, videos, audios)

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
        # Frontends that ship the tokenized template keep every surrounding id
        # verbatim; only the placeholder spans are rewritten here. A modality
        # whose placeholder is absent entirely is assumed to arrive
        # pre-expanded and is validated by _find_runs downstream.
        for index, replacement in enumerate(replacements):
            if _contains_sequence(ids, pattern):
                ids = _replace_sequence_once(ids, pattern, replacement, placeholder)
                continue
            spliced = _splice_placeholder_once(
                self._tokenizer, ids, placeholder, replacement
            )
            if spliced is None:
                if index == 0:
                    return ids
                raise ValueError(
                    f"fewer {placeholder!r} placeholders in the tokenized prompt "
                    "than supplied media"
                )
            ids = spliced
        return ids

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
        self, input_ids, images, videos, audios
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

        if self.preprocess_config.use_gpu_preprocess:
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
