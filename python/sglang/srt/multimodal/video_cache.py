"""Caller-identified, prompt-independent CPU video artifacts for Qwen2.5-VL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from typing import Any

import msgspec
import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.cache import (
    build_artifact_key,
    resolve_multimodal_item_hash,
)
from sglang.srt.runtime_context import get_disagg, get_mm
from sglang.srt.utils import VideoData, load_video


class QwenVideoArtifact(msgspec.Struct, frozen=True):
    feature: torch.Tensor
    grid: torch.Tensor
    second_per_grid: float
    feature_hash: int

    def cache_size_items(self):
        return self.feature, self.grid, self.second_per_grid, self.feature_hash


def _cache_id(source: Any):
    if isinstance(source, dict):
        return source.get("cache_id")
    return getattr(source, "cache_id", None)


def has_video_cache_ids(videos) -> bool:
    if videos is None:
        return False
    if not isinstance(videos, list):
        videos = [videos]
    return any(
        has_video_cache_ids(source)
        if isinstance(source, list)
        else _cache_id(source) is not None
        for source in videos
    )


def validate_video_cache_request(
    request, *, processor, enabled: bool, language_only: bool = False
) -> None:
    """Fail before media I/O when an opted-in request cannot honor its IDs."""
    videos = request.video_data
    if not has_video_cache_ids(videos):
        return
    if not isinstance(videos, list):
        videos = [videos]
    if not enabled:
        raise ValueError("video cache_id requires --trust-mm-cache-ids")
    if (
        getattr(processor, "model_type", None) != "qwen2_5_vl"
        or not getattr(processor, "supports_video_cache_ids", False)
        or request.image_data
        or request.audio_data
        or language_only
    ):
        raise ValueError(
            "video cache_id currently supports Qwen2.5-VL video-only requests "
            "with a local multimodal processor"
        )
    namespace = getattr(request, "cache_salt", None)
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("video cache_id requires a non-empty tenant cache_salt")
    if not processor.mm_preprocess_cache.enabled:
        raise ValueError("video cache_id requires --mm-preprocess-cache-size-mb > 0")
    if getattr(request, "mm_hashes", None):
        raise ValueError("video cache_id cannot be combined with mm_hashes")
    for source in videos:
        identity = _cache_id(source)
        if identity is not None and (
            not isinstance(identity, str) or not identity.strip() or len(identity) > 256
        ):
            raise ValueError(
                "video cache_id must be a non-empty string of at most 256 characters"
            )
        if isinstance(source, dict):
            if set(source) - {"url", "cache_id", "preprocess_kwargs"}:
                raise ValueError("Unsupported video cache input fields")
            if not isinstance(source.get("url"), str):
                raise ValueError("video cache inputs require a string url")
        elif not isinstance(source, (str, bytes, VideoData)):
            raise ValueError("video cache requests require raw video sources")
        elif isinstance(source, VideoData) and not isinstance(source.url, (str, bytes)):
            raise ValueError("video cache inputs require a string url or video bytes")


def _video_source(source) -> VideoData:
    if isinstance(source, VideoData):
        return source
    if isinstance(source, dict):
        return VideoData(**source)
    return VideoData(url=source)


def _video_config(processor, source: VideoData) -> dict:
    from sglang.srt.multimodal.processors.qwen_vl import (
        QWEN_VIDEO_PREPROCESS_CONFIG_KEYS,
    )

    options = source.preprocess_kwargs
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("video preprocess_kwargs must be an object")
    unsupported = set(options) - QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
    if unsupported:
        raise ValueError(
            f"Unsupported Qwen video preprocessing options: {sorted(unsupported)}"
        )
    config = {**processor.video_config, **options}
    # An explicit sampling mode overrides the server's other sampling mode.
    if "fps" in options and "nframes" not in options:
        config.pop("nframes", None)
    elif "nframes" in options and "fps" not in options:
        config.pop("fps", None)
    if "fps" in config and "nframes" in config:
        raise ValueError("Specify only one of video fps and nframes")
    for name in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS & config.keys():
        value = config[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or value <= 0
        ):
            raise ValueError(f"video {name} must be a positive finite number")
        # Chat's schema turns FPS into a float, while native/server JSON may
        # contain an integer. Equal sampling settings should share an entry.
        if isinstance(value, float) and value.is_integer():
            config[name] = int(value)
    return config


def video_artifact_key(*, cache_id, cache_salt, processor_fingerprint, video_config):
    # This is an opaque caller identity, not an asserted digest of media bytes.
    # Keep its domain separate from content-verified image artifacts.
    identity = json.dumps(
        ["qwen-video-caller-id-v1", cache_salt, cache_id],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return build_artifact_key(
        "sha256:" + hashlib.sha256(identity).hexdigest(),
        modality="video",
        processor_fingerprint=processor_fingerprint,
        preprocess_kwargs=video_config,
    )


async def _prepare_video(processor, *, source, config, key):
    from sglang.srt.multimodal.processors.qwen_vl import (
        _get_processor_video_config,
        preprocess_video,
    )

    future = asyncio.wrap_future(
        processor.io_executor.submit(load_video, source, use_gpu=False)
    )
    try:
        decoder = await asyncio.shield(future)
    except asyncio.CancelledError:
        # An untagged video in a mixed request has no cache-owned task. Its I/O
        # thread may still finish after the request is cancelled.
        future.add_done_callback(_close_cancelled_video)
        raise
    try:
        video, metadata = await preprocess_video(decoder, video_config=config)
    finally:
        decoder.close()
    processor_config = _get_processor_video_config(config, [metadata])
    # Match the existing Qwen2.5-VL processor call. Its temporal patch metadata
    # is retained with the artifact; prompt offsets and M-RoPE are not cached.
    kwargs = dict(
        input_text="<|vision_start|><|video_pad|><|vision_end|>",
        videos=[video],
        processor_video_config=processor_config,
    )
    if processor.mm_processor_executor is None:
        result = processor.process_mm_data(**kwargs)
    else:
        result = await processor.mm_processor_executor.run(
            processor.process_mm_data, **kwargs
        )
    feature = result["pixel_values_videos"].detach().cpu().contiguous()
    grid = torch.as_tensor(result["video_grid_thw"], dtype=torch.long).cpu()
    if grid.shape != (1, 3) or int(grid.prod()) != feature.shape[0]:
        raise ValueError("Qwen video processor returned inconsistent features and grid")
    seconds = result.get("second_per_grid_ts", result.get("video_second_per_grid"))
    if seconds is None or len(seconds) != 1:
        raise ValueError("Qwen2.5-VL video processor must return one temporal interval")
    feature_hash = (
        resolve_multimodal_item_hash(existing_hash=0, namespace=key)
        if key is not None
        else resolve_multimodal_item_hash(feature=feature)
    )
    return QwenVideoArtifact(feature, grid, float(seconds[0]), feature_hash)


def _close_cancelled_video(future):
    if not future.cancelled() and future.exception() is None:
        future.result().close()


def _video_prompt_ids(processor, input_text, num_videos):
    # The original raw-video path decodes tokenized prompts before the HF
    # processor tokenizes them again. Preserve that behavior on cache hits.
    tokenizer = processor._tokenizer
    if isinstance(input_text, list):
        input_text = tokenizer.decode(input_text)
    if not isinstance(input_text, str):
        raise ValueError("video cache requests require a text or token-ID prompt")
    kwargs = {}
    bos = getattr(tokenizer, "bos_token", None)
    if processor._tokenizer_auto_adds_specials and bos and input_text.startswith(bos):
        kwargs["add_special_tokens"] = False
    ids = tokenizer.encode(input_text, **kwargs)
    video_id = processor.mm_tokens.video_token_id
    positions = [index for index, token in enumerate(ids) if token == video_id]
    if (
        len(positions) != num_videos
        or processor.mm_tokens.image_token_id in ids
        or any(
            index == 0
            or index + 1 == len(ids)
            or ids[index - 1] != processor.vision_start_token_id
            or ids[index + 1] != processor.vision_end_token_id
            for index in positions
        )
    ):
        raise ValueError("Video inputs must match unexpanded Qwen video placeholders")
    return ids


async def process_cached_qwen_video(processor, *, input_text, request):
    validate_video_cache_request(
        request,
        processor=processor,
        enabled=get_mm().trust_mm_cache_ids,
        language_only=get_disagg().language_only or get_disagg().encoder_only,
    )
    # Resolve all settings before starting I/O, including sources without IDs.
    videos = request.video_data
    if not isinstance(videos, list):
        videos = [videos]
    sources = [_video_source(source) for source in videos]
    configs = [_video_config(processor, source) for source in sources]
    prompt_ids = _video_prompt_ids(processor, input_text, len(sources))
    artifacts = []
    for source, config in zip(sources, configs):
        key = (
            video_artifact_key(
                cache_id=source.cache_id,
                cache_salt=request.cache_salt,
                processor_fingerprint=processor.processor_fingerprint,
                video_config=config,
            )
            if source.cache_id is not None
            else None
        )

        # Bind the inputs: cache-owned work can outlive cancellation of this
        # request and must never read another loop iteration's source/settings.
        async def compute(source=source, config=config, key=key):
            return await _prepare_video(
                processor, source=source, config=config, key=key
            )

        if key is None:
            artifact = await compute()
        else:
            result = await processor.mm_preprocess_cache.get_or_compute(key, compute)
            artifact = result.value
        artifacts.append(artifact)
    return _compose_video_artifacts(
        processor, input_text=prompt_ids, artifacts=artifacts
    )


def _compose_video_artifacts(processor, *, input_text, artifacts):
    grid = torch.cat([artifact.grid for artifact in artifacts], dim=0)
    input_ids, offsets, modalities = processor.build_input_ids(
        input_text, video_grid_thw=grid
    )
    if modalities != [Modality.VIDEO] * len(artifacts):
        raise ValueError("Video artifacts do not match the prompt placeholders")
    items = []
    for artifact, offset in zip(artifacts, offsets):
        item = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=artifact.feature.clone(),
            offsets=[offset],
            model_specific_data={"video_grid_thw": artifact.grid.clone()},
        )
        item.set_hash(artifact.feature_hash)
        items.append(item)
    positions, delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=processor.spatial_merge_size,
        image_token_id=processor.mm_tokens.image_token_id,
        video_token_id=processor.mm_tokens.video_token_id,
        vision_start_token_id=processor.vision_start_token_id,
        model_type=processor.model_type,
        tokens_per_second=processor._tokens_per_second,
        input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        video_grid_thw=grid,
        second_per_grid_ts=[artifact.second_per_grid for artifact in artifacts],
    )
    return MultimodalProcessorOutput(
        input_ids=input_ids,
        padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
            input_ids, items
        ),
        mm_items=processor._prepare_mm_items_for_transport(items),
        im_start_id=processor.vision_start_token_id,
        im_end_id=processor.vision_end_token_id,
        im_token_id=processor.mm_tokens.image_token_id,
        video_token_id=processor.mm_tokens.video_token_id,
        audio_token_id=processor.mm_tokens.audio_token_id,
        mrope_positions=positions.squeeze(1),
        mrope_position_delta=delta,
    )
