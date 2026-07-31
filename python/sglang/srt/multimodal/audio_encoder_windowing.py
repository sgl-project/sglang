"""Prepare fixed audio windows for multimodal encoder reuse.

Splits audio into encoder-aligned complete windows plus a mutable tail. A
complete window's mel features are deterministic, so its cache identity is
stable across rolling requests and its encoder output is reused through the
existing multimodal embedding cache; extraction itself is recomputed cheaply.
"""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import torch

from sglang.srt.managers.mm_utils import hash_feature, hash_mm_item
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


class AudioEncoderWindowConfig(msgspec.Struct, frozen=True):
    """Window geometry in samples and encoder tokens, resolved once from the
    model config so every request splits audio identically."""

    min_input_samples: int
    window_samples: int
    window_tokens: int


class _AudioWindowFeatures(msgspec.Struct, frozen=True):
    feature: torch.Tensor
    attention_mask: torch.Tensor
    token_count: int


def resolve_audio_encoder_window_config(
    window_frames: int,
    alignment_frames: int,
    *,
    processor: Any,
    model_sample_rate: int,
) -> AudioEncoderWindowConfig:
    """Validate encoder window geometry against the feature extractor."""
    if window_frames % alignment_frames:
        raise ValueError("audio window is not encoder-block aligned")
    feature_extractor = processor.feature_extractor
    if int(feature_extractor.sampling_rate) != model_sample_rate:
        raise ValueError("audio windowing and realtime model sample rates differ")

    return AudioEncoderWindowConfig(
        min_input_samples=int(feature_extractor.n_fft),
        window_samples=window_frames * int(feature_extractor.hop_length),
        window_tokens=int(
            processor._get_feat_extract_output_lengths([window_frames])[0]
        ),
    )


def _extract_audio_window_features(
    samples: Any,
    config: AudioEncoderWindowConfig,
    *,
    processor: Any,
) -> tuple[list[_AudioWindowFeatures], int]:
    """Extract one feature row per complete window plus a mutable tail."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("audio windowing requires mono audio")

    complete_count, tail_samples = divmod(samples.size, config.window_samples)
    sample_counts = [config.window_samples] * complete_count
    if tail_samples:
        sample_counts.append(tail_samples)
    if not sample_counts:
        raise ValueError("audio windowing requires non-empty audio")

    windows = []
    offset = 0
    for sample_count in sample_counts:
        window = samples[offset : offset + sample_count]
        if sample_count < config.min_input_samples:
            # Whisper-style feature extractors reject inputs shorter than n_fft.
            window = np.pad(window, (0, config.min_input_samples - sample_count))
        windows.append(window)
        offset += sample_count

    feature_extractor = processor.feature_extractor
    audio_inputs = feature_extractor(
        windows,
        sampling_rate=feature_extractor.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=False,
        padding="longest",
    )
    processed_features = audio_inputs["input_features"]
    processed_masks = audio_inputs["attention_mask"]
    token_counts = [
        int(count)
        for count in processor._get_feat_extract_output_lengths(
            processed_masks.sum(dim=-1)
        )
    ]

    if token_counts[:complete_count] != [config.window_tokens] * complete_count:
        raise ValueError("audio window token geometry changed")
    if any(count <= 0 for count in token_counts):
        raise ValueError("audio window produced no encoder tokens")

    window_features = [
        _AudioWindowFeatures(
            feature=processed_features[row : row + 1].clone(),
            attention_mask=processed_masks[row : row + 1].clone(),
            token_count=token_counts[row],
        )
        for row in range(len(windows))
    ]
    return window_features, complete_count


def _build_audio_window_items(
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    window_features: list[_AudioWindowFeatures],
    complete_window_count: int,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Expand one audio placeholder and create feature-backed window items."""
    input_ids = input_ids.flatten()
    token_counts = [window.token_count for window in window_features]

    placeholder_positions = (input_ids == placeholder_token_id).nonzero(as_tuple=True)[
        0
    ]
    if placeholder_positions.numel() != 1:
        raise ValueError("audio windowing requires one audio placeholder")
    placeholder_position = int(placeholder_positions[0])
    expanded_ids = torch.cat(
        [
            input_ids[:placeholder_position],
            torch.full(
                (sum(token_counts),),
                placeholder_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
            input_ids[placeholder_position + 1 :],
        ]
    )

    mm_items = []
    offset_start = placeholder_position
    for index, window in enumerate(window_features):
        offset_end = offset_start + window.token_count - 1
        # The mutable tail changes every request; only complete windows get a
        # stable identity in the embedding cache.
        use_embedding_cache = index < complete_window_count
        item_hash = None
        if use_embedding_cache:
            item_hash = hash_mm_item(
                hash_feature(window.feature),
                Modality.AUDIO,
                [(offset_start, offset_end)],
            )
        mm_items.append(
            MultimodalDataItem(
                modality=Modality.AUDIO,
                hash=item_hash,
                offsets=[(offset_start, offset_end)],
                feature=window.feature,
                use_embedding_cache=use_embedding_cache,
                model_specific_data={
                    "feature_attention_mask": window.attention_mask,
                },
            )
        )
        offset_start = offset_end + 1

    return mm_items, expanded_ids


def build_audio_encoder_window_items(
    samples: Any,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    config: AudioEncoderWindowConfig,
    *,
    processor: Any,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Convert one audio input into reusable complete windows and a mutable tail."""
    window_features, complete_count = _extract_audio_window_features(
        samples,
        config,
        processor=processor,
    )
    return _build_audio_window_items(
        input_ids, placeholder_token_id, window_features, complete_count
    )
