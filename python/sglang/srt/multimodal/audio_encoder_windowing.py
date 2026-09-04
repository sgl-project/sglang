"""Prepare fixed audio windows for multimodal encoder reuse.

Splits audio into encoder-aligned complete windows plus a mutable tail. A
complete window's mel features are deterministic, so its cache identity is
stable across rolling requests and its encoder output is reused through the
existing multimodal embedding cache; extraction itself is recomputed cheaply.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import msgspec
import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

_FEATURE_OUTPUT_KEY = "input_features"
_ATTENTION_MASK_OUTPUT_KEY = "attention_mask"
_ITEM_ATTENTION_MASK_KEY = "feature_attention_mask"


class AudioEncoderWindowConfig(msgspec.Struct, frozen=True):
    """Window geometry in samples, feature frames, and encoder tokens, resolved
    once from the model config so every request splits audio identically."""

    min_input_samples: int
    window_samples: int
    feature_batch_frames: int
    window_tokens: int


class AudioEncoderWindowSpec(msgspec.Struct, frozen=True):
    """Model-declared geometry for independently encodable audio windows."""

    window_frames: int
    alignment_frames: int


class _AudioWindowFeatures(msgspec.Struct, frozen=True):
    feature: torch.Tensor
    attention_mask: torch.Tensor
    token_count: int


def _extract_feature_batch(
    windows: list[np.ndarray],
    config: AudioEncoderWindowConfig,
    *,
    extract_features_fn: Callable[[list[np.ndarray]], Any],
    output_length_fn: Callable[[Any], Any],
    complete: bool,
) -> list[_AudioWindowFeatures]:
    if not windows:
        return []

    audio_inputs = extract_features_fn(windows)
    processed_features = audio_inputs[_FEATURE_OUTPUT_KEY]
    processed_masks = audio_inputs[_ATTENTION_MASK_OUTPUT_KEY]
    token_counts = [
        int(count) for count in output_length_fn(processed_masks.sum(dim=-1))
    ]

    if complete and token_counts != [config.window_tokens] * len(windows):
        raise ValueError("audio window token geometry changed")
    if any(count <= 0 for count in token_counts):
        raise ValueError("audio window produced no encoder tokens")

    feature_frames = processed_features.shape[-1]
    if feature_frames > config.feature_batch_frames:
        raise ValueError("audio window feature width exceeds the configured maximum")

    padding = config.feature_batch_frames - feature_frames
    features = []
    for row in range(len(windows)):
        feature = processed_features[row : row + 1].clone()
        attention_mask = processed_masks[row : row + 1].clone()
        if padding:
            feature = torch.nn.functional.pad(feature, (0, padding))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding))
        features.append(
            _AudioWindowFeatures(
                feature=feature,
                attention_mask=attention_mask,
                token_count=token_counts[row],
            )
        )
    return features


def resolve_audio_encoder_window_config(
    spec: AudioEncoderWindowSpec,
    *,
    feature_extractor: Any,
    output_length_fn: Callable[[Any], Any],
    model_sample_rate: int,
) -> AudioEncoderWindowConfig:
    """Validate encoder window geometry against the feature extractor."""
    if spec.window_frames <= 0 or spec.alignment_frames <= 0:
        raise ValueError("audio window geometry must be positive")
    if spec.window_frames % spec.alignment_frames:
        raise ValueError("audio window is not encoder-block aligned")
    if int(feature_extractor.sampling_rate) != model_sample_rate:
        raise ValueError("audio windowing and realtime model sample rates differ")
    # Older Transformers releases do not expose dither; their extractor is
    # deterministic, which is equivalent to the current default of zero.
    if float(getattr(feature_extractor, "dither", 0.0)) != 0.0:
        raise ValueError("audio windowing requires deterministic feature extraction")

    min_input_samples = int(feature_extractor.n_fft)
    hop_length = int(feature_extractor.hop_length)
    window_samples = spec.window_frames * hop_length
    return AudioEncoderWindowConfig(
        min_input_samples=min_input_samples,
        window_samples=window_samples,
        # A sub-n_fft tail is attached to the preceding complete window. Pad
        # every feature row to this maximum width so main's normal MM batch can
        # concatenate complete windows and the mutable tail in one encoder call.
        feature_batch_frames=(window_samples + min_input_samples - 1) // hop_length,
        window_tokens=int(output_length_fn([spec.window_frames])[0]),
    )


def _extract_audio_window_features(
    samples: Any,
    config: AudioEncoderWindowConfig,
    *,
    extract_features_fn: Callable[[list[np.ndarray]], Any],
    output_length_fn: Callable[[Any], Any],
) -> tuple[list[_AudioWindowFeatures], int]:
    """Extract one feature row per complete window plus a mutable tail."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("audio windowing requires mono audio")

    complete_count, tail_samples = divmod(samples.size, config.window_samples)
    if tail_samples and tail_samples < config.min_input_samples:
        if complete_count == 0:
            # With no preceding window to absorb it, padding the sub-n_fft
            # input would make artificial samples look valid in the attention
            # mask. Unreachable from the realtime path (windowing activates
            # far past n_fft of audio); refuse rather than fabricate audio.
            raise ValueError("audio windowing requires at least n_fft samples")
        # Keep the sub-n_fft tail attached to the preceding window. Padding it
        # as a standalone item would make artificial samples look valid in the
        # attention mask; the merged item remains uncached and is recomputed.
        complete_count -= 1

    complete_windows = [
        samples[index * config.window_samples : (index + 1) * config.window_samples]
        for index in range(complete_count)
    ]
    # Extract the mutable tail separately: Whisper pads waveforms before STFT,
    # so batching it here could change a complete window's boundary mel frame.
    window_features = _extract_feature_batch(
        complete_windows,
        config,
        extract_features_fn=extract_features_fn,
        output_length_fn=output_length_fn,
        complete=True,
    )

    tail_start = complete_count * config.window_samples
    if tail_start < samples.size:
        tail = samples[tail_start:]
        if tail.size < config.min_input_samples:
            # Whisper-style feature extractors reject inputs shorter than n_fft.
            tail = np.pad(tail, (0, config.min_input_samples - tail.size))
        window_features.extend(
            _extract_feature_batch(
                [tail],
                config,
                extract_features_fn=extract_features_fn,
                output_length_fn=output_length_fn,
                complete=False,
            )
        )
    if not window_features:
        raise ValueError("audio windowing requires non-empty audio")
    return window_features, complete_count


def _build_audio_window_items(
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    window_features: list[_AudioWindowFeatures],
    complete_window_count: int,
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Expand one audio placeholder and create feature-backed window items."""
    # Keep hashing local to avoid an import cycle through schedule_batch.
    from sglang.srt.managers.mm_utils import hash_feature

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
        # The encoder reads both tensors. Including the mask prevents padded
        # tails from sharing a Radix/cache identity with a complete window that
        # has identical zero-padded features but a different effective length.
        item_hash = hash_feature([window.feature, window.attention_mask])
        # The mutable tail changes every request and must not evict reusable
        # complete-window embeddings, even though its Radix identity is valid.
        use_embedding_cache = index < complete_window_count
        item = MultimodalDataItem(
            modality=Modality.AUDIO,
            offsets=[(offset_start, offset_end)],
            feature=window.feature,
            use_embedding_cache=use_embedding_cache,
            separate_encoder_batch=True,
            model_specific_data={
                _ITEM_ATTENTION_MASK_KEY: window.attention_mask,
            },
        )
        item.set_hash(item_hash)
        mm_items.append(item)
        offset_start = offset_end + 1

    return mm_items, expanded_ids


def build_audio_encoder_window_items(
    samples: Any,
    input_ids: torch.Tensor,
    placeholder_token_id: int,
    config: AudioEncoderWindowConfig,
    *,
    extract_features_fn: Callable[[list[np.ndarray]], Any],
    output_length_fn: Callable[[Any], Any],
) -> tuple[list[MultimodalDataItem], torch.Tensor]:
    """Convert one audio input into reusable complete windows and a mutable tail."""
    window_features, complete_count = _extract_audio_window_features(
        samples,
        config,
        extract_features_fn=extract_features_fn,
        output_length_fn=output_length_fn,
    )
    return _build_audio_window_items(
        input_ids,
        placeholder_token_id,
        window_features,
        complete_count,
    )
