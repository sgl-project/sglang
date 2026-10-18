"""Limits shared by the multimodal audio decoders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


class AudioLimitExceeded(ValueError):
    """Raised when decoded audio exceeds the configured resource budget."""


@dataclass(frozen=True)
class AudioDecodeLimits:
    """Limits applied before decoded audio is retained by a caller."""

    max_duration_seconds: Optional[float] = None
    max_decoded_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_duration_seconds is not None and (
            not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("max_audio_duration_seconds must be positive")
        if self.max_decoded_bytes is not None and self.max_decoded_bytes < 0:
            raise ValueError("max_audio_decoded_bytes must be non-negative")

    @property
    def enabled(self) -> bool:
        return (
            self.max_duration_seconds is not None
            or self.max_decoded_bytes not in (None, 0)
        )

    def check_duration(self, duration_seconds: Optional[float]) -> None:
        if (
            duration_seconds is not None
            and self.max_duration_seconds is not None
            and duration_seconds > self.max_duration_seconds
        ):
            raise AudioLimitExceeded(
                "Decoded audio duration exceeds the configured "
                f"limit of {self.max_duration_seconds:g} seconds"
            )

    def check_samples(
        self,
        sample_count: int,
        sample_rate: int,
        channels: int,
        bytes_per_sample: int,
        *,
        kind: str = "decoded audio",
    ) -> None:
        if sample_count < 0 or sample_rate <= 0 or channels <= 0:
            raise ValueError("Invalid audio sample metadata")
        if self.max_duration_seconds is not None:
            duration = sample_count / sample_rate
            if duration > self.max_duration_seconds:
                raise AudioLimitExceeded(
                    f"{kind} duration exceeds the configured limit of "
                    f"{self.max_duration_seconds:g} seconds"
                )
        if self.max_decoded_bytes not in (None, 0):
            decoded_bytes = sample_count * channels * bytes_per_sample
            if decoded_bytes > self.max_decoded_bytes:
                raise AudioLimitExceeded(
                    f"{kind} exceeds the configured decoded-audio budget of "
                    f"{self.max_decoded_bytes} bytes"
                )


def configured_audio_decode_limits() -> AudioDecodeLimits:
    """Read audio limits from the published server configuration when present."""
    try:
        from sglang.srt.runtime_context import get_mm

        mm = get_mm()
    except (ImportError, AttributeError, ValueError):
        return AudioDecodeLimits()
    return AudioDecodeLimits(
        max_duration_seconds=getattr(mm, "max_audio_duration_seconds", None),
        max_decoded_bytes=getattr(mm, "max_audio_decoded_bytes", None),
    )


def resolve_audio_decode_limits(
    max_duration_seconds: Optional[float],
    max_decoded_bytes: Optional[int],
) -> AudioDecodeLimits:
    """Resolve explicit decoder limits, falling back to server configuration."""
    configured = configured_audio_decode_limits()
    return AudioDecodeLimits(
        max_duration_seconds=(
            configured.max_duration_seconds
            if max_duration_seconds is None
            else max_duration_seconds
        ),
        max_decoded_bytes=(
            configured.max_decoded_bytes
            if max_decoded_bytes is None
            else max_decoded_bytes
        ),
    )


def decode_torchcodec_samples(
    decoder,
    *,
    output_sample_rate: Optional[int] = None,
    output_channels: Optional[int] = None,
    limits: Optional[AudioDecodeLimits] = None,
):
    """Decode a bounded range and reject audio that exceeds the budget."""
    limits = limits or configured_audio_decode_limits()
    metadata = decoder.metadata
    limits.check_duration(getattr(metadata, "duration_seconds", None))
    limits.check_duration(getattr(metadata, "duration_seconds_from_header", None))

    if limits.enabled:
        sample_rate = output_sample_rate or getattr(metadata, "sample_rate", None)
        channels = output_channels or getattr(metadata, "num_channels", None)
        if not sample_rate or not channels:
            raise ValueError("Could not determine decoded audio sample rate or channels")
        stop_seconds = limits.max_duration_seconds
        if limits.max_decoded_bytes not in (None, 0):
            byte_limited_duration = limits.max_decoded_bytes / (
                sample_rate * channels * 4
            )
            stop_seconds = (
                byte_limited_duration
                if stop_seconds is None
                else min(stop_seconds, byte_limited_duration)
            )
        samples = decoder.get_samples_played_in_range(
            0.0, stop_seconds + 1.0 / sample_rate
        )
    else:
        samples = decoder.get_all_samples()

    limits.check_samples(
        int(samples.data.shape[-1]),
        int(samples.sample_rate),
        int(samples.data.shape[0]),
        int(samples.data.element_size()),
    )
    return samples
