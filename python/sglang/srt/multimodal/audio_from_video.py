# Copyright 2025 SGLang Team
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
"""Decode audio from media containers with PyAV."""

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)

_INVALID_AUDIO_CONTAINER_MESSAGE = (
    "Invalid input_audio: no decodable audio stream was found in the media container."
)
_AUDIO_CONTAINER_SIGNATURES = (
    ((4, b"ftyp"),),
    ((0, b"RIFF"), (8, b"AVI ")),
    ((0, b"#!AMR\n"),),
    ((0, b"#!AMR-WB\n"),),
    # EBML magic: WebM / Matroska (e.g. browser MediaRecorder output)
    ((0, b"\x1a\x45\xdf\xa3"),),
)


class _AudioContainerDecodeError(ValueError):
    pass


class _AudioDecodeLimitError(ValueError):
    pass


def is_audio_container(data: bytes) -> bool:
    """Return whether the header identifies a supported media container."""
    return any(
        all(data[offset : offset + len(magic)] == magic for offset, magic in signature)
        for signature in _AUDIO_CONTAINER_SIGNATURES
    )


def _append_resampled_frames(
    chunks: list[np.ndarray],
    frames,
    *,
    mono: bool,
    max_samples: int | None,
    max_decode_bytes: int | None,
    sample_count: int,
    decoded_bytes: int,
) -> tuple[int, int]:
    for frame in frames:
        # The resampler is configured for float32 output. PyAV already owns the
        # returned frame; reject from its dimensions before projecting it to an
        # application-owned NumPy buffer or retaining it for concatenation.
        frame_samples = int(frame.samples)
        channels = 1 if mono else len(frame.layout.channels)
        prospective_samples = sample_count + frame_samples
        if max_samples is not None and prospective_samples > max_samples:
            raise _AudioDecodeLimitError(
                "Audio exceeds the maximum allowed decode duration. Set "
                "SGLANG_MAX_AUDIO_DECODE_DURATION_S to change this limit."
            )
        prospective_bytes = decoded_bytes + frame_samples * channels * 4
        if max_decode_bytes is not None and prospective_bytes > max_decode_bytes:
            raise _AudioDecodeLimitError(
                "Audio exceeds the maximum allowed decoded size. Set "
                "SGLANG_MAX_AUDIO_DECODE_BYTES to change this limit."
            )

        array = frame.to_ndarray()
        chunk = array.reshape(-1) if mono else array.T
        sample_count = prospective_samples
        decoded_bytes = prospective_bytes
        chunks.append(chunk)
    return sample_count, decoded_bytes


def decode_audio_container(
    source: bytes | str,
    *,
    target_sr: int,
    mono: bool,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> np.ndarray:
    """Strictly decode the first audio stream from a supported container."""
    if not isinstance(target_sr, int) or target_sr <= 0:
        raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE)
    if isinstance(source, bytes) and not source:
        raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE)
    from sglang.srt.utils.common import (
        _audio_decode_byte_limit,
        _audio_decode_duration_limit,
        _audio_duration_frame_count,
    )

    max_duration_s = _audio_decode_duration_limit(max_duration_s)
    max_decode_bytes = _audio_decode_byte_limit(max_decode_bytes)
    max_samples = (
        _audio_duration_frame_count(max_duration_s, target_sr)
        if max_duration_s is not None
        else None
    )

    try:
        import av

        input_source = io.BytesIO(source) if isinstance(source, bytes) else source
        with av.open(input_source) as container:
            if not container.streams.audio:
                raise _AudioContainerDecodeError(_INVALID_AUDIO_CONTAINER_MESSAGE)

            audio_stream = container.streams.audio[0]
            if mono:
                resampler = av.audio.resampler.AudioResampler(
                    format="fltp", layout="mono", rate=target_sr
                )
            else:
                resampler = av.audio.resampler.AudioResampler(
                    format="fltp", rate=target_sr
                )

            chunks: list[np.ndarray] = []
            sample_count = 0
            decoded_bytes = 0
            skipped_packets = 0
            first_decode_error = None
            for packet in container.demux(audio_stream):
                try:
                    for frame in packet.decode():
                        sample_count, decoded_bytes = _append_resampled_frames(
                            chunks,
                            resampler.resample(frame),
                            mono=mono,
                            max_samples=max_samples,
                            max_decode_bytes=max_decode_bytes,
                            sample_count=sample_count,
                            decoded_bytes=decoded_bytes,
                        )
                except av.error.FFmpegError as error:
                    skipped_packets += 1
                    if first_decode_error is None:
                        first_decode_error = error

            sample_count, decoded_bytes = _append_resampled_frames(
                chunks,
                resampler.resample(None),
                mono=mono,
                max_samples=max_samples,
                max_decode_bytes=max_decode_bytes,
                sample_count=sample_count,
                decoded_bytes=decoded_bytes,
            )
            if skipped_packets:
                logger.warning(
                    "Skipped %d undecodable audio packet(s); kept %d decoded "
                    "chunk(s). First decode error: %s",
                    skipped_packets,
                    len(chunks),
                    first_decode_error,
                )
    except (_AudioContainerDecodeError, _AudioDecodeLimitError):
        raise
    except Exception as error:
        raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE) from error

    if not chunks:
        if first_decode_error is not None:
            raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE) from first_decode_error
        raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE)

    waveform = np.concatenate(chunks, axis=0)
    expected_ndim = 1 if mono else 2
    if waveform.ndim != expected_ndim or waveform.size == 0:
        raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def extract_audio_from_video_bytes(
    video_bytes: bytes,
    target_sr: int = 16000,
) -> np.ndarray | None:
    """Extract optional mono audio for callers that accept silent videos."""
    try:
        return decode_audio_container(
            video_bytes,
            target_sr=target_sr,
            mono=True,
        )
    except Exception:
        logger.warning("Error extracting audio from video", exc_info=True)
        return None
