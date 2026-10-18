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

from sglang.srt.multimodal.audio_limits import (
    AudioDecodeLimits,
    AudioLimitExceeded,
    resolve_audio_decode_limits,
)

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
    limits: AudioDecodeLimits,
    target_sr: int,
    total_samples: int,
) -> int:
    for frame in frames:
        channels = 1 if mono else frame.layout.nb_channels
        sample_count = frame.samples
        limits.check_samples(
            total_samples + sample_count,
            target_sr,
            channels,
            np.dtype(np.float32).itemsize,
        )
        array = frame.to_ndarray()
        chunks.append(array.reshape(-1) if mono else array.T)
        total_samples += sample_count
    return total_samples


def decode_audio_container(
    source: bytes | str,
    *,
    target_sr: int,
    mono: bool,
    max_duration_seconds: float | None = None,
    max_decoded_bytes: int | None = None,
) -> np.ndarray:
    """Strictly decode the first audio stream from a supported container."""
    if not isinstance(target_sr, int) or target_sr <= 0:
        raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE)
    if isinstance(source, bytes) and not source:
        raise ValueError(_INVALID_AUDIO_CONTAINER_MESSAGE)
    limits = resolve_audio_decode_limits(max_duration_seconds, max_decoded_bytes)

    try:
        import av

        input_source = io.BytesIO(source) if isinstance(source, bytes) else source
        with av.open(input_source) as container:
            if not container.streams.audio:
                raise _AudioContainerDecodeError(_INVALID_AUDIO_CONTAINER_MESSAGE)

            audio_stream = container.streams.audio[0]
            if audio_stream.duration is not None and audio_stream.time_base is not None:
                limits.check_duration(
                    float(audio_stream.duration * audio_stream.time_base)
                )
            if mono:
                resampler = av.audio.resampler.AudioResampler(
                    format="fltp", layout="mono", rate=target_sr
                )
            else:
                resampler = av.audio.resampler.AudioResampler(
                    format="fltp", rate=target_sr
                )

            chunks: list[np.ndarray] = []
            skipped_packets = 0
            first_decode_error = None
            total_source_samples = 0
            total_samples = 0
            for packet in container.demux(audio_stream):
                try:
                    for frame in packet.decode():
                        if frame.sample_rate and frame.layout:
                            total_source_samples += frame.samples
                            limits.check_samples(
                                total_source_samples,
                                frame.sample_rate,
                                frame.layout.nb_channels,
                                np.dtype(np.float32).itemsize,
                                kind="source audio",
                            )
                        total_samples = _append_resampled_frames(
                            chunks,
                            resampler.resample(frame),
                            mono=mono,
                            limits=limits,
                            target_sr=target_sr,
                            total_samples=total_samples,
                        )
                except av.error.FFmpegError as error:
                    skipped_packets += 1
                    if first_decode_error is None:
                        first_decode_error = error

            total_samples = _append_resampled_frames(
                chunks,
                resampler.resample(None),
                mono=mono,
                limits=limits,
                target_sr=target_sr,
                total_samples=total_samples,
            )
            if skipped_packets:
                logger.warning(
                    "Skipped %d undecodable audio packet(s); kept %d decoded "
                    "chunk(s). First decode error: %s",
                    skipped_packets,
                    len(chunks),
                    first_decode_error,
                )
    except _AudioContainerDecodeError:
        raise
    except AudioLimitExceeded:
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
    except AudioLimitExceeded:
        raise
    except Exception:
        logger.warning("Error extracting audio from video", exc_info=True)
        return None
