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
"""Extract audio from video bytes using PyAV (in-process, CUDA-safe).

PyAV wraps FFmpeg's C libraries in-process, avoiding subprocess forks which
would crash CUDA-active workers.
"""

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_audio_from_video_bytes(
    video_bytes: bytes,
    target_sr: int = 16000,
) -> np.ndarray | None:
    """Extract mono audio from video bytes at the target sample rate.

    Args:
        video_bytes: Raw video file bytes (e.g. MP4).
        target_sr: Target sample rate for the output waveform.

    Returns:
        1-D float32 numpy array of audio samples, or None if the video
        has no audio track.
    """
    try:
        import av
    except ImportError:
        logger.warning(
            "PyAV (av) is not installed. Cannot extract audio from video. "
            "Install with: pip install av"
        )
        return None

    try:
        container = av.open(io.BytesIO(video_bytes))
    except Exception:
        logger.warning("Failed to open video bytes for audio extraction")
        return None

    if not container.streams.audio:
        container.close()
        return None

    try:
        audio_stream = container.streams.audio[0]
        native_sr = audio_stream.rate or target_sr

        resampler = av.audio.resampler.AudioResampler(
            format="flt",
            layout="mono",
            rate=target_sr,
        )

        chunks = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for rf in resampled:
                arr = rf.to_ndarray().flatten()
                chunks.append(arr)

        container.close()

        if not chunks:
            return None

        waveform = np.concatenate(chunks).astype(np.float32)
        return waveform

    except Exception:
        logger.warning("Error extracting audio from video", exc_info=True)
        container.close()
        return None
