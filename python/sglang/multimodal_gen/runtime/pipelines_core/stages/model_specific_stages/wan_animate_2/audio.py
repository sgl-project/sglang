# SPDX-License-Identifier: Apache-2.0
"""Reference-video audio extraction for the Wan-Animate-2 before-denoising stage."""

from __future__ import annotations

from collections.abc import Callable

import torch

from sglang.srt.multimodal.audio_from_video import decode_audio_container

# Logger callables with the ``PipelineStage.log_*`` signature: ``(msg, *args)``.
LogFn = Callable[..., None]


def extract_reference_video_audio(
    reference_video_path: str,
    *,
    log_info: LogFn,
    log_warning: LogFn,
    log_no_audio_track: LogFn | None = None,
) -> tuple[torch.Tensor | None, int | None]:
    """Return ``(audio, sample_rate)`` for ``OutputBatch.audio``: audio ``[1, C, L]`` fp32 in
    [-1, 1] (dim 0 is the per-output axis the save path indexes) at the track's native sample
    rate and channel count, or ``(None, None)`` when the video has no decodable audio track.
    The track is decoded in-process by the shared ``decode_audio_container`` reader (PyAV).
    Best-effort: every failure is logged, none is raised. A video without an audio track goes
    to ``log_no_audio_track`` (``log_warning`` when None); a track that cannot be decoded goes
    to ``log_warning`` with the reader's message."""
    try:
        sample_rate = _audio_track_sample_rate(reference_video_path)
        if sample_rate is None:
            (log_no_audio_track or log_warning)(
                "Wan-Animate-2: reference video %s has no audio track; the output is silent.",
                reference_video_path,
            )
            return None, None
        # [L, C] fp32 in [-1, 1]; the reader needs a target rate, so the native one keeps the
        # track unresampled.
        samples = decode_audio_container(
            reference_video_path, target_sr=sample_rate, mono=False
        )
    except Exception as e:  # audio is best-effort: a failure never fails the request
        log_warning(
            "Wan-Animate-2: cannot decode the audio track of reference video %s (%s); "
            "the output is silent.",
            reference_video_path,
            e,
        )
        return None, None

    audio = torch.from_numpy(samples).T.contiguous().unsqueeze(0)  # [1, C, L]
    log_info(
        "Wan-Animate-2: extracted reference-video audio %s @ %d Hz.",
        tuple(audio.shape),
        sample_rate,
    )
    return audio, sample_rate


def _audio_track_sample_rate(path: str) -> int | None:
    """Native sample rate of the video's first audio stream; None when it has none."""
    import av

    with av.open(path) as container:
        if not container.streams.audio:
            return None
        return int(container.streams.audio[0].rate)
