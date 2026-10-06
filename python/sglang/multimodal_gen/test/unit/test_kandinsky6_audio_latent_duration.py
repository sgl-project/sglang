# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``audio_latent_duration``, the formula that keeps
Kandinsky6's audio and video latents time-aligned.

Ported from FastVideo's
``fastvideo/tests/stages/test_kandinsky6_audio_latent_duration.py``. Byte-for-
byte match to the diffusers reference's
``pipeline_kandinsky6_t2va.audio_latent_duration``:
``ceil(((T_lat-1)*4+1) / fps * audio_sample_rate / audio_downsample_factor)``,
where ``(T_lat-1)*4+1`` is the causal video VAE's pixel-frame count for a
given latent-frame count (temporal_compression_ratio == 4, matching
HunyuanVAEConfig -- the video VAE Kandinsky6 reuses).

Pure Python/math, no GPU or model weights needed.
"""

from __future__ import annotations

import math

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.latent_preparation import (
    audio_latent_duration,
)


def test_audio_latent_duration_matches_diffusers_reference_default_5s_clip():
    # Native T2VA default operating point: num_frames=121 -> T_lat=31 at
    # temporal_compression_ratio=4 ((121-1)//4+1 == 31).
    result = audio_latent_duration(
        31, fps=24.0, audio_sample_rate=44100, audio_downsample_factor=1024
    )
    pixel_frames = (31 - 1) * 4 + 1
    expected = math.ceil(pixel_frames / 24.0 * 44100 / 1024)
    assert result == expected
    assert result == 218


def test_audio_latent_duration_single_latent_frame():
    # T_lat=1 -> pixel_frames=1 (the causal VAE's "+1 first frame" edge case).
    result = audio_latent_duration(
        1, fps=24.0, audio_sample_rate=44100, audio_downsample_factor=1024
    )
    assert result == math.ceil(1 / 24.0 * 44100 / 1024)


def test_audio_latent_duration_scales_with_video_length():
    short = audio_latent_duration(
        16, fps=24.0, audio_sample_rate=44100, audio_downsample_factor=1024
    )
    long = audio_latent_duration(
        31, fps=24.0, audio_sample_rate=44100, audio_downsample_factor=1024
    )
    assert long > short


def test_audio_latent_duration_returns_int():
    result = audio_latent_duration(
        31, fps=24.0, audio_sample_rate=44100, audio_downsample_factor=1024
    )
    assert isinstance(result, int)
