# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V audio bucketing: which audio features each output video frame receives."""

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.models.encoders.wan_s2v_audio import (
    WanS2VAudioEncoder,
)

LAYERS, DIM = 2, 4


def _bucket(audio_frames, m):
    # Feature frame f of layer l holds 1000*l + f + 1: nonzero, and names its source.
    emb = (
        (
            1000 * torch.arange(LAYERS)[:, None, None]
            + torch.arange(audio_frames)[None, :, None]
            + 1
        )
        .float()
        .expand(LAYERS, audio_frames, DIM)
    )
    encoder = SimpleNamespace(video_rate=30)
    return WanS2VAudioEncoder.get_audio_embed_bucket_fps(
        encoder, emb.contiguous(), fps=16, batch_frames=81, m=m
    )


def _taps(bucket, frame, layer, m):
    return [int(v) for v in bucket[frame, layer].view(2 * m + 1, DIM)[:, 0]]


def test_audio_bucket_clamps_taps_and_zero_fills_past_the_audio():
    """Video frame k samples feature round(1.875 k) at 16 fps from 30 Hz features. Taps
    outside the audio clamp to its first/last frame; frames past its end are zero."""
    bucket, clips = _bucket(audio_frames=40, m=1)
    assert clips == 1 and bucket.shape == (81, LAYERS, 3 * DIM)
    # frame 0 -> feature 0: taps (-1, 0, 1) clamp to (0, 0, 1)
    assert _taps(bucket, 0, 0, m=1) == [1, 1, 2]
    # frame 21 -> feature 39, the last: taps (38, 39, 40) clamp to (38, 39, 39), not zero
    assert _taps(bucket, 21, 1, m=1) == [1039, 1040, 1040]
    # frame 22 -> feature 41, past the audio: every tap of every layer is zero
    assert not bucket[22:].any()


def test_audio_bucket_of_empty_audio_is_all_zero():
    """Audio shorter than one feature frame interpolates to zero frames; every video
    frame is then past the audio and must be zero-filled, not an index error."""
    bucket, clips = _bucket(audio_frames=0, m=0)
    assert clips == 1 and bucket.shape == (81, LAYERS, DIM)
    assert not bucket.any()
