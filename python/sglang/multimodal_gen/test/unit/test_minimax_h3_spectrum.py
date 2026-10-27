# SPDX-License-Identifier: Apache-2.0
"""H3 Spectrum packing helpers and sampling-param gates."""

from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.cache.spectrum import (
    H3_SPECTRUM_HISTORY_SIZE,
    blend_h3_spectrum_prediction,
    local_target_hidden,
    target_audio_video_span,
)


def test_target_span_uses_contiguous_audio_before_video():
    audio_pos = torch.tensor([4, 5, 6, 7, 8, 9])
    infer_out_pos = torch.tensor([10, 11, 12, 13])
    start, stop, video_stop = target_audio_video_span(audio_pos, infer_out_pos)
    assert (start, stop, video_stop) == (4, 10, 14)


def test_target_span_drops_earlier_ref_audio_gap():
    audio_pos = torch.tensor([1, 2, 8, 9])
    infer_out_pos = torch.tensor([10, 11])
    start, stop, video_stop = target_audio_video_span(audio_pos, infer_out_pos)
    assert (start, stop, video_stop) == (8, 10, 12)


def test_local_target_hidden_is_rank_local_view():
    hidden = torch.arange(20, dtype=torch.float32).unsqueeze(-1).expand(20, 3).clone()
    sliced = local_target_hidden(
        hidden, audio_start=4, video_stop=10, row_start=0, row_stop=20
    )
    assert sliced is not None
    local, start, stop = sliced
    assert start == 4 and stop == 10
    assert local.shape == (6, 3)


def test_blend_holds_audio_and_mixes_video():
    predicted = torch.ones(6, 2)
    last_audio = torch.full((2, 2), 3.0)
    last_video = torch.full((4, 2), 5.0)
    out = blend_h3_spectrum_prediction(
        predicted.clone(),
        n_audio=2,
        last_audio=last_audio,
        last_video=last_video,
        audio_blend=0.0,
        video_blend=0.5,
    )
    assert torch.equal(out[:2], last_audio)
    assert torch.allclose(out[2:], torch.full((4, 2), 3.0))


def test_h3_sampling_params_reject_quality_high_with_spectrum():
    with pytest.raises(ValueError, match="enable_spectrum cannot be combined"):
        MiniMaxH3SamplingParams(
            prompt="cat",
            enable_spectrum=True,
            quality="high",
        )


def test_h3_sampling_params_default_history_is_comfy_window():
    params = MiniMaxH3SamplingParams(prompt="cat", enable_spectrum=True)
    assert params.spectrum_params is not None
    assert params.spectrum_params.history_size == H3_SPECTRUM_HISTORY_SIZE


def test_h3_sampling_params_keep_explicit_history_size():
    params = MiniMaxH3SamplingParams(
        prompt="cat",
        enable_spectrum=True,
        spectrum_params={"history_size": 6},
    )
    assert params.spectrum_params.history_size == 6
