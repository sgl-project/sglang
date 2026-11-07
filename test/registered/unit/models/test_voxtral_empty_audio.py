"""Regression test: Voxtral._encode_audio must not crash on empty waveforms.

An empty audio clip (e.g. a 0-sample WAV) passes the processor, which still
inserts one chunk of [AUDIO] placeholder tokens, but _encode_audio used to feed
the empty tensor straight into torch.stft and raise a RuntimeError inside the
scheduler process, taking the whole server down.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.models.voxtral import VoxtralForConditionalGeneration
from sglang.test.test_utils import CustomTestCase

_WINDOW_SIZE = 32
_HOP_LENGTH = 16
_NUM_MEL_BINS = 8
_MAX_SOURCE_POSITIONS = 8
# conv1.stride * conv2.stride of the tiny tower stub below
_CONV_DOWNSAMPLE = 2
_CHUNK_SIZE = _MAX_SOURCE_POSITIONS * _CONV_DOWNSAMPLE
_CHUNK_SAMPLES = _CHUNK_SIZE * _HOP_LENGTH


class _TinyAudioTower(nn.Module):
    """Stands in for VoxtralWhisperEncoder: same interfaces _encode_audio uses
    (conv1/conv2 strides + weights, callable on [B, mel, frames])."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(_NUM_MEL_BINS, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)).permute(0, 2, 1)


def _make_model() -> VoxtralForConditionalGeneration:
    model = object.__new__(VoxtralForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        audio_config=SimpleNamespace(max_source_positions=_MAX_SOURCE_POSITIONS)
    )
    model.audio_tower = _TinyAudioTower()
    model._window_size = _WINDOW_SIZE
    model._hop_length = _HOP_LENGTH
    model.mel_filters = torch.ones(1 + _WINDOW_SIZE // 2, _NUM_MEL_BINS)
    return model


class TestVoxtralEmptyAudio(CustomTestCase):
    def test_empty_waveform_yields_one_chunk(self):
        model = _make_model()
        features = model._encode_audio([torch.zeros(0)])
        self.assertEqual(len(features), 1)

    def test_chunk_counts(self):
        model = _make_model()
        # _encode_audio returns one entry per waveform, with all its chunks
        # flattened along dim 0: expected rows = n_chunks * max_source_positions.
        cases = [
            (1, 1),
            (_CHUNK_SAMPLES, 1),
            (_CHUNK_SAMPLES + 1, 2),
            (2 * _CHUNK_SAMPLES, 2),
        ]
        for n_samples, expected_chunks in cases:
            with self.subTest(n_samples=n_samples):
                features = model._encode_audio([torch.zeros(n_samples)])
                self.assertEqual(len(features), 1)
                self.assertEqual(
                    features[0].shape[0], expected_chunks * _MAX_SOURCE_POSITIONS
                )


if __name__ == "__main__":
    unittest.main()
