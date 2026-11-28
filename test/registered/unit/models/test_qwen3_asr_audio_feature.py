"""Qwen3-ASR audio-feature batching on CPU (no weights, no GPU).

Long audio is split into chunks and transcribed concurrently, so one encoder
batch can hold audio items of different mel lengths. ``get_audio_feature`` used
to ``torch.cat`` the per-item features directly and raised
"Sizes of tensors must match except in dimension 0". These tests pin the padded
+ masked path, and that a batch of equal-length items still produces exactly
what the per-item (pre-change) path produced.

The audio tower is replaced by a recorder, so the assertions are about the
tensor layout handed to the encoder rather than about encoder numerics.
"""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

NUM_MEL_BINS = 8


class _RecordingAudioTower(nn.Module):
    """Stand-in for ``Qwen3OmniMoeAudioEncoder`` that records its inputs."""

    dtype = torch.float32

    def __init__(self):
        super().__init__()
        # ``get_audio_feature`` reads the device off the first parameter.
        self.proj = nn.Parameter(torch.zeros(1))
        self.feature_lens = []
        self.input_shapes = []

    def forward(self, input_features, feature_lens=None):
        self.feature_lens.append(feature_lens.tolist())
        self.input_shapes.append(tuple(input_features.shape))
        hidden = input_features.new_zeros((1, input_features.shape[-1], 4))
        return SimpleNamespace(last_hidden_state=hidden)


def _model():
    model = Qwen3ASRForConditionalGeneration.__new__(Qwen3ASRForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_tower = _RecordingAudioTower()
    return model


def _record_inputs(model):
    """Swap the tower's forward for a recorder and return the recorded inputs."""
    recorded = []

    def _record(input_features, feature_lens=None):
        recorded.append(input_features)
        return SimpleNamespace(last_hidden_state=input_features.new_zeros(1))

    model.audio_tower.forward = _record
    return recorded


def _item(frames, mask=None):
    # Duck-typed MultimodalDataItem: the model only reads ``feature`` and an
    # optional ``feature_attention_mask``. Shape is (batch, mel, frames), as
    # produced by the HF feature extractor.
    item = SimpleNamespace(feature=torch.randn(1, NUM_MEL_BINS, frames))
    if mask is not None:
        item.feature_attention_mask = mask
    return item


class TestGetAudioFeatureBatching(CustomTestCase):
    def test_uneven_length_items_are_padded_and_masked(self):
        model = _model()
        items = [
            _item(100, torch.ones((1, 100), dtype=torch.long)),
            _item(120, torch.ones((1, 120), dtype=torch.long)),
        ]

        model.get_audio_feature(items)

        # Each item keeps its own length; the concatenated feature holds only
        # the valid frames, so the padding never reaches the encoder.
        self.assertEqual(model.audio_tower.feature_lens, [[100, 120]])
        self.assertEqual(model.audio_tower.input_shapes, [(NUM_MEL_BINS, 220)])

    def test_uneven_length_items_without_mask_are_padded_and_masked(self):
        model = _model()

        model.get_audio_feature([_item(100), _item(120)])

        self.assertEqual(model.audio_tower.feature_lens, [[100, 120]])
        self.assertEqual(model.audio_tower.input_shapes, [(NUM_MEL_BINS, 220)])

    def test_equal_length_items_match_legacy_masked_path(self):
        # Qwen3-ASR's processor always emits a mask, so this is the production
        # path: the batched result must be element-wise identical to the
        # pre-change expression (same permute / boolean-index order).
        model = _model()
        first = _item(100, torch.ones((1, 100), dtype=torch.long))
        second = _item(100, torch.ones((1, 100), dtype=torch.long))
        legacy = (
            torch.cat([first.feature, second.feature])
            .type(torch.float32)
            .permute(0, 2, 1)[torch.ones((2, 100), dtype=torch.long).bool()]
            .permute(1, 0)
        )
        recorded = _record_inputs(model)

        model.get_audio_feature([first, second])

        self.assertTrue(torch.equal(recorded[0], legacy))

    def test_equal_length_unmasked_matches_per_item_encoding(self):
        # Without a mask the old code reshaped straight to (frames, mel), i.e.
        # the transpose of the layout the encoder consumes it as. Encoding the
        # items one by one must give the same encoder input as batching them.
        model = _model()
        first = _item(100)
        second = _item(100)
        recorded = _record_inputs(model)

        model.get_audio_feature([first, second])
        batched = recorded.pop()
        model.get_audio_feature([first])
        model.get_audio_feature([second])

        self.assertTrue(torch.equal(batched, torch.cat(recorded, dim=-1)))
        self.assertEqual(batched.shape, (NUM_MEL_BINS, 200))


if __name__ == "__main__":
    unittest.main()
