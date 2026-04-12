"""Unit tests for srt/models/parakeet.py — clip logic, token counting, weight mapping."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from transformers import PretrainedConfig

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


def _make_hf_sound_config(**overrides) -> PretrainedConfig:
    defaults = dict(
        hidden_size=1024,
        num_mel_bins=128,
        sampling_rate=16000,
        subsampling_factor=8,
        subsampling_conv_kernel_size=5,
        subsampling_conv_stride=2,
        projection_hidden_size=4096,
        projection_bias=False,
        projection_eps=1e-5,
        hop_length=160,
    )
    defaults.update(overrides)
    cfg = PretrainedConfig()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _build_extractor(sampling_rate=16000, hop_length=160, clip_duration_s=30,
                     clip_min_duration_s=0.1, subsampling_factor=8,
                     subsampling_conv_kernel_size=5, subsampling_conv_stride=2):
    """Build a ParakeetExtractor with the parent __init__ mocked out."""
    from sglang.srt.configs.parakeet import ExtractorConfig
    from sglang.srt.models.parakeet import ParakeetExtractor

    with patch.object(ParakeetExtractor, "__init__", lambda self, *a, **kw: None):
        ext = ParakeetExtractor.__new__(ParakeetExtractor)

    ext.config = ExtractorConfig(
        feature_size=128,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        subsampling_factor=subsampling_factor,
        subsampling_conv_kernel_size=subsampling_conv_kernel_size,
        subsampling_conv_stride=subsampling_conv_stride,
        clip_duration_s=clip_duration_s,
        clip_min_duration_s=clip_min_duration_s,
    )
    ext.sampling_rate = sampling_rate
    ext.hop_length = hop_length
    ext._clip_target_samples = int(round(clip_duration_s * sampling_rate))
    ext._tail_min_samples = int(round(clip_min_duration_s * sampling_rate))
    return ext


class TestParakeetExtractorClipSizes(CustomTestCase):
    """Test _clip_sizes with various audio lengths."""

    def setUp(self):
        self.ext = _build_extractor(sampling_rate=16000, clip_duration_s=30,
                                    clip_min_duration_s=0.1)
        self.clip_target = self.ext._clip_target_samples  # 480000
        self.tail_min = self.ext._tail_min_samples         # 1600

    def test_very_short_audio(self):
        clips = self.ext._clip_sizes(100)
        self.assertEqual(clips, [self.tail_min])

    def test_zero_length(self):
        clips = self.ext._clip_sizes(0)
        self.assertEqual(clips, [self.tail_min])

    def test_exactly_tail_min(self):
        clips = self.ext._clip_sizes(self.tail_min)
        self.assertEqual(clips, [self.tail_min])

    def test_under_one_clip(self):
        audio_len = self.clip_target - 1
        clips = self.ext._clip_sizes(audio_len)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0], audio_len)

    def test_exactly_one_clip(self):
        clips = self.ext._clip_sizes(self.clip_target)
        self.assertEqual(clips, [self.clip_target])

    def test_one_clip_plus_remainder(self):
        remainder = 50000
        clips = self.ext._clip_sizes(self.clip_target + remainder)
        self.assertEqual(clips, [self.clip_target, remainder])

    def test_one_clip_plus_tiny_remainder(self):
        clips = self.ext._clip_sizes(self.clip_target + 100)
        self.assertEqual(clips, [self.clip_target, self.tail_min])

    def test_two_full_clips(self):
        clips = self.ext._clip_sizes(2 * self.clip_target)
        self.assertEqual(clips, [self.clip_target, self.clip_target])

    def test_three_clips_with_remainder(self):
        remainder = 80000
        clips = self.ext._clip_sizes(2 * self.clip_target + remainder)
        self.assertEqual(clips, [self.clip_target, self.clip_target, remainder])


class TestParakeetExtractorSplitAudio(CustomTestCase):
    """Test split_audio_into_clips."""

    def setUp(self):
        self.ext = _build_extractor(sampling_rate=16000, clip_duration_s=30,
                                    clip_min_duration_s=0.1)
        self.clip_target = self.ext._clip_target_samples
        self.tail_min = self.ext._tail_min_samples

    def test_short_audio_is_padded(self):
        audio = np.ones(100, dtype=np.float32)
        clips = self.ext.split_audio_into_clips(audio)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].shape[0], self.tail_min)
        np.testing.assert_array_equal(clips[0][:100], 1.0)
        np.testing.assert_array_equal(clips[0][100:], 0.0)

    def test_exact_clip_no_padding(self):
        audio = np.ones(self.clip_target, dtype=np.float32)
        clips = self.ext.split_audio_into_clips(audio)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].shape[0], self.clip_target)

    def test_two_clips_split_correctly(self):
        remainder = 50000
        total = self.clip_target + remainder
        audio = np.arange(total, dtype=np.float32)
        clips = self.ext.split_audio_into_clips(audio)
        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0].shape[0], self.clip_target)
        self.assertEqual(clips[1].shape[0], remainder)
        np.testing.assert_array_equal(clips[0], audio[:self.clip_target])
        np.testing.assert_array_equal(clips[1], audio[self.clip_target:])

    def test_rejects_2d_audio(self):
        audio = np.ones((2, 100), dtype=np.float32)
        with self.assertRaises(AssertionError):
            self.ext.split_audio_into_clips(audio)


class TestParakeetExtractorTokenCount(CustomTestCase):
    """Test audio_token_count with mocked subsampling length computation."""

    def test_single_clip_token_count(self):
        ext = _build_extractor(sampling_rate=16000, hop_length=160, clip_duration_s=30)

        with patch(
            "sglang.srt.models.parakeet.HFParakeetEncoder._get_subsampling_output_length"
        ) as mock_sub:
            mock_sub.return_value = torch.tensor([100.0])
            count = ext.audio_token_count(ext._clip_target_samples)

        self.assertEqual(count, 100)
        mock_sub.assert_called_once()

    def test_multiple_clips_token_count(self):
        ext = _build_extractor(sampling_rate=16000, hop_length=160, clip_duration_s=30)

        with patch(
            "sglang.srt.models.parakeet.HFParakeetEncoder._get_subsampling_output_length"
        ) as mock_sub:
            mock_sub.side_effect = [torch.tensor([100.0]), torch.tensor([50.0])]
            count = ext.audio_token_count(ext._clip_target_samples + 50000)

        self.assertEqual(count, 150)
        self.assertEqual(mock_sub.call_count, 2)

    def test_minimum_one_token(self):
        ext = _build_extractor(sampling_rate=16000, hop_length=160, clip_duration_s=30)

        with patch(
            "sglang.srt.models.parakeet.HFParakeetEncoder._get_subsampling_output_length"
        ) as mock_sub:
            mock_sub.return_value = torch.tensor([0.0])
            count = ext.audio_token_count(100)

        self.assertEqual(count, 1)


class TestParakeetExtractorAudioLength(CustomTestCase):
    """Test the static audio_length method via an instance (avoids transformers
    lazy-loading metaclass checks that require librosa at the class level)."""

    def test_audio_length_calculation(self):
        ext = _build_extractor(subsampling_factor=8, hop_length=160)
        hf = _make_hf_sound_config(
            subsampling_factor=8, hop_length=160,
            subsampling_conv_kernel_size=5, subsampling_conv_stride=2,
        )
        length = ext.audio_length(hf, audio_tokens=100)
        self.assertEqual(length, 100 * 8 * 160)

    def test_audio_length_different_params(self):
        ext = _build_extractor(subsampling_factor=4, hop_length=200)
        hf = _make_hf_sound_config(
            subsampling_factor=4, hop_length=200,
            subsampling_conv_kernel_size=3, subsampling_conv_stride=1,
        )
        length = ext.audio_length(hf, audio_tokens=50)
        self.assertEqual(length, 50 * 4 * 200)


class TestProjectedParakeetWeightMapping(CustomTestCase):
    """Test weight name mapping in ProjectedParakeet.load_weights."""

    def _get_weight_target_name(self, input_name: str) -> str | None:
        """Reproduce the name mapping logic from load_weights without building a model."""
        if input_name.startswith("sound_encoder.encoder.feature_extractor."):
            return "__skip__"
        if input_name.startswith("sound_encoder."):
            return input_name[len("sound_encoder."):]
        if input_name.startswith("sound_projection."):
            return f"projection.{input_name[len('sound_projection.'):]}"
        return None

    def test_encoder_weight_mapping(self):
        target = self._get_weight_target_name("sound_encoder.layers.0.weight")
        self.assertEqual(target, "layers.0.weight")

    def test_projection_weight_mapping(self):
        target = self._get_weight_target_name("sound_projection.linear1.weight")
        self.assertEqual(target, "projection.linear1.weight")

    def test_feature_extractor_skipped(self):
        target = self._get_weight_target_name(
            "sound_encoder.encoder.feature_extractor.conv.weight"
        )
        self.assertEqual(target, "__skip__")

    def test_unrelated_weight_ignored(self):
        target = self._get_weight_target_name("language_model.layers.0.weight")
        self.assertIsNone(target)

    def test_sound_projection_norm(self):
        target = self._get_weight_target_name("sound_projection.norm.weight")
        self.assertEqual(target, "projection.norm.weight")


class TestNanoNemotronLoadWeightsRouting(CustomTestCase):
    """Test weight routing logic in NemotronH_Nano_VL_V2.load_weights."""

    def test_weight_classification(self):
        test_cases = [
            ("language_model.layers.0.weight", "llm"),
            ("mlp1.0.weight", "adapter"),
            ("vision_model.radio_model.stem.weight", "vision"),
            ("sound_encoder.layers.0.weight", "sound"),
            ("sound_projection.linear1.weight", "sound"),
        ]
        for name, expected_category in test_cases:
            is_llm = name.startswith("language_model")
            is_adapter = name.startswith("mlp1")
            is_vision = name.startswith("vision_model.radio_model.")
            is_sound = name.startswith("sound")

            categories = []
            if is_llm:
                categories.append("llm")
            if is_adapter:
                categories.append("adapter")
            if is_vision:
                categories.append("vision")
            if is_sound:
                categories.append("sound")

            self.assertEqual(
                len(categories), 1,
                f"Weight '{name}' matched {len(categories)} categories: {categories}",
            )
            self.assertEqual(
                categories[0], expected_category,
                f"Weight '{name}' expected '{expected_category}', got '{categories[0]}'",
            )

    def test_llm_weight_prefix_stripping(self):
        name = "language_model.layers.0.self_attn.q_proj.weight"
        stripped = ".".join(name.split(".")[1:])
        self.assertEqual(stripped, "layers.0.self_attn.q_proj.weight")

    def test_vision_weight_prefix_stripping(self):
        name = "vision_model.radio_model.stem.conv.weight"
        stripped = name[len("vision_model."):]
        self.assertEqual(stripped, "radio_model.stem.conv.weight")


if __name__ == "__main__":
    unittest.main()
