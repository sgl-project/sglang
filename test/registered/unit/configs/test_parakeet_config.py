"""Unit tests for srt/configs/parakeet.py and sound_config in nano_nemotron_vl config."""

import unittest

from transformers import PretrainedConfig

from sglang.srt.configs.parakeet import ExtractorConfig, ParakeetConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_hf_sound_config(**overrides) -> PretrainedConfig:
    """Build a minimal PretrainedConfig that mimics a Parakeet sound_config blob."""
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


class TestParakeetConfig(CustomTestCase):
    def test_basic_construction(self):
        cfg = ParakeetConfig(
            llm_hidden_size=2048,
            projection_hidden_size=4096,
            projection_bias=False,
            sampling_rate=16000,
        )
        self.assertEqual(cfg.llm_hidden_size, 2048)
        self.assertEqual(cfg.projection_hidden_size, 4096)
        self.assertFalse(cfg.projection_bias)
        self.assertEqual(cfg.sampling_rate, 16000)
        self.assertAlmostEqual(cfg.projection_eps, 1e-5)

    def test_from_hf_config_copies_attributes(self):
        hf = _make_hf_sound_config(hidden_size=512)
        cfg = ParakeetConfig.from_hf_config(hf, llm_hidden_size=3072, max_model_len=4096)

        self.assertEqual(cfg.llm_hidden_size, 3072)
        self.assertEqual(cfg.hidden_size, 512)
        self.assertFalse(cfg.scale_input)
        self.assertFalse(cfg.attention_bias)
        self.assertEqual(cfg.max_position_embeddings, 4097)

    def test_from_hf_config_preserves_projection_params(self):
        hf = _make_hf_sound_config(
            projection_hidden_size=8192,
            projection_bias=True,
            projection_eps=1e-6,
        )
        cfg = ParakeetConfig.from_hf_config(hf, llm_hidden_size=2048, max_model_len=1024)

        self.assertEqual(cfg.projection_hidden_size, 8192)
        self.assertTrue(cfg.projection_bias)
        self.assertAlmostEqual(cfg.projection_eps, 1e-6)

    def test_from_hf_config_rejects_non_pretrained(self):
        with self.assertRaises(AssertionError):
            ParakeetConfig.from_hf_config(
                {"hidden_size": 512}, llm_hidden_size=2048, max_model_len=1024
            )


class TestExtractorConfig(CustomTestCase):
    def test_from_hf_config_maps_fields(self):
        hf = _make_hf_sound_config(
            num_mel_bins=80,
            sampling_rate=16000,
            hop_length=200,
            subsampling_factor=4,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=1,
        )
        ec = ExtractorConfig.from_hf_config(hf)

        self.assertEqual(ec.feature_size, 80)
        self.assertEqual(ec.sampling_rate, 16000)
        self.assertEqual(ec.hop_length, 200)
        self.assertEqual(ec.subsampling_factor, 4)
        self.assertEqual(ec.subsampling_conv_kernel_size, 3)
        self.assertEqual(ec.subsampling_conv_stride, 1)

    def test_from_hf_config_default_hop_length(self):
        hf = _make_hf_sound_config()
        delattr(hf, "hop_length")
        ec = ExtractorConfig.from_hf_config(hf)
        self.assertEqual(ec.hop_length, 160)

    def test_from_hf_config_rejects_non_pretrained(self):
        with self.assertRaises(AssertionError):
            ExtractorConfig.from_hf_config({"num_mel_bins": 128})

    def test_frozen_dataclass(self):
        ec = ExtractorConfig(
            feature_size=128,
            sampling_rate=16000,
            subsampling_factor=8,
            subsampling_conv_kernel_size=5,
            subsampling_conv_stride=2,
        )
        with self.assertRaises(AttributeError):
            ec.feature_size = 64

    def test_default_clip_params(self):
        ec = ExtractorConfig(
            feature_size=128,
            sampling_rate=16000,
            subsampling_factor=8,
            subsampling_conv_kernel_size=5,
            subsampling_conv_stride=2,
        )
        self.assertEqual(ec.clip_duration_s, 30)
        self.assertAlmostEqual(ec.clip_min_duration_s, 0.1)


class TestNemotronHConfigSoundConfig(CustomTestCase):
    """Test sound_config parsing in NemotronH_Nano_VL_V2_Config."""

    def _make_config(self, sound_config=None):
        from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config

        return NemotronH_Nano_VL_V2_Config(sound_config=sound_config)

    def test_sound_config_none(self):
        cfg = self._make_config(sound_config=None)
        self.assertIsNone(cfg.sound_config)

    def test_sound_config_from_dict(self):
        sound_dict = {
            "hidden_size": 1024,
            "num_mel_bins": 128,
            "sampling_rate": 16000,
            "subsampling_factor": 8,
            "subsampling_conv_kernel_size": 5,
            "subsampling_conv_stride": 2,
            "projection_hidden_size": 4096,
            "projection_bias": False,
        }
        cfg = self._make_config(sound_config=sound_dict)

        self.assertIsInstance(cfg.sound_config, PretrainedConfig)
        self.assertEqual(cfg.sound_config.hidden_size, 1024)
        self.assertEqual(cfg.sound_config.num_mel_bins, 128)
        self.assertEqual(cfg.sound_config.sampling_rate, 16000)

    def test_sound_config_preserves_pretrained_config(self):
        pc = PretrainedConfig()
        pc.hidden_size = 512
        cfg = self._make_config(sound_config=pc)

        self.assertIs(cfg.sound_config, pc)
        self.assertEqual(cfg.sound_config.hidden_size, 512)

    def test_audio_token_defaults(self):
        cfg = self._make_config()
        self.assertEqual(cfg.audio_context_token, "<so_embedding>")
        self.assertEqual(cfg.audio_start_token, "<so_start>")
        self.assertEqual(cfg.audio_end_token, "<so_end>")


if __name__ == "__main__":
    unittest.main()
