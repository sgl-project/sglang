"""Config regressions for Bolt E4B NVFP4 DFlash.

Checkpoint: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4.
"""

import unittest

from transformers import Qwen3Config

from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
    parse_dflash_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashSlidingWindow(CustomTestCase):
    def make_config(self, **overrides):
        fields = dict(
            architectures=["DFlashDraftModel"],
            num_hidden_layers=5,
            layer_types=["sliding_attention"] * 5,
            sliding_window=1024,
            dflash_config={"use_swa": True, "swa_window_size": 1024},
        )
        fields.update(overrides)
        return Qwen3Config(**fields)

    def test_window_survives_qwen3_config_normalization(self):
        """Qwen3 clears the top-level window when its enable flag is absent."""
        config = self.make_config()
        self.assertIsNone(config.sliding_window)
        self.assertEqual(get_dflash_attention_sliding_window_size(config), 1023)

    def test_explicit_top_level_window_keeps_precedence(self):
        config = self.make_config(use_sliding_window=True, sliding_window=2048)
        self.assertEqual(get_dflash_attention_sliding_window_size(config), 2047)

    def test_metadata_fallback_requires_dflash_and_explicit_swa(self):
        for overrides in (
            {"architectures": ["Qwen3DSparkModel"]},
            {"architectures": ["Qwen3ForCausalLM"]},
            {"dflash_config": {"swa_window_size": 1024}},
            {"dflash_config": {"use_swa": False, "swa_window_size": 1024}},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    ValueError, "require config.sliding_window"
                ):
                    get_dflash_attention_sliding_window_size(
                        self.make_config(**overrides)
                    )

    def test_missing_or_invalid_metadata_window_is_rejected(self):
        for window in (None, 0, -1, True, 1.5):
            with self.subTest(window=window):
                config = self.make_config(
                    dflash_config={"use_swa": True, "swa_window_size": window}
                )
                with self.assertRaises(ValueError):
                    get_dflash_attention_sliding_window_size(config)

    def test_metadata_does_not_enable_full_attention_layers(self):
        config = self.make_config(layer_types=["full_attention"] * 5)
        self.assertIsNone(get_dflash_attention_sliding_window_size(config))


class TestDFlashHeadFlag(CustomTestCase):
    def test_head_flag_rejects_truthy_and_falsy_non_booleans(self):
        """Malformed flags must not silently select a different output head."""
        for value in ("false", "true", "", 0, 1, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "has_own_lm_head must be a bool"
                ):
                    parse_dflash_draft_config(
                        draft_hf_config={"has_own_lm_head": value}
                    )

    def test_head_flag_preserves_booleans_and_absent_default(self):
        for config, expected in (
            ({}, False),
            ({"has_own_lm_head": False}, False),
            ({"has_own_lm_head": True}, True),
        ):
            with self.subTest(config=config):
                parsed = parse_dflash_draft_config(draft_hf_config=config)
                self.assertIs(parsed.has_own_lm_head, expected)


if __name__ == "__main__":
    unittest.main()
