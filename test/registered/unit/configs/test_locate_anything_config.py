"""Unit tests for ``sglang.srt.configs.locate_anything.LocateAnythingConfig``."""

import unittest

from transformers.models.qwen2 import Qwen2Config

from sglang.srt.configs import LocateAnythingConfig
from sglang.srt.configs.kimi_vl_moonvit import MoonViTConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocateAnythingConfig(CustomTestCase):
    def test_default_fields(self):
        """Defaults reflect the nvidia/LocateAnything-3B reference config."""
        cfg = LocateAnythingConfig()
        self.assertEqual(cfg.model_type, "locateanything")
        # Special token ids used by the grounding grammar.
        self.assertEqual(cfg.image_token_index, 151665)
        self.assertEqual(cfg.box_start_token_id, 151668)
        self.assertEqual(cfg.box_end_token_id, 151669)
        self.assertEqual(cfg.ref_start_token_id, 151672)
        self.assertEqual(cfg.ref_end_token_id, 151673)
        self.assertEqual(cfg.coord_start_token_id, 151677)
        self.assertEqual(cfg.coord_end_token_id, 152677)
        self.assertEqual(cfg.none_token_id, 4064)
        self.assertEqual(cfg.mlp_connector_layers, 2)

    def test_composite_subconfigs_default(self):
        cfg = LocateAnythingConfig()
        self.assertIsInstance(cfg.vision_config, MoonViTConfig)
        self.assertIsInstance(cfg.text_config, Qwen2Config)

    def test_subconfigs_from_dict(self):
        cfg = LocateAnythingConfig(
            vision_config={"hidden_size": 1152, "merge_kernel_size": [2, 2]},
            text_config={"hidden_size": 2048, "tie_word_embeddings": True},
        )
        self.assertIsInstance(cfg.vision_config, MoonViTConfig)
        self.assertIsInstance(cfg.text_config, Qwen2Config)
        self.assertEqual(cfg.vision_config.hidden_size, 1152)
        self.assertEqual(cfg.text_config.hidden_size, 2048)
        self.assertTrue(cfg.text_config.tie_word_embeddings)

    def test_subconfigs_passthrough_instances(self):
        vision = MoonViTConfig(hidden_size=1152)
        text = Qwen2Config(hidden_size=2048)
        cfg = LocateAnythingConfig(vision_config=vision, text_config=text)
        self.assertIs(cfg.vision_config, vision)
        self.assertIs(cfg.text_config, text)

    def test_registered_in_config_registry(self):
        """``model_type`` resolves to the config class via SGLang's registry."""
        from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

        self.assertIs(_CONFIG_REGISTRY.get("locateanything"), LocateAnythingConfig)


if __name__ == "__main__":
    unittest.main()
