"""Qwen3-VL config objects must survive construction and serialization."""

import unittest

from sglang.srt.configs.qwen3_vl import Qwen3VLConfig, Qwen3VLMoeConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3VLConfig(CustomTestCase):
    def test_subconfig_objects(self):
        """Object inputs used to leave text_config/vision_config unset."""
        for config_cls in (Qwen3VLConfig, Qwen3VLMoeConfig):
            for object_keys in (
                ("text_config",),
                ("vision_config",),
                ("text_config", "vision_config"),
            ):
                with self.subTest(config=config_cls.__name__, objects=object_keys):
                    inputs = {
                        "text_config": {"hidden_size": 128},
                        "vision_config": {"hidden_size": 64},
                    }
                    for key in object_keys:
                        inputs[key] = config_cls.sub_configs[key](**inputs[key])
                    config = config_cls(**inputs)
                    for key in object_keys:
                        self.assertIs(getattr(config, key), inputs[key])
                    self.assertEqual(config.text_config.hidden_size, 128)
                    self.assertEqual(config.vision_config.hidden_size, 64)

                    restored = config_cls.from_dict(config.to_dict())
                    self.assertEqual(restored.text_config.hidden_size, 128)
                    self.assertEqual(restored.vision_config.hidden_size, 64)
                    # save_pretrained uses the diff serialization path.
                    self.assertIn('"hidden_size": 128', config.to_json_string())
                    self.assertIn('"hidden_size": 64', config.to_json_string())


if __name__ == "__main__":
    unittest.main()
