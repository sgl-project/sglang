import unittest

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLoadConfig(CustomTestCase):
    def test_model_loader_extra_config_normalization(self):
        """Ensure None, empty strings, and JSON object strings normalize to dict."""
        self.assertEqual(
            LoadConfig(model_loader_extra_config=None).model_loader_extra_config, {}
        )
        self.assertEqual(
            LoadConfig(model_loader_extra_config="").model_loader_extra_config, {}
        )
        self.assertEqual(
            LoadConfig(
                model_loader_extra_config='{"concurrency": 4}'
            ).model_loader_extra_config,
            {"concurrency": 4},
        )
        with self.assertRaises(ValueError):
            LoadConfig(model_loader_extra_config="[1, 2, 3]")

    def test_ignore_patterns_and_load_format_normalization(self):
        """Ensure string ignore_patterns and load formats normalize cleanly."""
        cfg = LoadConfig(
            load_format="SAFETENSORS",
            fallback_load_format="PT",
            ignore_patterns="*.bin",
        )
        self.assertEqual(cfg.load_format, LoadFormat.SAFETENSORS)
        self.assertEqual(cfg.fallback_load_format, LoadFormat.PT)
        self.assertEqual(cfg.ignore_patterns, ["*.bin"])
        self.assertEqual(
            LoadConfig(ignore_patterns="").ignore_patterns, ["original/**/*"]
        )


if __name__ == "__main__":
    unittest.main()
