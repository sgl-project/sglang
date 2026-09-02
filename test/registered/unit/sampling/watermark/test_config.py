import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.sampling.watermark import (
    WatermarkConfigError,
    load_watermark_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_SECRET_A = "741852963"
_SECRET_B = "963852741"


class TestWatermarkConfig(CustomTestCase):
    def test_loads_textseal_config(self):
        registry = self._load(
            {
                "providers": {
                    "textseal": {
                        "enabled": True,
                        "key_a": _SECRET_A,
                        "key_b": _SECRET_B,
                        "ngram": 2,
                        "mixing_probability": 0.25,
                    }
                }
            }
        )

        config = registry.textseal
        self.assertEqual(config.key_a, int(_SECRET_A))
        self.assertEqual(config.key_b, int(_SECRET_B))
        self.assertEqual(config.ngram, 2)
        self.assertEqual(config.mixing_probability, 0.25)
        self.assertNotIn(_SECRET_A, repr(config))
        self.assertNotIn(_SECRET_B, repr(registry))

    def test_none_and_disabled_are_empty(self):
        self.assertFalse(load_watermark_config(None).textseal_enabled)
        registry = self._load({"providers": {"textseal": {"enabled": False}}})
        self.assertFalse(registry.textseal_enabled)

    def test_rejects_invalid_configuration_without_leaking_keys(self):
        invalid_configs = [
            {"providers": {"unknown": {}}},
            {
                "providers": {
                    "textseal": {
                        "enabled": True,
                        "key_a": _SECRET_A,
                        "key_b": _SECRET_B,
                        "unexpected": _SECRET_A,
                    }
                }
            },
            {
                "providers": {
                    "textseal": {
                        "enabled": True,
                        "key_a": _SECRET_A,
                        "key_b": _SECRET_A,
                    }
                }
            },
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(WatermarkConfigError) as context:
                    self._load(config)
                message = str(context.exception)
                self.assertNotIn(_SECRET_A, message)
                self.assertNotIn(_SECRET_B, message)

    def test_rejects_malformed_json_without_echoing_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watermark.json"
            path.write_text('{"secret":"741852963"', encoding="utf-8")
            with self.assertRaisesRegex(
                WatermarkConfigError, "failed to read watermark config JSON"
            ) as context:
                load_watermark_config(str(path))
        self.assertNotIn(_SECRET_A, str(context.exception))

    def test_server_args_validation_rejects_malformed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watermark.json"
            path.write_text('{"secret":"741852963"', encoding="utf-8")
            server_args = ServerArgs(model_path="dummy", watermark_config=str(path))
            server_args.resolve_once()
            with self.assertRaises(WatermarkConfigError) as context:
                server_args.check_server_args()
        self.assertNotIn(_SECRET_A, str(context.exception))

    def _load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watermark.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return load_watermark_config(str(path))


if __name__ == "__main__":
    unittest.main()
