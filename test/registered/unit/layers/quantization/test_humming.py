"""Unit tests for Humming quantization config detection."""

import unittest

from sglang.srt.layers.quantization.humming import HummingConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHummingQuantizationOverride(unittest.TestCase):
    def test_override_ignores_config_without_quant_method(self):
        self.assertIsNone(
            HummingConfig.override_quantization_method(
                {"group_size": 64, "bits": 4}, None
            )
        )


if __name__ == "__main__":
    unittest.main()
