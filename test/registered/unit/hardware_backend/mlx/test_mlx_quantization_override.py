"""Unit tests for ``MlxQuantizationConfig.override_quantization_method``.

The override is a classmethod over a dict; no mlx / Apple Silicon dependency.
Runs on every CI platform and guards #25119 from regression.
"""

from __future__ import annotations

import unittest

from sglang.srt.layers.quantization.mlx import MlxQuantizationConfig
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestMlxQuantizationOverride(unittest.TestCase):
    """Pure-logic tests for ``MlxQuantizationConfig.override_quantization_method``.

    The override is a classmethod over a dict; no mlx / Apple Silicon
    dependency. Runs on every CI platform and guards #25119 from regression.
    """

    def test_mlx_q4_dict_config_autodetect(self):
        """Bare {group_size, bits=4} dict maps to mlx_q4."""
        result = MlxQuantizationConfig.override_quantization_method(
            {"group_size": 64, "bits": 4}, None
        )
        self.assertEqual(result, "mlx_q4")

    def test_mlx_q8_dict_config_autodetect(self):
        """Bare {group_size, bits=8} dict maps to mlx_q8."""
        result = MlxQuantizationConfig.override_quantization_method(
            {"group_size": 32, "bits": 8}, None
        )
        self.assertEqual(result, "mlx_q8")

    def test_non_mlx_dict_not_matched(self):
        """Dicts with an explicit quant_method belong to that method, not ours."""
        # modelopt-style: explicit quant_method takes priority.
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(
                {"quant_method": "modelopt", "bits": 4, "group_size": 64}, None
            )
        )
        # gptq-style: same.
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(
                {"quant_method": "gptq", "bits": 4, "group_size": 128}, None
            )
        )

    def test_non_dict_not_matched(self):
        """Non-dict inputs and malformed dicts return None."""
        # None / string inputs.
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(None, None)
        )
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method("mlx_q4", None)
        )
        # Missing keys.
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method({"bits": 4}, None)
        )
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method({"group_size": 64}, None)
        )
        # Non-integer values.
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(
                {"bits": "4", "group_size": 64}, None
            )
        )
        # Unsupported bit-width.
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(
                {"bits": 2, "group_size": 64}, None
            )
        )

    def test_user_quant_explicit_defers_to_user(self):
        """When the user passes --quantization explicitly, defer to that choice."""
        # User chose mlx_q8 explicitly, even though config dict shape suggests q4
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(
                {"group_size": 64, "bits": 4}, "mlx_q8"
            )
        )
        # User chose mlx_q4 explicitly with matching config
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(
                {"group_size": 64, "bits": 4}, "mlx_q4"
            )
        )
        # User chose something completely different
        self.assertIsNone(
            MlxQuantizationConfig.override_quantization_method(
                {"group_size": 64, "bits": 4}, "fp8"
            )
        )


if __name__ == "__main__":
    unittest.main()
