"""Unit tests for fp8 KV cache k/v scale validation at load - no server, no model."""

import math
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer_with_scales(k_scale: float, v_scale: float) -> torch.nn.Module:
    layer = torch.nn.Module()
    BaseKVCacheMethod(quant_config=mock.MagicMock()).create_weights(layer)
    layer.k_scale.copy_(torch.tensor(k_scale, dtype=torch.float32))
    layer.v_scale.copy_(torch.tensor(v_scale, dtype=torch.float32))
    return layer


class TestKVCacheScaleValidation(CustomTestCase):
    def setUp(self):
        self.method = BaseKVCacheMethod(quant_config=mock.MagicMock())
        patcher = mock.patch(
            "sglang.srt.layers.quantization.kv_cache.is_fp8_fnuz",
            return_value=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_zero_k_scale_rejected(self):
        layer = _layer_with_scales(0.0, 1.0)
        with self.assertRaises(AssertionError):
            self.method.process_weights_after_loading(layer)

    def test_inf_k_scale_rejected(self):
        layer = _layer_with_scales(math.inf, 1.0)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.method.process_weights_after_loading(layer)

    def test_neg_inf_v_scale_rejected(self):
        layer = _layer_with_scales(1.0, -math.inf)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.method.process_weights_after_loading(layer)

    def test_nan_k_scale_rejected(self):
        layer = _layer_with_scales(math.nan, 1.0)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.method.process_weights_after_loading(layer)

    def test_multi_element_scale_rejected_with_value_error(self):
        layer = _layer_with_scales(1.0, 1.0)
        layer.k_scale = torch.nn.Parameter(
            torch.tensor([1.0, math.nan]), requires_grad=False
        )
        with self.assertRaisesRegex(ValueError, "per-tensor"):
            self.method.process_weights_after_loading(layer)

        layer = _layer_with_scales(1.0, 1.0)
        layer.k_scale = torch.nn.Parameter(
            torch.tensor([1.0, 2.0]), requires_grad=False
        )
        with self.assertRaisesRegex(ValueError, "per-tensor"):
            self.method.process_weights_after_loading(layer)

    def test_finite_scales_accepted(self):
        layer = _layer_with_scales(1.0, 1.0)
        self.method.process_weights_after_loading(layer)
        self.assertEqual(layer.k_scale_float, 1.0)
        self.assertEqual(layer.v_scale_float, 1.0)

    def test_fnuz_doubling_overflow_rejected(self):
        # 3e38 is finite float32 but doubles past the float32 maximum on FNUZ.
        with mock.patch(
            "sglang.srt.layers.quantization.kv_cache.is_fp8_fnuz",
            return_value=True,
        ):
            layer = _layer_with_scales(3e38, 1.0)
            with self.assertRaisesRegex(ValueError, "must be finite"):
                self.method.process_weights_after_loading(layer)

    def test_fnuz_doubling_within_float32_accepted(self):
        with mock.patch(
            "sglang.srt.layers.quantization.kv_cache.is_fp8_fnuz",
            return_value=True,
        ):
            layer = _layer_with_scales(1e38, 1.0)
            self.method.process_weights_after_loading(layer)
            # 1e38 is not exactly representable in float32, so the doubled
            # value is twice the float32-rounded checkpoint value.
            self.assertEqual(
                layer.k_scale_float, torch.tensor(1e38, dtype=torch.float32).item() * 2
            )
            self.assertEqual(layer.v_scale_float, 2.0)

    def test_unset_scales_default_to_one(self):
        layer = _layer_with_scales(-1.0, -1.0)
        self.method.process_weights_after_loading(layer)
        self.assertEqual(layer.k_scale_float, 1.0)
        self.assertEqual(layer.v_scale_float, 1.0)


if __name__ == "__main__":
    unittest.main()
