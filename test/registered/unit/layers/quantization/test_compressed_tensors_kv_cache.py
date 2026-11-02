"""Unit tests for compressed-tensors KV cache scale loading — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.test.test_utils import CustomTestCase

_FP8_TENSOR_KV_SCHEME = {
    "type": "float",
    "num_bits": 8,
    "strategy": "tensor",
    "symmetric": True,
    "dynamic": False,
}


def _config(kv_cache_scheme):
    cfg = {
        "format": "float-quantized",
        "quant_method": "compressed-tensors",
        "ignore": [],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }
    if kv_cache_scheme is not None:
        cfg["kv_cache_scheme"] = kv_cache_scheme
    return CompressedTensorsConfig.from_config(cfg)


def _attn():
    # __new__ is enough: get_quant_method only isinstance-checks the layer.
    return RadixAttention.__new__(RadixAttention)


class TestCompressedTensorsKVCacheMethod(CustomTestCase):
    def test_declared_scheme_gets_kv_cache_method(self):
        """A declared supported scheme must produce the KV cache method;
        without it the calibrated k_scale/v_scale have no parameters to
        load into and fp8 KV runs unscaled."""
        config = _config(_FP8_TENSOR_KV_SCHEME)
        method = config.get_quant_method(_attn(), "model.layers.0.attn")
        self.assertIsInstance(method, CompressedTensorsKVCacheMethod)

    def test_no_scheme_returns_none(self):
        config = _config(None)
        self.assertIsNone(config.get_quant_method(_attn(), "model.layers.0.attn"))

    def test_kv_cache_quant_algo_resolves_auto_dtype(self):
        """configure_kv_cache_dtype duck-types this field for --kv-cache-dtype
        auto: supported schemes must report FP8, everything else None, so
        loaded scales always meet an fp8 pool."""
        self.assertEqual(_config(_FP8_TENSOR_KV_SCHEME).kv_cache_quant_algo, "FP8")
        self.assertIsNone(_config(None).kv_cache_quant_algo)
        self.assertIsNone(
            _config(dict(_FP8_TENSOR_KV_SCHEME, dynamic=True)).kv_cache_quant_algo
        )

    def test_unsupported_scheme_degrades_to_none(self):
        """Unsupported declared schemes must skip the method, not fail
        the boot: such checkpoints serve with an unquantized-scale cache."""
        for bad in (
            dict(_FP8_TENSOR_KV_SCHEME, type="int"),
            dict(_FP8_TENSOR_KV_SCHEME, strategy="channel"),
            dict(_FP8_TENSOR_KV_SCHEME, symmetric=False),
            dict(_FP8_TENSOR_KV_SCHEME, dynamic=True),
        ):
            self.assertIsNone(
                _config(bad).get_quant_method(_attn(), "model.layers.0.attn")
            )


if __name__ == "__main__":
    unittest.main()
