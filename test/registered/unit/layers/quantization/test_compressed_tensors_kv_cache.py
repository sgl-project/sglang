"""Unit tests for compressed-tensors KV cache scale loading — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch
from torch import nn

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
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


def _model_with_loaded_kv_scales(k_scale: float, v_scale: float) -> nn.Module:
    attn = _attn()
    nn.Module.__init__(attn)
    attn.k_scale = nn.Parameter(torch.tensor(k_scale), requires_grad=False)
    attn.v_scale = nn.Parameter(torch.tensor(v_scale), requires_grad=False)
    attn.k_scale_float, attn.v_scale_float = k_scale, v_scale
    return nn.Sequential(attn)


def _configure(kv_cache_dtype: str, model: nn.Module) -> torch.dtype:
    _, dtype = configure_kv_cache_dtype(
        server_args_kv_cache_dtype=kv_cache_dtype,
        model=model,
        model_dtype=torch.bfloat16,
        is_draft_worker=False,
        is_dflash=False,
        speculative_draft_attention_backend=None,
    )
    return dtype


class TestKVScalesFollowCacheDtype(CustomTestCase):
    def test_bf16_cache_ignores_checkpoint_scales(self):
        """A bf16 cache stores K/V unscaled while attention backends multiply the
        checkpoint's fp8 calibration scales back on read; keeping ~0.03 scales
        flattened attention and produced incoherent long-context output."""
        model = _model_with_loaded_kv_scales(0.035, 0.025)
        self.assertEqual(_configure("bf16", model), torch.bfloat16)
        attn = model[0]
        self.assertEqual((attn.k_scale_float, attn.v_scale_float), (1.0, 1.0))
        self.assertEqual((attn.k_scale.item(), attn.v_scale.item()), (1.0, 1.0))

    def test_fp8_cache_keeps_checkpoint_scales(self):
        model = _model_with_loaded_kv_scales(0.035, 0.025)
        self.assertEqual(_configure("fp8_e4m3", model), torch.float8_e4m3fn)
        self.assertEqual(
            (model[0].k_scale_float, model[0].v_scale_float), (0.035, 0.025)
        )


if __name__ == "__main__":
    unittest.main()
