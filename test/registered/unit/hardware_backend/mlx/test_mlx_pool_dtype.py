"""Unit tests for MLX attention-KV pool dtype inference.

The shared pool stores dequantized projection outputs, so its dtype must
follow the model's compute dtype, not the packed integer dtype of
QuantizedLinear weights.  Storing float32 for quantized models (the old
fallback) doubled pool bytes per slot for no precision gain — the source
values are bf16/fp16 — and made prefix-hit forwards run in float32 (the
pool gather promoted the concat) while no-hit forwards ran in the compute
dtype.  These tests pin the inference rules:

- unquantized projections: weight dtype (unchanged behavior);
- quantized projections: the quantization ``scales`` dtype (== compute
  dtype);
- quantized without usable scales: float32 (the conservative fallback).
"""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm"

if _HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen2

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        MlxModelCacheLayout,
        find_attention_layers,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


def _tiny_qwen2_model():
    """Randomly initialized 2-layer dense qwen2 (plain full attention)."""
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
    )
    return qwen2.Model(args)


def _runner_for(model):
    layers, attrs = find_attention_layers(model)
    runner = MlxModelRunner.__new__(MlxModelRunner)
    runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(layers, attrs)
    return runner


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestPoolDtypeInference(CustomTestCase):
    def test_unquantized_model_uses_weight_dtype(self):
        model = _tiny_qwen2_model()
        model.set_dtype(mx.float16)
        _, _, dtype = _runner_for(model)._get_attn_config()
        self.assertEqual(dtype, mx.float16)

    def test_quantized_model_uses_scales_dtype(self):
        model = _tiny_qwen2_model()
        model.set_dtype(mx.bfloat16)
        nn.quantize(model, group_size=64, bits=4)
        attn = model.model.layers[0].self_attn
        self.assertNotIn(attn.k_proj.weight.dtype, {mx.float16, mx.bfloat16})
        self.assertEqual(attn.k_proj.scales.dtype, mx.bfloat16)
        _, _, dtype = _runner_for(model)._get_attn_config()
        self.assertEqual(dtype, mx.bfloat16)

    def test_quantized_fp16_model_uses_scales_dtype(self):
        model = _tiny_qwen2_model()
        model.set_dtype(mx.float16)
        nn.quantize(model, group_size=64, bits=4)
        _, _, dtype = _runner_for(model)._get_attn_config()
        self.assertEqual(dtype, mx.float16)

    def test_quantized_without_usable_scales_falls_back_to_float32(self):
        model = _tiny_qwen2_model()
        model.set_dtype(mx.bfloat16)
        nn.quantize(model, group_size=64, bits=4)
        for layer in model.model.layers:
            # Simulate a packed layer whose scales are not a float array
            # (e.g. an exotic quant format): the conservative fallback must
            # hold.
            layer.self_attn.k_proj.scales = layer.self_attn.k_proj.scales.astype(
                mx.uint32
            )
        _, _, dtype = _runner_for(model)._get_attn_config()
        self.assertEqual(dtype, mx.float32)


if __name__ == "__main__":
    unittest.main()
