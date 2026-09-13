"""CPU regression tests for compressed-tensors quantized ParallelLMHead dispatch.

A compressed-tensors checkpoint that targets ``lm_head`` (e.g. FP8 W8A8
per-channel or WNA16 pack-quantized) used to fall through to
``UnquantizedEmbeddingMethod`` because ``get_quant_method()`` only dispatched
``LinearBase`` and ``FusedMoE``. The per-channel ``weight_scale`` was then never
loaded and the raw quantized weight was consumed as bf16, corrupting logits and
producing degenerate repetition loops. These tests cover the
``ParallelLMHead`` branch added to ``CompressedTensorsConfig.get_quant_method()``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
    CompressedTensorsWNA16,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.test.test_utils import CustomTestCase


def _make_cfg() -> CompressedTensorsConfig:
    return CompressedTensorsConfig(
        target_scheme_map={"lm_head": {"weights": None, "input_activations": None}},
        ignore=[],
        quant_format="mixed-precision",
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
    )


class TestParallelLMHeadDispatch(CustomTestCase):
    def test_w8a8_fp8_lm_head_dispatches(self):
        # FP8 W8A8 per-channel lm_head (e.g. unsloth/Qwen3.8-27B-NVFP4) must
        # route through the linear scheme so weight_scale is applied.
        cfg = _make_cfg()
        sentinel = mock.MagicMock(spec=CompressedTensorsW8A8Fp8)
        cfg.get_linear_scheme = mock.MagicMock(return_value=sentinel)
        layer = ParallelLMHead.__new__(ParallelLMHead)

        method = cfg.get_quant_method(layer, prefix="lm_head")

        self.assertIsInstance(method, CompressedTensorsLinearMethod)
        self.assertIs(layer.scheme, sentinel)
        cfg.get_linear_scheme.assert_called_once_with(layer=layer, layer_name="lm_head")

    def test_wna16_lm_head_dispatches(self):
        # WNA16 pack-quantized lm_head (the case covered by #28130) must also
        # dispatch, keeping behavior consistent with vLLM #37291.
        cfg = _make_cfg()
        sentinel = mock.MagicMock(spec=CompressedTensorsWNA16)
        cfg.get_linear_scheme = mock.MagicMock(return_value=sentinel)
        layer = ParallelLMHead.__new__(ParallelLMHead)

        method = cfg.get_quant_method(layer, prefix="lm_head")

        self.assertIsInstance(method, CompressedTensorsLinearMethod)
        self.assertIs(layer.scheme, sentinel)

    def test_untargeted_lm_head_stays_unquantized(self):
        # A ValueError from target matching means the head is not covered by
        # any config group; keep the previous unquantized behavior.
        cfg = _make_cfg()
        cfg.get_linear_scheme = mock.MagicMock(side_effect=ValueError("Not targeted"))
        layer = ParallelLMHead.__new__(ParallelLMHead)

        method = cfg.get_quant_method(layer, prefix="lm_head")

        self.assertIsNone(method)
        self.assertFalse(hasattr(layer, "scheme"))

    def test_none_scheme_stays_unquantized(self):
        cfg = _make_cfg()
        cfg.get_linear_scheme = mock.MagicMock(return_value=None)
        layer = ParallelLMHead.__new__(ParallelLMHead)

        method = cfg.get_quant_method(layer, prefix="lm_head")

        self.assertIsNone(method)
        self.assertFalse(hasattr(layer, "scheme"))


if __name__ == "__main__":
    unittest.main()
