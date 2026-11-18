"""The AITER MXFP4 MoE intermediate padding follows the layer's declared alignment."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import mxfp4
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Kimi-K3 at TP8: moe_intermediate_size 3072 / 8.
_INTERMEDIATE = 384
_HIDDEN = 256


def _aiter_method() -> Mxfp4MoEMethod:
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_deep_gemm = False
    method.use_mega_moe = False
    method._fi_kernel = None
    method.use_aiter_moe = True
    return method


def _layer(align_128: bool) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.hidden_size = _HIDDEN
    layer.intermediate_size_per_partition = _INTERMEDIATE
    layer.num_local_experts = 2
    layer.aiter_mxfp4_intermediate_align_128 = align_128
    return layer


class TestAiterMxfp4IntermediateAlign(CustomTestCase):
    def _padded(self, align_128: bool) -> int:
        method = _aiter_method()
        with mock.patch.object(mxfp4, "_use_aiter", True):
            method.create_weights(
                layer=_layer(align_128),
                num_experts=2,
                hidden_size=_HIDDEN,
                intermediate_size_per_partition=_INTERMEDIATE,
                params_dtype=torch.bfloat16,
            )
        return method.intermediate_size_per_partition

    def test_default_pads_to_256(self):
        """Models that declare nothing keep AITER's generic 256 alignment."""
        self.assertEqual(self._padded(align_128=False), 512)

    def test_declared_128_alignment_keeps_the_native_width(self):
        self.assertEqual(self._padded(align_128=True), 384)


if __name__ == "__main__":
    unittest.main()
