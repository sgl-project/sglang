"""Unit tests for MiMoV2 fused qkv_proj layout detection
(srt/configs/model_config.py: is_mimo_v2_fused_qkv_plain_layout) and the
plain-layout delegation in load_mimo_v2_qkv_proj_weight."""

import unittest

import pytest
import torch
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import is_mimo_v2_fused_qkv_plain_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMiMoV2QkvLayoutDetection(CustomTestCase):
    def test_default_is_interleaved(self):
        # Original Pro release: fused_qkv, no quantization_config.
        config = PretrainedConfig(attention_projection_layout="fused_qkv")
        self.assertFalse(is_mimo_v2_fused_qkv_plain_layout(config))

    def test_modelopt_quantization_config_implies_plain(self):
        # Community re-exports through transformers + ModelOpt serialize
        # qkv_proj in plain [Q;K;V] order (e.g. lukealonso/MiMo-V2.5-NVFP4).
        config = PretrainedConfig(
            attention_projection_layout="fused_qkv",
            quantization_config={"quant_method": "modelopt"},
        )
        self.assertTrue(is_mimo_v2_fused_qkv_plain_layout(config))

    def test_non_modelopt_quantization_config_stays_interleaved(self):
        config = PretrainedConfig(
            attention_projection_layout="fused_qkv",
            quantization_config={"quant_method": "fp8"},
        )
        self.assertFalse(is_mimo_v2_fused_qkv_plain_layout(config))

    def test_explicit_order_overrides_heuristic(self):
        # An explicit field wins over ModelOpt provenance, in both directions.
        config = PretrainedConfig(
            fused_qkv_weight_order="interleaved",
            quantization_config={"quant_method": "modelopt"},
        )
        self.assertFalse(is_mimo_v2_fused_qkv_plain_layout(config))

        config = PretrainedConfig(fused_qkv_weight_order="plain")
        self.assertTrue(is_mimo_v2_fused_qkv_plain_layout(config))

    def test_explicit_order_on_text_config(self):
        config = PretrainedConfig(
            text_config=PretrainedConfig(fused_qkv_weight_order="plain")
        )
        self.assertTrue(is_mimo_v2_fused_qkv_plain_layout(config))

    def test_invalid_order_raises(self):
        config = PretrainedConfig(fused_qkv_weight_order="zigzag")
        with self.assertRaises(ValueError):
            is_mimo_v2_fused_qkv_plain_layout(config)


class TestPlainLayoutLoaderDelegation(CustomTestCase):
    """load_mimo_v2_qkv_proj_weight must hand plain-layout fused tensors to
    the parameter's QKVParallelLinear weight loader instead of row-chunking."""

    @classmethod
    def setUpClass(cls):
        mimo_v2 = pytest.importorskip(
            "sglang.srt.models.mimo_v2",
            reason="mimo_v2 model module not importable in this environment",
        )
        cls.load_qkv = staticmethod(mimo_v2.load_mimo_v2_qkv_proj_weight)

    @staticmethod
    def _make_param(shape, loader=None):
        param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        if loader is not None:
            param.weight_loader = loader
        return param

    def test_plain_layout_delegates_full_tensor(self):
        calls = []
        param = self._make_param((10, 4), loader=lambda p, w: calls.append(w))
        fused = torch.randn(40, 4)  # rank shard 10 rows, fused 4x
        self.load_qkv("layer.qkv_proj.weight", param, fused, plain_layout=True)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0], fused))

    def test_plain_layout_without_loader_raises(self):
        param = self._make_param((10, 4))
        with self.assertRaisesRegex(ValueError, "plain \\[Q;K;V\\]"):
            self.load_qkv(
                "layer.qkv_proj.weight",
                param,
                torch.randn(40, 4),
                plain_layout=True,
            )

    def test_presharded_weight_short_circuits(self):
        # Already-sharded tensors copy directly regardless of layout flag.
        param = self._make_param((10, 4))
        shard = torch.randn(10, 4)
        self.load_qkv("layer.qkv_proj.weight", param, shard, plain_layout=True)
        self.assertTrue(torch.equal(param.data, shard))


if __name__ == "__main__":
    unittest.main()
