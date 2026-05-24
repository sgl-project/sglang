"""Unit tests for GPTQ-to-BF16 dequantization used by DeepSeek V4
when loading AutoRound/GPTQ checkpoints with unquantized layers.

These tests verify the dequant logic without launching a model server.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestDequantGPTQWeight(CustomTestCase):
    """Verify _dequant_gptq_weight produces correct dense weights."""

    def _import_dequant(self):
        from sglang.srt.models.deepseek_v4 import _dequant_gptq_weight

        return _dequant_gptq_weight

    def test_roundtrip_identity(self):
        """Pack a known dense weight into GPTQ format and verify dequant recovers it."""
        _dequant_gptq_weight = self._import_dequant()

        bits = 4
        group_size = 128
        in_size, out_size = 256, 64
        pack_factor = 32 // bits  # 8
        n_groups = in_size // group_size
        mask = (1 << bits) - 1

        torch.manual_seed(42)
        raw_uint = torch.randint(0, 1 << bits, (in_size, out_size), dtype=torch.int32)
        zeros_uint = torch.full((n_groups, out_size), 8, dtype=torch.int32)
        scales = torch.ones((n_groups, out_size), dtype=torch.float16)

        # Pack qweight along dim-0: row i of unpacked maps to
        #   qweight[i // pack_factor] |= (val << (bits * (i % pack_factor)))
        qweight = torch.zeros(in_size // pack_factor, out_size, dtype=torch.int32)
        for i in range(pack_factor):
            qweight |= (raw_uint[i::pack_factor] & mask) << (bits * i)

        # Pack qzeros along dim-1 with GPTQ v1 convention (store zeros - 1)
        zeros_v1 = zeros_uint - 1
        qzeros = torch.zeros(n_groups, out_size // pack_factor, dtype=torch.int32)
        for i in range(pack_factor):
            qzeros |= (zeros_v1[:, i::pack_factor] & mask) << (bits * i)

        result = _dequant_gptq_weight(qweight, qzeros, scales, bits, group_size)

        expected = (raw_uint.float() - 8.0).to(torch.bfloat16)

        self.assertEqual(result.shape, (in_size, out_size))
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.01)

    def test_output_shape_and_dtype(self):
        """Verify output shape and dtype for various configurations."""
        _dequant_gptq_weight = self._import_dequant()

        for bits in [4]:
            for group_size in [32, 128]:
                with self.subTest(bits=bits, group_size=group_size):
                    in_size, out_size = group_size * 4, 32
                    pack_factor = 32 // bits
                    n_groups = in_size // group_size

                    qw = torch.zeros(
                        in_size // pack_factor, out_size, dtype=torch.int32
                    )
                    qz = torch.zeros(
                        n_groups, out_size // pack_factor, dtype=torch.int32
                    )
                    sc = torch.ones(n_groups, out_size, dtype=torch.float16)

                    result = _dequant_gptq_weight(qw, qz, sc, bits, group_size)
                    self.assertEqual(result.shape, (in_size, out_size))
                    self.assertEqual(result.dtype, torch.bfloat16)


class TestDequantGPTQForUnquantLayers(CustomTestCase):
    """Verify _dequant_gptq_for_unquant_layers filters and dequantizes correctly."""

    def _import_fn(self):
        from sglang.srt.models.deepseek_v4 import _dequant_gptq_for_unquant_layers

        return _dequant_gptq_for_unquant_layers

    def _make_gptq_triplet(self, in_size, out_size, bits=4, group_size=128):
        pack_factor = 32 // bits
        n_groups = in_size // group_size
        qw = torch.zeros(in_size // pack_factor, out_size, dtype=torch.int32)
        qz = torch.zeros(n_groups, out_size // pack_factor, dtype=torch.int32)
        sc = torch.ones(n_groups, out_size, dtype=torch.float16)
        return qw, qz, sc

    def test_dequants_wo_a(self):
        fn = self._import_fn()
        qw, qz, sc = self._make_gptq_triplet(256, 64)
        weights = [
            ("model.layers.0.self_attn.wo_a.qweight", qw),
            ("model.layers.0.self_attn.wo_a.qzeros", qz),
            ("model.layers.0.self_attn.wo_a.scales", sc),
            ("model.layers.0.self_attn.wo_b.qweight", torch.zeros(1)),
        ]
        result = dict(fn(weights, bits=4, group_size=128))
        self.assertIn("model.layers.0.self_attn.wo_a.weight", result)
        self.assertNotIn("model.layers.0.self_attn.wo_a.qweight", result)
        self.assertNotIn("model.layers.0.self_attn.wo_a.qzeros", result)
        self.assertNotIn("model.layers.0.self_attn.wo_a.scales", result)
        # wo_b should pass through unchanged
        self.assertIn("model.layers.0.self_attn.wo_b.qweight", result)

    def test_dequants_compressor_wkv_and_wgate(self):
        fn = self._import_fn()
        qw, qz, sc = self._make_gptq_triplet(256, 64)
        weights = [
            ("model.layers.0.self_attn.compressor.wkv.qweight", qw),
            ("model.layers.0.self_attn.compressor.wkv.qzeros", qz),
            ("model.layers.0.self_attn.compressor.wkv.scales", sc),
            ("model.layers.0.self_attn.compressor.wgate.qweight", qw),
            ("model.layers.0.self_attn.compressor.wgate.qzeros", qz),
            ("model.layers.0.self_attn.compressor.wgate.scales", sc),
        ]
        result = dict(fn(weights, bits=4, group_size=128))
        self.assertIn("model.layers.0.self_attn.compressor.wkv.weight", result)
        self.assertIn("model.layers.0.self_attn.compressor.wgate.weight", result)
        self.assertEqual(len(result), 2)

    def test_dequants_weights_proj(self):
        fn = self._import_fn()
        qw, qz, sc = self._make_gptq_triplet(128, 32, group_size=128)
        weights = [
            ("model.layers.10.self_attn.indexer.weights_proj.qweight", qw),
            ("model.layers.10.self_attn.indexer.weights_proj.qzeros", qz),
            ("model.layers.10.self_attn.indexer.weights_proj.scales", sc),
        ]
        result = dict(fn(weights, bits=4, group_size=128))
        self.assertIn("model.layers.10.self_attn.indexer.weights_proj.weight", result)
        self.assertEqual(len(result), 1)

    def test_passthrough_other_weights(self):
        fn = self._import_fn()
        weights = [
            ("model.layers.0.self_attn.wq_b.qweight", torch.zeros(1)),
            ("model.layers.0.self_attn.wq_b.qzeros", torch.zeros(1)),
            ("model.layers.0.self_attn.wq_b.scales", torch.zeros(1)),
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.zeros(1)),
        ]
        result = dict(fn(weights, bits=4, group_size=128))
        self.assertEqual(len(result), 4)
        self.assertIn("model.layers.0.self_attn.wq_b.qweight", result)

    def test_drops_g_idx(self):
        fn = self._import_fn()
        qw, qz, sc = self._make_gptq_triplet(256, 64)
        weights = [
            ("model.layers.0.self_attn.wo_a.qweight", qw),
            ("model.layers.0.self_attn.wo_a.qzeros", qz),
            ("model.layers.0.self_attn.wo_a.scales", sc),
            ("model.layers.0.self_attn.wo_a.g_idx", torch.arange(256)),
        ]
        result = dict(fn(weights, bits=4, group_size=128))
        self.assertNotIn("model.layers.0.self_attn.wo_a.g_idx", result)
        self.assertIn("model.layers.0.self_attn.wo_a.weight", result)


if __name__ == "__main__":
    unittest.main()
