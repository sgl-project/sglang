"""CPU-only regression tests for fused DSA indexer wk/weights_proj loading.

_load_fused_indexer_wk folds indexer.wk + indexer.weights_proj into the bf16
wk_weights_proj param. Channelwise-quantized checkpoints (w8a8 int8, fp8
channelwise) name the per-channel scale `weight_scale` instead of the block-fp8
suffix `weight_scale_inv`; the loader must pair it with the weight and
dequantize, not copy it over the wk rows as a plain weight.
"""

import unittest

import torch

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _load_fused_indexer_wk,
)

PREFIX = "model.layers.0.self_attn.indexer"
FUSED_NAME = PREFIX + ".wk_weights_proj.weight"
HEAD_DIM = 128
N_HEADS = 32
IN_FEATURES = 256


class TestFusedIndexerWkLoader(unittest.TestCase):
    def _make_params(self):
        return {
            FUSED_NAME: torch.zeros(
                HEAD_DIM + N_HEADS, IN_FEATURES, dtype=torch.bfloat16
            )
        }

    def _load(self, params, pending, name, weight, quant_config=None):
        return _load_fused_indexer_wk(name, weight, params, pending, quant_config)

    def test_int8_per_channel_scale_dequantized(self):
        params = self._make_params()
        pending = {}
        wk_int8 = torch.randint(-127, 128, (HEAD_DIM, IN_FEATURES), dtype=torch.int8)
        scale = torch.rand(HEAD_DIM, 1) * 0.02 + 0.001
        # Checkpoint load order: weight first, then weight_scale.
        self.assertTrue(self._load(params, pending, PREFIX + ".wk.weight", wk_int8))
        self.assertTrue(self._load(params, pending, PREFIX + ".wk.weight_scale", scale))
        ref = (wk_int8.float() * scale.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(params[FUSED_NAME][:HEAD_DIM], ref))
        self.assertEqual(pending, {})

    def test_int8_scale_loaded_before_weight(self):
        params = self._make_params()
        pending = {}
        wk_int8 = torch.randint(-127, 128, (HEAD_DIM, IN_FEATURES), dtype=torch.int8)
        scale = torch.rand(HEAD_DIM, 1) * 0.02 + 0.001
        self.assertTrue(self._load(params, pending, PREFIX + ".wk.weight_scale", scale))
        self.assertTrue(self._load(params, pending, PREFIX + ".wk.weight", wk_int8))
        ref = (wk_int8.float() * scale.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(params[FUSED_NAME][:HEAD_DIM], ref))
        self.assertEqual(pending, {})

    def test_fp8_channelwise_scale_dequantized(self):
        params = self._make_params()
        pending = {}
        wk_fp8 = (torch.rand(HEAD_DIM, IN_FEATURES) * 2 - 1).to(torch.float8_e4m3fn)
        scale = torch.rand(HEAD_DIM, 1) * 0.02 + 0.001
        self.assertTrue(self._load(params, pending, PREFIX + ".wk.weight", wk_fp8))
        self.assertTrue(self._load(params, pending, PREFIX + ".wk.weight_scale", scale))
        ref = (wk_fp8.float() * scale.float()).to(torch.bfloat16)
        self.assertTrue(torch.equal(params[FUSED_NAME][:HEAD_DIM], ref))
        self.assertEqual(pending, {})

    def test_block_fp8_weight_scale_inv_uses_block_dequant(self):
        params = self._make_params()
        pending = {}
        block = [128, 128]
        wk_fp8 = (torch.rand(HEAD_DIM, IN_FEATURES) * 2 - 1).to(torch.float8_e4m3fn)
        scale = torch.rand(HEAD_DIM // block[0], IN_FEATURES // block[1]) + 0.5
        self.assertTrue(self._load(params, pending, PREFIX + ".wk.weight", wk_fp8))
        self.assertTrue(
            self._load(params, pending, PREFIX + ".wk.weight_scale_inv", scale)
        )
        scale_repeat = scale.repeat_interleave(block[0], dim=-2).repeat_interleave(
            block[1], dim=-1
        )[:HEAD_DIM, :IN_FEATURES]
        ref = (wk_fp8.float() * scale_repeat).to(torch.bfloat16)
        self.assertTrue(torch.equal(params[FUSED_NAME][:HEAD_DIM], ref))
        self.assertEqual(pending, {})

    def test_bf16_wk_copied_directly(self):
        params = self._make_params()
        wk = torch.randn(HEAD_DIM, IN_FEATURES, dtype=torch.bfloat16)
        self.assertTrue(self._load(params, {}, PREFIX + ".wk.weight", wk))
        self.assertTrue(torch.equal(params[FUSED_NAME][:HEAD_DIM], wk))

    def test_weights_proj_fills_bottom_rows(self):
        params = self._make_params()
        wp = torch.randn(N_HEADS, IN_FEATURES, dtype=torch.bfloat16)
        self.assertTrue(self._load(params, {}, PREFIX + ".weights_proj.weight", wp))
        self.assertTrue(torch.equal(params[FUSED_NAME][-N_HEADS:], wp))
        self.assertTrue(torch.all(params[FUSED_NAME][:HEAD_DIM] == 0))

    def test_returns_false_without_fused_param(self):
        # Unfused mode (SGLANG_DISABLE_DSA_INDEXER_FUSION=1): no fused param,
        # the caller must fall through to per-tensor loading.
        wk = torch.randn(HEAD_DIM, IN_FEATURES, dtype=torch.bfloat16)
        self.assertFalse(self._load({}, {}, PREFIX + ".wk.weight", wk))
        self.assertFalse(self._load({}, {}, PREFIX + ".wk.weight_scale", wk))


if __name__ == "__main__":
    unittest.main()
