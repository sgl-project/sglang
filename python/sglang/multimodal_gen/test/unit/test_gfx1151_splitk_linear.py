# SPDX-License-Identifier: Apache-2.0
"""gfx1151 large-K bf16 linears accumulated over K chunks.

The thresholds are patched down so the dispatch is exercised on any device; the
CU count and the K-dependent rocBLAS behaviour are not testable here.
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers import linear


class _Layer(nn.Module):
    def __init__(self, n: int, k: int, dtype=torch.bfloat16):
        super().__init__()
        self.prefix = "test.splitk"
        self.weight = nn.Parameter(torch.randn(n, k, dtype=dtype), requires_grad=False)


class TestGfx1151SplitKLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self._patches = [
            patch.object(linear, "_SPLITK_PARTS", 4),
            patch.object(linear, "_SPLITK_MIN_M", 8),
            patch.object(linear, "_SPLITK_MIN_K", 8),
        ]
        for p in self._patches:
            p.start()
        self.method = linear.UnquantizedLinearMethod()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _inputs(self, m: int, n: int, k: int):
        x = torch.randn(1, m, k, dtype=torch.bfloat16)
        return x, torch.randn(n, k, dtype=torch.bfloat16)

    def test_engages_and_matches_fp32(self):
        x, w = self._inputs(m=16, n=8, k=16)
        layer = _Layer(8, 16)
        layer.weight.data = w.clone()
        with patch.object(linear, "splitk_linear", wraps=linear.splitk_linear) as run:
            out = self.method.apply(layer, x)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(out.shape, (1, 16, 8))
        ref = F.linear(x.float(), w.float())
        torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=2e-2)

    def test_rebuilds_chunks_when_the_weight_storage_changes(self):
        layer = _Layer(8, 16)
        x, _ = self._inputs(m=16, n=8, k=16)
        self.method.apply(layer, x)
        first = layer._splitk_chunks[1][0]

        replacement = _Layer(8, 16).weight.detach()
        layer.weight.data = replacement
        self.method.apply(layer, x)
        second = layer._splitk_chunks[1][0]

        self.assertIsNot(first, second)
        torch.testing.assert_close(second, replacement[:, :4].t())

    def test_small_m_keeps_the_single_gemm(self):
        x, w = self._inputs(m=4, n=8, k=16)
        layer = _Layer(8, 16)
        layer.weight.data = w.clone()
        with patch.object(linear, "splitk_linear") as run:
            out = self.method.apply(layer, x)
        run.assert_not_called()
        torch.testing.assert_close(out, F.linear(x, w))

    def test_bias_and_non_bf16_keep_the_single_gemm(self):
        x, w = self._inputs(m=16, n=8, k=16)
        layer = _Layer(8, 16)
        layer.weight.data = w.clone()
        bias = torch.randn(8, dtype=torch.bfloat16)
        with patch.object(linear, "splitk_linear") as run:
            out = self.method.apply(layer, x, bias=bias)
        run.assert_not_called()
        torch.testing.assert_close(out, F.linear(x, w, bias))

        half = x.half()
        half_layer = _Layer(8, 16, dtype=torch.float16)
        with patch.object(linear, "splitk_linear") as run:
            out = self.method.apply(half_layer, half)
        run.assert_not_called()
        torch.testing.assert_close(out, F.linear(half, half_layer.weight))

    def test_k_not_divisible_by_the_chunk_count_falls_back(self):
        x, w = self._inputs(m=16, n=8, k=14)
        layer = _Layer(8, 14)
        layer.weight.data = w.clone()
        with patch.object(linear, "splitk_linear") as run:
            out = self.method.apply(layer, x)
        run.assert_not_called()
        torch.testing.assert_close(out, F.linear(x, w))


if __name__ == "__main__":
    unittest.main()
