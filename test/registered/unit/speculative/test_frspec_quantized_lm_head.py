# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for FR-Spec (``--speculative-token-map``) over a quantized target lm_head.

SGLang keeps a W8A8 FP8 (per-channel) lm_head weight transposed (``[K, V]``) with a ``[V, 1]``
scale. The old ``head.data[hot_token_id]`` sliced the K axis (OOM at startup on large vocabularies)
and left the scale full size. ``prune_quantized_lm_head`` must slice weight and scale along the
vocab axis, keep the original layout, leave the target head untouched, return ``None`` for
unquantized heads (old path) and refuse layouts it cannot slice per row.
"""

import unittest

import torch

from sglang.srt.speculative.spec_utils import prune_quantized_lm_head
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fp8_head(w_vk: torch.Tensor, scale_v1: torch.Tensor) -> torch.nn.Module:
    """Same layout CompressedTensorsW8A8Fp8 leaves after process_weights_after_loading."""
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(
        w_vk.t(), requires_grad=False
    )  # [K, V], stride (1, K)
    m.weight_scale = torch.nn.Parameter(scale_v1, requires_grad=False)  # [V, 1]
    m.input_scale = None
    v = w_vk.shape[0]
    m.num_embeddings = m.num_embeddings_padded = m.num_embeddings_per_partition = v
    m.org_vocab_size = m.org_vocab_size_padded = v
    m.logical_widths = [v]
    return m


def _logits(x: torch.Tensor, head: torch.nn.Module) -> torch.Tensor:
    w = head.weight.float().t() * head.weight_scale.float()  # dequantized [V, K]
    return x.float() @ w.t()


class TestFRSpecQuantizedLmHead(CustomTestCase):
    V, K, N_HOT = 4096, 256, 640

    def setUp(self):
        torch.manual_seed(0)
        w = (torch.randn(self.V, self.K) * 0.05).to(torch.float8_e4m3fn)
        s = torch.rand(self.V, 1) * 0.01 + 1e-3
        self.head = _fp8_head(w, s)
        self.hot = torch.randperm(self.V)[: self.N_HOT].sort().values

    def test_prunes_weight_and_scale_along_vocab(self):
        p = prune_quantized_lm_head(self.head, self.hot)
        self.assertIsNot(p, self.head)
        self.assertEqual(tuple(p.weight.shape), (self.K, self.N_HOT))
        self.assertEqual(p.weight.stride(), (1, self.K))  # same transposed layout
        self.assertEqual(p.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(p.weight_scale.shape), (self.N_HOT, 1))
        self.assertEqual(p.num_embeddings, self.N_HOT)
        self.assertEqual(p.logical_widths, [self.N_HOT])

    def test_logits_match_hot_rows_of_full_head(self):
        p = prune_quantized_lm_head(self.head, self.hot)
        x = torch.randn(4, self.K)
        self.assertTrue(torch.equal(_logits(x, self.head)[:, self.hot], _logits(x, p)))

    def test_target_head_untouched(self):
        before_w, before_s = self.head.weight.clone(), self.head.weight_scale.clone()
        prune_quantized_lm_head(self.head, self.hot)
        self.assertEqual(tuple(self.head.weight.shape), (self.K, self.V))
        self.assertTrue(
            torch.equal(self.head.weight.view(torch.uint8), before_w.view(torch.uint8))
        )
        self.assertTrue(torch.equal(self.head.weight_scale, before_s))
        self.assertEqual(self.head.num_embeddings, self.V)

    def test_unquantized_head_keeps_old_path(self):
        h = torch.nn.Module()
        h.weight = torch.nn.Parameter(torch.randn(self.V, self.K, dtype=torch.bfloat16))
        self.assertIsNone(prune_quantized_lm_head(h, self.hot))

    def test_block_scales_are_refused(self):
        self.head.weight_scale = torch.nn.Parameter(
            torch.rand(self.V // 128, self.K // 128), requires_grad=False
        )
        with self.assertRaises(NotImplementedError):
            prune_quantized_lm_head(self.head, self.hot)

    def test_out_of_range_ids_are_refused(self):
        bad = torch.tensor([0, self.V])
        with self.assertRaises(ValueError):
            prune_quantized_lm_head(self.head, bad)


if __name__ == "__main__":
    unittest.main()
