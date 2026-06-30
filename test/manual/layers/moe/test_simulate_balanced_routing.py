"""Unit tests for the fused benchmark-only balanced-routing override in
``sglang.srt.layers.moe.topk`` (``_simulate_balanced_routing`` /
``_simulate_balanced_routing_kernel``).

Verifies the single fused Triton kernel reproduces the
``_make_round_robin_expert_ids`` reference exactly (incl. the per-layer offset),
writes uniform ``1/k`` weights, and that the uniform path is structurally
balanced. GPU-only (skips without CUDA).

Run:
    python -m pytest test/manual/layers/moe/test_simulate_balanced_routing.py -v
"""

import unittest

import torch
from parameterized import parameterized

from sglang.srt.layers.moe.topk import (
    _make_round_robin_expert_ids,
    _simulate_balanced_routing,
)

E = 256  # num_experts
K = 8  # top-k


def _alloc(num_tokens, k, device="cuda"):
    # Pre-filled with junk so the test fails if the kernel doesn't overwrite.
    ids = torch.full((num_tokens, k), -7, dtype=torch.int32, device=device)
    weights = torch.full((num_tokens, k), -7.0, dtype=torch.float32, device=device)
    return ids, weights


class TestSimulateBalancedRouting(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    # round-robin output must equal the reference exactly, for several layer
    # offsets and both a power-of-2 and a non-power-of-2 top-k (BLOCK_K masking).
    @parameterized.expand(
        [
            ("layer0_k8", 0, 8),
            ("layer5_k8", 5, 8),
            ("noneLayer_k8", None, 8),
            ("layer3_k6", 3, 6),
        ]
    )
    def test_round_robin_matches_reference(self, _name, layer_id, k):
        T = 512
        ids, weights = _alloc(T, k)
        _simulate_balanced_routing(ids, weights, E, random=False, layer_id=layer_id)
        ref = _make_round_robin_expert_ids(
            T, k, E, device="cuda", dtype=torch.int32, layer_id=layer_id
        )
        self.assertTrue(torch.equal(ids, ref))
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / k))

    def test_round_robin_perfectly_balanced(self):
        T = 512  # multiple of E -> exactly uniform per-expert load
        ids, weights = _alloc(T, K)
        _simulate_balanced_routing(ids, weights, E, random=False, layer_id=0)
        counts = torch.bincount(ids.flatten().long(), minlength=E)
        self.assertTrue(torch.all(counts == (T * K // E)))
        for row in ids:
            self.assertEqual(row.unique().numel(), K)

    def test_uniform_structural(self):
        # uniform: random per-token base, so assert only seed-independent props.
        T = 4096
        ids, weights = _alloc(T, K)
        _simulate_balanced_routing(ids, weights, E, random=True, layer_id=0)
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / K))
        self.assertGreaterEqual(int(ids.min()), 0)
        self.assertLess(int(ids.max()), E)
        # offset + j*step spreads the k experts out -> k distinct per row
        for row in ids[:64]:
            self.assertEqual(row.unique().numel(), K)


if __name__ == "__main__":
    unittest.main()
