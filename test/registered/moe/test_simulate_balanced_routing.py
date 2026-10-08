import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import _simulate_balanced_routing
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")

NUM_EXPERTS = 256
TOPK = 8


class TestSimulateBalancedRouting(CustomTestCase):
    """SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS must not leave EP ranks idle.

    Spacing each token's slots by num_experts // topk balances ranks only while
    num_tokens * topk >= num_experts. Below that, a rank whose expert block is
    narrower than the spacing is never selected -- at num_experts=256, topk=8,
    EP16 every odd rank drew zero tokens for any batch under 32.
    """

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/HIP not available")

    def _route(self, num_tokens, num_ranks, *, random=False, topk=TOPK):
        ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device="cuda")
        weights = torch.zeros(num_tokens, topk, dtype=torch.float32, device="cuda")
        with patch(
            "sglang.srt.layers.moe.topk._balanced_routing_num_ranks",
            return_value=num_ranks,
        ):
            _simulate_balanced_routing(ids, weights, NUM_EXPERTS, random=random, seed=0)
        return ids, weights

    def _per_rank_counts(self, ids, num_ranks):
        block = NUM_EXPERTS // num_ranks
        return torch.bincount(
            ids.flatten().to(torch.int64) // block, minlength=num_ranks
        ).tolist()

    def test_every_rank_draws_the_same_token_count(self):
        for num_ranks in (1, 2, 4, 8, 16):
            for num_tokens in (1, 2, 4, 8, 16, 32, 300):
                with self.subTest(num_ranks=num_ranks, num_tokens=num_tokens):
                    ids, _ = self._route(num_tokens, num_ranks)
                    counts = self._per_rank_counts(ids, num_ranks)
                    assignments = num_tokens * TOPK
                    self.assertEqual(sum(counts), assignments)
                    # Off by at most one only when the assignments cannot divide
                    # across the ranks; exactly equal otherwise.
                    self.assertLessEqual(max(counts) - min(counts), 1)
                    if assignments % num_ranks == 0:
                        self.assertEqual(min(counts), assignments // num_ranks)

    def test_activated_expert_count_and_weights(self):
        for num_tokens in (1, 4, 32, 300):
            with self.subTest(num_tokens=num_tokens):
                ids, weights = self._route(num_tokens, 16)
                self.assertEqual(
                    ids.unique().numel(), min(num_tokens * TOPK, NUM_EXPERTS)
                )
                torch.testing.assert_close(
                    weights, torch.full_like(weights, 1.0 / TOPK)
                )

    def test_each_token_gets_distinct_experts(self):
        for num_ranks in (1, 4, 16):
            with self.subTest(num_ranks=num_ranks):
                ids, _ = self._route(64, num_ranks)
                for row in ids:
                    self.assertEqual(row.unique().numel(), TOPK)

    def test_ids_stay_in_range(self):
        for num_ranks in (1, 4, 16):
            with self.subTest(num_ranks=num_ranks):
                ids, _ = self._route(300, num_ranks)
                self.assertGreaterEqual(int(ids.min()), 0)
                self.assertLess(int(ids.max()), NUM_EXPERTS)

    def test_uniform_is_flat_across_ranks_in_expectation(self):
        # The random path keeps its own spacing; assert only that it does not
        # systematically starve a rank.
        ids, weights = self._route(4096, 8, random=True)
        counts = self._per_rank_counts(ids, 8)
        ideal = 4096 * TOPK // 8
        self.assertLess(max(counts) - min(counts), ideal // 8)
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / TOPK))


if __name__ == "__main__":
    unittest.main()
