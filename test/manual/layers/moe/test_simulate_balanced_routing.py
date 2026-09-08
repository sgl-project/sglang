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
from typing import Optional, Tuple

import torch
from parameterized import parameterized

from sglang.srt.layers.moe.topk import _simulate_balanced_routing
from sglang.test.test_utils import CustomTestCase

E = 256  # num_experts
K = 8  # top-k


def _make_round_robin_expert_ids(
    num_tokens: int,
    topk: int,
    num_experts: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    layer_id: Optional[int] = None,
) -> torch.Tensor:
    # Deterministic, perfectly balanced expert assignment: each token's top-k is
    # spread by num_experts//topk. Returns global expert ids of shape
    # [num_tokens, topk].
    if topk == 0:
        return torch.empty((num_tokens, 0), device=device, dtype=dtype)

    step = max(num_experts // topk, 1)
    layer_offset = 0 if layer_id is None else layer_id
    offsets = torch.arange(num_tokens, device=device, dtype=dtype).unsqueeze(
        1
    )  # [num_tokens, 1]
    steps = (
        torch.arange(topk, device=device, dtype=dtype).unsqueeze(0) * step
    )  # [1, topk]
    return (offsets + layer_offset + steps) % num_experts  # [num_tokens, topk]


def _alloc(
    num_tokens: int, k: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Pre-filled with junk so the test fails if the kernel doesn't overwrite.
    ids = torch.full((num_tokens, k), -7, dtype=torch.int32, device=device)
    weights = torch.full((num_tokens, k), -7.0, dtype=torch.float32, device=device)
    return ids, weights


class TestSimulateBalancedRouting(CustomTestCase):
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
    def test_round_robin_matches_reference(
        self, _name: str, layer_id: Optional[int], k: int
    ) -> None:
        T = 512
        ids, weights = _alloc(T, k)
        _simulate_balanced_routing(ids, weights, E, random=False, layer_id=layer_id)
        ref = _make_round_robin_expert_ids(
            T, k, E, device="cuda", dtype=torch.int32, layer_id=layer_id
        )
        self.assertTrue(torch.equal(ids, ref))
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / k))

    def test_round_robin_perfectly_balanced(self) -> None:
        T = 512  # multiple of E -> exactly uniform per-expert load
        ids, weights = _alloc(T, K)
        _simulate_balanced_routing(ids, weights, E, random=False, layer_id=0)
        counts = torch.bincount(ids.flatten().long(), minlength=E)
        self.assertTrue(torch.all(counts == (T * K // E)))
        for row in ids:
            self.assertEqual(row.unique().numel(), K)

    def test_uniform_structural(self) -> None:
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

    @parameterized.expand(
        [
            ("round_robin_dp2", False, 2),
            ("round_robin_dp4", False, 4),
            ("uniform_dp2", True, 2),
            ("uniform_dp4", True, 4),
        ]
    )
    def test_interleaved_dp_assignments_match_dp1(
        self, _name: str, random: bool, dp_size: int
    ) -> None:
        # Interleaving the DP-local outputs must exactly reproduce the expert
        # assignments for the equivalent DP=1 input. The fixed seed models
        # independent processes entering the same uniform-routing call with
        # the same initial seed.
        T = 16
        seed = 17
        layer_id = 3
        ids_by_rank = []
        weights_by_rank = []
        for dp_rank in range(dp_size):
            ids, weights = _alloc(T, K)
            _simulate_balanced_routing(
                ids,
                weights,
                E,
                random=random,
                layer_id=layer_id,
                token_shard_rank=dp_rank,
                num_token_shards=dp_size,
                seed=seed,
            )
            ids_by_rank.append(ids)
            weights_by_rank.append(weights)

        interleaved_ids = torch.stack(ids_by_rank, dim=1).reshape(T * dp_size, K)
        interleaved_weights = torch.stack(weights_by_rank, dim=1).reshape(
            T * dp_size, K
        )

        expected_ids, expected_weights = _alloc(T * dp_size, K)
        _simulate_balanced_routing(
            expected_ids,
            expected_weights,
            E,
            random=random,
            layer_id=layer_id,
            seed=seed,
        )

        self.assertTrue(torch.equal(interleaved_ids, expected_ids))
        torch.testing.assert_close(interleaved_weights, expected_weights)


if __name__ == "__main__":
    unittest.main()
