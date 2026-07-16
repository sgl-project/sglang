"""Unit tests for the seeded EAGLE verify coins (_seeded_verify_coins).

Locks the deterministic-coin contract behind seeded speculative sampling:
identical (seed, seq_lens) inputs produce bitwise-identical coins, distinct
seeds diverge, and the float32 conversion never emits a coin of exactly 1.0
(the sampling kernels expect half-open [0, 1) coins — a 1.0 coin walks past
the final CDF bucket and can return a zero-probability token).

Requires a GPU: the coins hash through the murmur_hash32 Triton kernel.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_utils import _seeded_verify_coins
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

DRAFT_TOKEN_NUM = 4


def _coins(seeds, seq_lens):
    device = "cuda"
    return _seeded_verify_coins(
        sampling_seed=torch.tensor(seeds, device=device, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int64),
        draft_token_num=DRAFT_TOKEN_NUM,
        device=device,
    )


class TestSeededVerifyCoins(CustomTestCase):
    def test_seeded_coins_are_reproducible(self):
        coins_a, final_a = _coins([12345, 67890, 12345], [7, 9, 7])
        coins_b, final_b = _coins([12345, 67890, 12345], [7, 9, 7])

        self.assertEqual(coins_a.shape, (3, DRAFT_TOKEN_NUM))
        self.assertEqual(final_a.shape, (3,))
        self.assertTrue(torch.equal(coins_a, coins_b))
        self.assertTrue(torch.equal(final_a, final_b))
        # Same (seed, seq_len) pair hashes to the same coins regardless of row.
        self.assertTrue(torch.equal(coins_a[0], coins_a[2]))
        self.assertEqual(final_a[0].item(), final_a[2].item())
        # Coins live in [0, 1).
        self.assertTrue(bool((coins_a >= 0).all() and (coins_a < 1).all()))
        self.assertTrue(bool((final_a >= 0).all() and (final_a < 1).all()))

    def test_distinct_seeds_or_positions_diverge(self):
        coins, final = _coins([12345, 67890, 12345], [7, 9, 11])
        self.assertFalse(torch.equal(coins[0], coins[1]))  # different seed
        self.assertFalse(torch.equal(coins[0], coins[2]))  # different seq_len

    def test_max_hash_clamps_coins_below_one(self):
        # The top 129 uint32 hashes round to exactly 1.0 under the float32
        # cast; force the worst case and assert the clamp holds the contract.
        umax = torch.iinfo(torch.uint32).max

        def _all_max_hash(seed, positions, col_indices):
            return torch.full(
                (seed.shape[0], col_indices.shape[0]),
                umax,
                dtype=torch.uint32,
                device=seed.device,
            )

        with patch(
            "sglang.srt.layers.utils.hash.murmur_hash32", side_effect=_all_max_hash
        ):
            coins, final = _coins([1, 2], [3, 4])

        self.assertTrue(bool((coins < 1).all()))
        self.assertTrue(bool((final < 1).all()))
        # Clamped to the largest float32 strictly below one.
        self.assertEqual(coins.max().item(), 1.0 - 2**-24)


if __name__ == "__main__":
    unittest.main()
