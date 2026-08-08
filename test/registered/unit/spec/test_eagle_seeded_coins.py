"""Unit tests for the EAGLE verify coins (_verify_coins / _seeded_verify_coins).

Locks the deterministic-coin contract behind seeded speculative sampling:
identical (seed, seq_lens) inputs produce bitwise-identical coins, distinct
seeds diverge, the column split (first draft_token_num columns -> rejection
coins, last column -> final-sampling coin) holds, unseeded requests keep
torch.rand, and the float32 conversion never emits a coin of exactly 1.0
(the sampling kernels expect half-open [0, 1) coins — a 1.0 coin walks past
the final CDF bucket and can return a zero-probability token).

Requires a GPU: the coins hash through the murmur_hash32 Triton kernel.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_utils import _seeded_verify_coins, _verify_coins
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

    def test_column_split_maps_rejection_then_final(self):
        # Structured hash: hashed[i, j] = i * 1000 + j, so each coin names its
        # (row, column) origin. Locks the column-space contract: columns
        # [0, draft_token_num) become the per-draft rejection coins and column
        # draft_token_num becomes the final-sampling coin.
        umax = torch.iinfo(torch.uint32).max

        def _structured_hash(seed, positions, col_indices):
            rows = torch.arange(seed.shape[0], device=seed.device).unsqueeze(1)
            return (rows * 1000 + col_indices.unsqueeze(0)).to(torch.uint32)

        with patch(
            "sglang.kernels.ops.sampling.murmur_hash.murmur_hash32",
            side_effect=_structured_hash,
        ):
            coins, final = _coins([1, 2], [3, 4])

        def _expected(row, col):
            return (
                torch.tensor(row * 1000 + col, dtype=torch.float64)
                .div(umax)
                .to(torch.float32)
                .item()
            )

        for row in range(2):
            for col in range(DRAFT_TOKEN_NUM):
                self.assertEqual(coins[row, col].item(), _expected(row, col))
            self.assertEqual(final[row].item(), _expected(row, DRAFT_TOKEN_NUM))

    def test_unseeded_requests_keep_torch_rand(self):
        device = "cuda"
        kwargs = dict(
            sampling_info=SimpleNamespace(sampling_seed=None),
            seq_lens=torch.tensor([3, 4, 5], device=device, dtype=torch.int64),
            draft_token_num=DRAFT_TOKEN_NUM,
            candidates=torch.zeros(
                (3, DRAFT_TOKEN_NUM), device=device, dtype=torch.int64
            ),
            device=device,
        )
        with patch(
            "sglang.kernels.ops.sampling.murmur_hash.murmur_hash32"
        ) as mock_hash:
            coins_a, final_a = _verify_coins(**kwargs)
            coins_b, final_b = _verify_coins(**kwargs)

        mock_hash.assert_not_called()
        self.assertEqual(coins_a.shape, (3, DRAFT_TOKEN_NUM))
        self.assertEqual(final_a.shape, (3,))
        self.assertEqual(coins_a.dtype, torch.float32)
        # torch.rand draws: two calls must not repeat.
        self.assertFalse(torch.equal(coins_a, coins_b))
        self.assertFalse(torch.equal(final_a, final_b))

    def test_seeded_requests_dispatch_to_seeded_coins(self):
        device = "cuda"
        seeds = torch.tensor([12345, 67890], device=device, dtype=torch.int64)
        seq_lens = torch.tensor([7, 9], device=device, dtype=torch.int64)
        coins, final = _verify_coins(
            sampling_info=SimpleNamespace(sampling_seed=seeds),
            seq_lens=seq_lens,
            draft_token_num=DRAFT_TOKEN_NUM,
            candidates=torch.zeros(
                (2, DRAFT_TOKEN_NUM), device=device, dtype=torch.int64
            ),
            device=device,
        )
        expected_coins, expected_final = _seeded_verify_coins(
            sampling_seed=seeds,
            seq_lens=seq_lens,
            draft_token_num=DRAFT_TOKEN_NUM,
            device=device,
        )
        self.assertTrue(torch.equal(coins, expected_coins))
        self.assertTrue(torch.equal(final, expected_final))

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
            "sglang.kernels.ops.sampling.murmur_hash.murmur_hash32",
            side_effect=_all_max_hash,
        ):
            coins, final = _coins([1, 2], [3, 4])

        self.assertTrue(bool((coins < 1).all()))
        self.assertTrue(bool((final < 1).all()))
        # Clamped to the largest float32 strictly below one.
        self.assertEqual(coins.max().item(), 1.0 - 2**-24)


if __name__ == "__main__":
    unittest.main()
