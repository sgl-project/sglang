"""CPU regression for the round-robin DSA prefill-CP RoPE positions.

The Blackwell trtllm DSA-prefill path applies RoPE inside the backend and must
use the rank's *global* token positions, not the full forward_batch.positions.
This guards the inverse-map invariant the fix relies on (token i keeps its true
sequence index i even though it lives at local slot i // cp_size), independent
of any GPU kernel.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-b")

from sglang.srt.layers.attention.dsa import utils as dsa_utils


def _make_forward_batch(num_tokens: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(positions=torch.arange(num_tokens, dtype=torch.int64))


class TestDSACPRoundRobinPositions(CustomTestCase):
    def _rope_positions_for_rank(self, num_tokens: int, cp_rank: int, cp_size: int):
        fb = _make_forward_batch(num_tokens)
        with mock.patch.object(
            dsa_utils, "get_attention_cp_rank", return_value=cp_rank
        ), mock.patch.object(
            dsa_utils, "get_attention_cp_size", return_value=cp_size
        ):
            return dsa_utils.dsa_cp_round_robin_rope_positions(fb)

    def test_each_rank_gets_its_global_positions(self):
        # token i -> rank i % cp_size; the positions returned for that rank must
        # be exactly [r, r + cp_size, r + 2*cp_size, ...] (the true global ids).
        for cp_size in (2, 4, 8):
            for num_tokens in (cp_size, cp_size * 3, cp_size * 7):
                for cp_rank in range(cp_size):
                    out = self._rope_positions_for_rank(num_tokens, cp_rank, cp_size)
                    expected = torch.arange(cp_rank, num_tokens, cp_size)
                    self.assertTrue(
                        torch.equal(out, expected),
                        f"cp_size={cp_size} rank={cp_rank} tokens={num_tokens}: "
                        f"{out.tolist()} != {expected.tolist()}",
                    )

    def test_ranks_partition_full_sequence(self):
        # The concatenation of all ranks' positions must be a permutation of the
        # full position range (no token dropped, none duplicated).
        for cp_size in (2, 4, 8):
            num_tokens = cp_size * 5
            gathered = torch.cat(
                [
                    self._rope_positions_for_rank(num_tokens, r, cp_size)
                    for r in range(cp_size)
                ]
            )
            self.assertEqual(
                sorted(gathered.tolist()), list(range(num_tokens))
            )

    def test_matches_qk_split_ordering(self):
        # Positions must split with the SAME ordering as the q/k rows
        # (dsa_cp_round_robin_split_data), or RoPE misaligns and produces garbage.
        cp_size = 4
        num_tokens = cp_size * 6
        fb = _make_forward_batch(num_tokens)
        fake_qk = torch.arange(num_tokens, dtype=torch.int64).view(num_tokens, 1)
        for cp_rank in range(cp_size):
            with mock.patch.object(
                dsa_utils, "get_attention_cp_rank", return_value=cp_rank
            ), mock.patch.object(
                dsa_utils, "get_attention_cp_size", return_value=cp_size
            ):
                pos = dsa_utils.dsa_cp_round_robin_rope_positions(fb)
                qk = dsa_utils.dsa_cp_round_robin_split_data(fake_qk)
            self.assertTrue(torch.equal(pos, qk.view(-1)))


if __name__ == "__main__":
    unittest.main()
