"""CPU unit test for the invariant behind DCP decode on a DCP-unaware kernel.

A ``q_len == 1`` decode needs no in-kernel DCP support. The kernel-side DCP
arguments (``cp_world`` / ``cp_rank`` / ``causal_seqlens_kv_global``) exist so a
kernel can resolve a per-query global causal bound, and that bound only varies
across query rows when ``q_len > 1``. With a single query token, the cyclic
owner rule already partitions exactly the causally visible prefix, so rank-local
lengths plus a rank-local page table describe the shard completely.

This pins the two properties that argument rests on:

1. Partition: the per-rank lengths sum to the global length, so no token is
   dropped or double counted by the cross-rank merge.
2. Causal exactness: a rank's length equals the number of positions it owns in
   ``[0, global_len)``, i.e. every token it holds is causally visible to the
   token at position ``global_len - 1`` and none is missing.

Usage:
    python -m pytest test_dcp_decode_partition_unit.py -v
    python test_dcp_decode_partition_unit.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 8]
GLOBAL_LENS = list(range(0, 65))


class TestDcpDecodePartition(CustomTestCase):
    def test_ranks_partition_the_global_length(self):
        lens = torch.tensor(GLOBAL_LENS, dtype=torch.int32)
        for n in DCP_SIZES:
            with self.subTest(dcp_size=n):
                total = sum(
                    get_dcp_lens(lens, n, rank).to(torch.int64) for rank in range(n)
                )
                self.assertTrue(
                    torch.equal(total, lens.to(torch.int64)),
                    f"per-rank lengths do not sum to the global length at n={n}",
                )

    def test_rank_length_matches_causally_visible_owned_positions(self):
        """The decoding token sits at position global_len - 1 and attends to
        [0, global_len). A rank must hold exactly the owned subset of that."""
        for n in DCP_SIZES:
            for rank in range(n):
                with self.subTest(dcp_size=n, rank=rank):
                    lens = torch.tensor(GLOBAL_LENS, dtype=torch.int32)
                    got = get_dcp_lens(lens, n, rank).to(torch.int64).tolist()
                    expected = [
                        sum(1 for p in range(L) if p % n == rank) for L in GLOBAL_LENS
                    ]
                    self.assertEqual(got, expected)

    def test_newest_token_is_owned_by_exactly_one_rank(self):
        """The token just written by this decode step must be visible to one and
        only one rank, otherwise the merge double counts or drops it."""
        for n in DCP_SIZES:
            for global_len in range(1, 65):
                with self.subTest(dcp_size=n, global_len=global_len):
                    prev = torch.tensor([global_len - 1], dtype=torch.int32)
                    cur = torch.tensor([global_len], dtype=torch.int32)
                    grew = [
                        int(get_dcp_lens(cur, n, rank).item())
                        - int(get_dcp_lens(prev, n, rank).item())
                        for rank in range(n)
                    ]
                    self.assertEqual(sum(grew), 1)
                    self.assertEqual(grew[(global_len - 1) % n], 1)


if __name__ == "__main__":
    unittest.main()
