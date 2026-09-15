"""CPU unit test for the NPU DCP extend-gather plan.

Pins ``plan_dcp_extend_gather`` (``layers/dcp/layout.py``). Under DCP each rank
holds a 1/dcp_size share of every request's prefix KV. At extend the NPU path
all-gathers those shares into one scratch buffer, appends this chunk's own KV
after them, and writes ``dcp_kv_buffer`` with a single ``index_select``. The
plan is that index.

A wrong entry puts real KV at the wrong position, and attention reads it without
complaint -- output stays fluent and stops following the prompt. So the test
builds every rank's share from a known KV, lays the shares out the way
``all_gather_into_tensor`` does (rank 0's whole send, then rank 1's, ...), fills
padding with NaN, and requires the indexed result to equal the position-ordered
KV exactly. A pad row or a neighbour's row cannot pass.

Usage:
    python -m pytest test_dcp_extend_gather_plan.py -v
    python test_dcp_extend_gather_plan.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import plan_dcp_extend_gather
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 16]
DIM = 3

# (prefix_lens, extend_lens). The first rows are what a prefix-cache hit and a
# chunked prefill produce -- page-aligned prefixes -- and the rest are the
# general case the plan still has to get right.
CASES = [
    ([0], [7]),
    ([16], [5]),
    ([2048 * 3], [1558]),
    ([16384, 32768], [16384, 1]),
    ([5], [3]),
    ([1], [1]),
    ([33, 0, 48, 7], [1, 9, 16, 2]),
    ([0, 0], [4, 4]),
]


def _gather(prefix_lens, extend_lens, dcp_size, seed=0):
    """Run the plan against a simulated all-gather. Returns (out, expected, plans)."""
    g = torch.Generator().manual_seed(seed)
    prefixes = [torch.randn(p, 1, DIM, generator=g) for p in prefix_lens]
    extends = [torch.randn(e, 1, DIM, generator=g) for e in extend_lens]
    plans = [
        plan_dcp_extend_gather(prefix_lens, extend_lens, dcp_size, rank)
        for rank in range(dcp_size)
    ]

    sends = []
    for rank, plan in enumerate(plans):
        # What the planner lists for this rank: positions rank, rank + dcp_size,
        # ... of each request, concatenated per request.
        local = [kv[rank::dcp_size] for kv in prefixes]
        send = torch.full((plan.send_rows, 1, DIM), float("nan"))
        dst = 0
        for shard, local_len, padded_len in zip(
            local, plan.local_lens, plan.padded_lens
        ):
            assert shard.shape[0] == local_len, (shard.shape[0], local_len)
            send[dst : dst + local_len] = shard
            dst += padded_len
        sends.append(send)

    scratch = torch.cat(sends + extends) if sends or extends else torch.empty(0)
    out = scratch.index_select(0, plans[0].index)
    expected = torch.cat([torch.cat([p, e]) for p, e in zip(prefixes, extends)])
    return out, expected, plans


class TestDcpExtendGatherPlan(CustomTestCase):
    def test_the_index_rebuilds_every_request_in_position_order(self):
        for prefix_lens, extend_lens in CASES:
            for dcp_size in DCP_SIZES:
                with self.subTest(prefix=prefix_lens, extend=extend_lens, dcp=dcp_size):
                    out, expected, _ = _gather(prefix_lens, extend_lens, dcp_size)
                    self.assertTrue(torch.equal(out, expected))

    def test_every_rank_plans_the_same_collective(self):
        # all_gather_into_tensor needs equal sends on every rank, and every rank
        # must read the gathered rows the same way.
        for prefix_lens, extend_lens in CASES:
            for dcp_size in DCP_SIZES:
                _, _, plans = _gather(prefix_lens, extend_lens, dcp_size)
                for plan in plans[1:]:
                    self.assertEqual(plan.send_rows, plans[0].send_rows)
                    self.assertEqual(plan.gather_rows, plans[0].gather_rows)
                    self.assertTrue(torch.equal(plan.index, plans[0].index))

    def test_the_ranks_partition_each_prefix(self):
        for prefix_lens, extend_lens in CASES:
            for dcp_size in DCP_SIZES:
                _, _, plans = _gather(prefix_lens, extend_lens, dcp_size)
                for i, prefix_len in enumerate(prefix_lens):
                    self.assertEqual(
                        sum(plan.local_lens[i] for plan in plans), prefix_len
                    )

    def test_aligned_prefixes_need_no_padding(self):
        # The case the NPU path takes a single-copy branch for.
        for dcp_size in DCP_SIZES:
            prefix_lens = [dcp_size * 128, 0, dcp_size * 3]
            for rank in range(dcp_size):
                plan = plan_dcp_extend_gather(prefix_lens, [7, 3, 1], dcp_size, rank)
                self.assertEqual(plan.local_lens, plan.padded_lens)

    def test_a_batch_without_prefix_gathers_nothing(self):
        plan = plan_dcp_extend_gather([0, 0], [4, 6], 16, 3)
        self.assertEqual(plan.gather_rows, 0)
        self.assertTrue(torch.equal(plan.index, torch.arange(10, dtype=torch.int64)))

    def test_the_index_stays_inside_the_scratch_buffer(self):
        for prefix_lens, extend_lens in CASES:
            for dcp_size in DCP_SIZES:
                plan = plan_dcp_extend_gather(prefix_lens, extend_lens, dcp_size, 0)
                self.assertEqual(
                    plan.index.numel(), sum(prefix_lens) + sum(extend_lens)
                )
                if plan.index.numel():
                    self.assertLess(
                        int(plan.index.max()), plan.gather_rows + plan.extend_rows
                    )


if __name__ == "__main__":
    unittest.main()
