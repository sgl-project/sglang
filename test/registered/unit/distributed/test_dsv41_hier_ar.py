"""8×V100 hierarchical 2-step AR/A2A rank partition and A2A packing."""

from __future__ import annotations

import torch

from sglang.srt.distributed.device_communicators.dsv41_hier_ar import (
    partition_quads_and_pairs,
    simulate_two_step_a2a,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsv41HierArPartition(CustomTestCase):
    def test_pci_order_quads_and_bridges(self):
        quads, pairs = partition_quads_and_pairs(range(8))
        self.assertEqual(quads, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(pairs, [[0, 4], [1, 5], [2, 6], [3, 7]])

    def test_rejects_wrong_world(self):
        with self.assertRaises(ValueError):
            partition_quads_and_pairs(range(4))

    def test_two_step_a2a_matches_direct(self):
        # Each src writes a unique (src, dest) marker into dest-major row dest.
        n = 3
        sends = []
        for src in range(8):
            t = torch.empty(8, n, dtype=torch.int32)
            for dest in range(8):
                t[dest] = src * 100 + dest
            sends.append(t)
        got = simulate_two_step_a2a(sends)
        for dest in range(8):
            for src in range(8):
                self.assertTrue(
                    torch.equal(got[dest][src], sends[src][dest]),
                    f"dest={dest} src={src}",
                )
