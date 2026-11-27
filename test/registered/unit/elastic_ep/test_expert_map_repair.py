"""Unit tests for elastic expert-map repair, which every survivor computes alone."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.elastic_ep.expert_map_repair import (
    local_relabelled_logicals,
    repair_orphan_logicals,
)
from sglang.test.test_utils import CustomTestCase


class TestRepairOrphanLogicals(CustomTestCase):
    CASES = [
        # (case, p2l, num_logical, repaired)
        ("no orphans", [[0, 1, 2, 3]], 4, {}),
        ("exactly covered", [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], 5, {}),
        ("orphan takes over a duplicate slot", [[0, 0, 1, 2]], 4, {0: [3]}),
        ("every layer", [[0, 0, 0, 0], [1, 1, 2, 2]], 4, {0: [1, 2, 3], 1: [0, 3]}),
        ("keyed by layer", [[0, 1, 2, 3], [0, 0, 1, 1]], 4, {1: [2, 3]}),
        ("negative slots are not replicas", [[-1, 0, 0, 1]], 3, {0: [2]}),
    ]

    def test_every_logical_has_a_replica_after_repair(self):
        for case, rows, n, expected in self.CASES:
            with self.subTest(case):
                p2l = torch.tensor(rows)
                before = p2l.clone()
                self.assertEqual(
                    repair_orphan_logicals(shrunk_p2l=p2l, num_logical=n), expected
                )
                if not expected:
                    self.assertTrue(torch.equal(p2l, before))
                for layer in p2l.tolist():
                    self.assertLessEqual(set(range(n)), set(layer))

    def test_repair_is_deterministic_across_ranks(self):
        """Survivors never exchange the repaired map, so they must agree bit for bit."""
        first = torch.tensor([[0, 0, 0, 1], [2, 2, 3, 3]])
        second = first.clone()
        self.assertEqual(
            repair_orphan_logicals(shrunk_p2l=first, num_logical=4),
            repair_orphan_logicals(shrunk_p2l=second, num_logical=4),
        )
        self.assertTrue(torch.equal(first, second))

    def test_too_few_slots_or_no_donor_raises_rather_than_dropping_an_expert(self):
        for rows, n in (([[0, 1]], 4), ([[0, 1, 2, 3, -1]], 5)):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(RuntimeError, "ep-num-redundant-experts"):
                    repair_orphan_logicals(shrunk_p2l=torch.tensor(rows), num_logical=n)


class TestLocalRelabelledLogicals(CustomTestCase):
    CASES = [
        # (case, old, new, num_local, ep_rank, expected)
        ("unchanged window", [0, 1, 2, 3], [0, 1, 2, 3], 2, 0, {}),
        ("another rank's slots changed", [0, 1, 2, 3], [0, 1, 9, 9], 2, 0, {}),
        ("this rank's slots changed", [0, 1, 2, 3], [0, 1, 9, 9], 2, 1, {0: [9]}),
        # On a grow these are the freshly appended slots, which hold nothing yet.
        ("slots past the old width", [0, 1], [0, 1, 4, 5], 2, 1, {0: [4, 5]}),
        ("window beyond the map", [0, 1], [0, 1], 2, 5, {}),
        ("negative slots excluded", [0, 1], [-1, -1], 2, 0, {}),
        ("sorted and deduplicated", [0, 0, 0, 0], [7, 3, 3, 7], 4, 0, {0: [3, 7]}),
    ]

    def test_relabelled_logicals(self):
        for case, old, new, num_local, rank, expected in self.CASES:
            with self.subTest(case):
                maps = torch.tensor([old]), torch.tensor([new])
                got = local_relabelled_logicals(
                    *maps, num_local=num_local, ep_rank=rank
                )
                self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
