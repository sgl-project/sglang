"""Unit tests for elastic expert-map repair — no server, no model loading.

Every survivor recomputes this repair independently instead of exchanging the map, so
the rule has to be deterministic as well as correct.
"""

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
    def test_no_orphans_leaves_the_map_untouched(self):
        p2l = torch.tensor([[0, 1, 2, 3]])
        before = p2l.clone()
        self.assertEqual(repair_orphan_logicals(shrunk_p2l=p2l, num_logical=4), {})
        self.assertTrue(torch.equal(p2l, before))

    def test_orphan_takes_over_a_duplicate_slot(self):
        # Logical 3 is gone; logical 0 has two replicas and donates one.
        p2l = torch.tensor([[0, 0, 1, 2]])
        repaired = repair_orphan_logicals(shrunk_p2l=p2l, num_logical=4)
        self.assertEqual(repaired, {0: [3]})
        self.assertEqual(sorted(p2l[0].tolist()), [0, 1, 2, 3])

    def test_every_logical_has_a_replica_after_repair(self):
        p2l = torch.tensor([[0, 0, 0, 0], [1, 1, 2, 2]])
        repair_orphan_logicals(shrunk_p2l=p2l, num_logical=4)
        for layer in range(p2l.shape[0]):
            present = set(p2l[layer].tolist())
            self.assertEqual(
                present, {0, 1, 2, 3}, f"layer {layer} still missing a logical"
            )

    def test_repair_is_deterministic_across_ranks(self):
        """Survivors never exchange the repaired map, so they must agree bit for bit."""
        start = torch.tensor([[0, 0, 0, 1], [2, 2, 3, 3]])
        first = start.clone()
        second = start.clone()
        self.assertEqual(
            repair_orphan_logicals(shrunk_p2l=first, num_logical=4),
            repair_orphan_logicals(shrunk_p2l=second, num_logical=4),
        )
        self.assertTrue(torch.equal(first, second))

    def test_per_layer_orphans_are_keyed_by_layer(self):
        p2l = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        repaired = repair_orphan_logicals(shrunk_p2l=p2l, num_logical=4)
        self.assertNotIn(0, repaired)
        self.assertEqual(repaired[1], [2, 3])

    def test_too_few_slots_raises_rather_than_dropping_an_expert(self):
        p2l = torch.tensor([[0, 1]])
        with self.assertRaises(RuntimeError) as ctx:
            repair_orphan_logicals(shrunk_p2l=p2l, num_logical=4)
        self.assertIn("ep-num-redundant-experts", str(ctx.exception))

    def test_no_donor_available_raises(self):
        # Four distinct logicals in four slots, but num_logical is 5: nothing to donate.
        p2l = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        repair_orphan_logicals(shrunk_p2l=p2l, num_logical=5)  # exactly covered
        tight = torch.tensor([[0, 1, 2, 3, -1]])
        with self.assertRaises(RuntimeError):
            repair_orphan_logicals(shrunk_p2l=tight, num_logical=5)

    def test_negative_slots_are_not_counted_as_replicas(self):
        p2l = torch.tensor([[-1, 0, 0, 1]])
        repaired = repair_orphan_logicals(shrunk_p2l=p2l, num_logical=3)
        self.assertEqual(repaired, {0: [2]})
        self.assertIn(2, p2l[0].tolist())


class TestLocalRelabelledLogicals(CustomTestCase):
    def test_unchanged_window_reports_nothing(self):
        p2l = torch.tensor([[0, 1, 2, 3]])
        self.assertEqual(
            local_relabelled_logicals(p2l, p2l.clone(), num_local=2, ep_rank=0), {}
        )

    def test_only_this_ranks_own_slots_are_reported(self):
        old = torch.tensor([[0, 1, 2, 3]])
        new = torch.tensor([[0, 1, 9, 9]])
        # Rank 0 owns slots 0-1, which did not change.
        self.assertEqual(
            local_relabelled_logicals(old, new, num_local=2, ep_rank=0), {}
        )
        # Rank 1 owns slots 2-3, which did.
        self.assertEqual(
            local_relabelled_logicals(old, new, num_local=2, ep_rank=1), {0: [9]}
        )

    def test_slots_past_the_old_width_count_as_changed(self):
        """On a grow these are the freshly appended slots, which hold nothing yet."""
        old = torch.tensor([[0, 1]])
        new = torch.tensor([[0, 1, 4, 5]])
        self.assertEqual(
            local_relabelled_logicals(old, new, num_local=2, ep_rank=1), {0: [4, 5]}
        )

    def test_window_beyond_the_map_is_empty(self):
        old = torch.tensor([[0, 1]])
        new = torch.tensor([[0, 1]])
        self.assertEqual(
            local_relabelled_logicals(old, new, num_local=2, ep_rank=5), {}
        )

    def test_negative_slots_are_excluded(self):
        old = torch.tensor([[0, 1]])
        new = torch.tensor([[-1, -1]])
        self.assertEqual(
            local_relabelled_logicals(old, new, num_local=2, ep_rank=0), {}
        )

    def test_results_are_sorted_and_deduplicated(self):
        old = torch.tensor([[0, 0, 0, 0]])
        new = torch.tensor([[7, 3, 3, 7]])
        self.assertEqual(
            local_relabelled_logicals(old, new, num_local=4, ep_rank=0), {0: [3, 7]}
        )


if __name__ == "__main__":
    unittest.main()
