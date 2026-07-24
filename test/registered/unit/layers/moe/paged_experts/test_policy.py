"""Unit tests for srt/layers/moe/paged_experts/policy.py (pure-Python eviction policies; no GPU)."""

import unittest

from sglang.srt.layers.moe.paged_experts.policy import (
    LFUPolicy,
    LRUPolicy,
    ResidencyPolicy,
    make_residency_policy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResidencyPolicy(CustomTestCase):
    def test_factory_and_abstract(self):
        self.assertIsInstance(make_residency_policy("lru", 4, 8), LRUPolicy)
        self.assertIsInstance(make_residency_policy("lfu", 4, 8), LFUPolicy)
        with self.assertRaises(ValueError):
            make_residency_policy("nope", 4, 8)
        with self.assertRaises(TypeError):
            ResidencyPolicy(4, 8)  # abstract: pick_victim unimplemented

    def test_lru_picks_least_recent_and_skips_needed(self):
        p = LRUPolicy(4, 8)
        slot_expert = [0, 1, 2, 3]
        p.begin_step()  # step 1
        p.record_use(1, 1)  # expert 1 (slot 1) used most recently
        # needed={1} -> slot 1 excluded; slots 0/2/3 all lastuse 0 -> lowest index 0
        self.assertEqual(p.pick_victim(slot_expert, {1}), 0)
        # make slot 0 recent; now slot 2 is the LRU non-needed
        p.begin_step()  # step 2
        p.record_use(0, 0)
        self.assertEqual(p.pick_victim(slot_expert, set()), 2)

    def test_lru_returns_minus_one_when_all_needed(self):
        p = LRUPolicy(2, 8)
        self.assertEqual(p.pick_victim([5, 6], {5, 6}), -1)

    def test_lfu_evicts_least_frequent_keeps_hot(self):
        p = LFUPolicy(4, 8)
        slot_expert = [5, 2, 0, 3]
        # expert 5 hot (3 uses), expert 2 warm (1), experts 0/3 cold (0)
        for _ in range(3):
            p.begin_step()
            p.record_use(5, 0)
        p.begin_step()
        p.record_use(2, 1)
        # cold experts 0/3 tie on freq=0 -> LRU tiebreak (both lastuse 0) -> lowest index slot 2
        self.assertEqual(p.pick_victim(slot_expert, set()), 2)
        # the hot expert 5 is never the victim
        self.assertNotEqual(p.pick_victim(slot_expert, set()), 0)
        # excluding the cold slot 2 -> next lowest freq is the other cold slot 3, not warm/hot
        self.assertEqual(p.pick_victim(slot_expert, {0}), 3)

    def test_lfu_frequency_follows_the_expert_across_slots(self):
        # freq is per-expert (indexed by expert, not slot), so it follows the expert across eviction and
        # re-paging — a briefly-absent hot expert is favored back in even if it lands in a different slot.
        p = LFUPolicy(2, 8)
        for _ in range(5):
            p.begin_step()
            p.record_use(7, 0)  # expert 7 very hot (freq 5) while in slot 0
        # expert 7 has since re-paged into slot 1; slot 0 now holds cold expert 2 (freq 0).
        # 7's retained freq protects it -> evict the cold slot 0, not slot 1.
        self.assertEqual(p.pick_victim([2, 7], set()), 0)


if __name__ == "__main__":
    unittest.main()
