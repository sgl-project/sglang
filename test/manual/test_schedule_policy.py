import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


class TestSchedulePolicy(CustomTestCase):

    def setUp(self):
        self.tree_cache = RadixCache.create_simulated()

    def test_init_with_cache_aware_policy(self):
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAwarePolicy.LPM)

    def test_init_with_cache_agnostic_policy(self):
        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_init_with_unknown_policy(self):
        with self.assertRaises(ValueError):
            SchedulePolicy(
                policy="invalid",
                tree_cache=self.tree_cache,
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
            )

    def test_init_with_disabled_cache(self):
        tree_cache = RadixCache.create_simulated(disable=True)
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_calc_priority_fcfs(self):
        tree_cache = RadixCache.create_simulated()
        waiting_queue = [
            _make_req(1, "a b", [1, 2]),
            _make_req(3, "a b c", [1, 2, 3]),
            _make_req(2, "a", [1]),
        ]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if FCFS keeps the original order
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 3)
        self.assertEqual(waiting_queue[2].rid, 2)

    def test_calc_priority_priority_enabled_fcfs_scheduling(self):
        tree_cache = RadixCache.create_simulated()
        r1 = _make_req(1, "a b", [1, 2])
        r2 = _make_req(3, "a b c", [1, 2, 3])
        r3 = _make_req(2, "a", [1])
        r1.priority, r1.time_stats.wait_queue_entry_time = 1, 1
        r2.priority, r2.time_stats.wait_queue_entry_time = 0, 1
        r3.priority, r3.time_stats.wait_queue_entry_time = 0, 0

        waiting_queue = [r1, r2, r3]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)

        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_fcfs_scheduling_with_low_priority_values_first(
        self,
    ):
        tree_cache = RadixCache.create_simulated()
        r1 = _make_req(1, "a b", [1, 2])
        r2 = _make_req(3, "a b c", [1, 2, 3])
        r3 = _make_req(2, "a", [1])
        r1.priority, r1.time_stats.wait_queue_entry_time = -1, 1
        r2.priority, r2.time_stats.wait_queue_entry_time = 0, 1
        r3.priority, r3.time_stats.wait_queue_entry_time = 0, 0

        waiting_queue = [r1, r2, r3]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_longest_output_first_scheduling(self):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1000)),
            _make_req(3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10)),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100)),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_longest_output_first_scheduling(self):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=1),
            _make_req(
                3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=0
            ),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=0),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_longest_output_first_scheduling_with_low_priority_values_first(
        self,
    ):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=0),
            _make_req(
                3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=1
            ),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=1),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_routing_key_scheduling(self):
        """Test routing-key policy: prioritize by routing key frequency in running batch."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_a"),
            _make_req("r2", "b", [2], routing_key="key_a"),
            _make_req("r3", "c", [3], routing_key="key_b"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_b"),
            _make_req("w2", "e", [5], routing_key="key_a"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w2")
        self.assertEqual(waiting_queue[1].rid, "w1")
        self.assertEqual(waiting_queue[2].rid, "w3")

    def test_calc_priority_routing_key_tie_break_by_lexicographic_order(self):
        """Test routing-key policy: tie-break by lexicographic order."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_b"),
            _make_req("r2", "b", [2], routing_key="key_a"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_b"),
            _make_req("w2", "e", [5], routing_key="key_a"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w2")
        self.assertEqual(waiting_queue[1].rid, "w1")

    def test_calc_priority_routing_key_no_match_deprioritized(self):
        """Test routing-key policy: requests without matching routing keys are deprioritized."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_a"),
            _make_req("r2", "b", [2], routing_key="key_b"),
            _make_req("r3", "c", [3], routing_key="key_c"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_d"),
            _make_req("w2", "e", [5], routing_key="key_e"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w3")
        self.assertEqual(waiting_queue[1].rid, "w1")
        self.assertEqual(waiting_queue[2].rid, "w2")

    def test_calc_priority_routing_key_empty_running_batch(self):
        """Test routing-key policy: empty running batch keeps original order."""
        tree_cache = RadixCache.create_simulated()

        running_batch = ScheduleBatch(reqs=[])

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_a"),
            _make_req("w2", "e", [5], routing_key="key_b"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w1")
        self.assertEqual(waiting_queue[1].rid, "w2")
        self.assertEqual(waiting_queue[2].rid, "w3")


class TestEpochMemoizedMatch(CustomTestCase):
    """Correctness of epoch-memoized prefix matching in match_prefix_for_req."""

    def _tree_with_prefix(self):
        import torch

        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        tree = RadixCache.create_simulated()
        toks = array("q", list(range(1, 33)))
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=toks),
                value=torch.arange(1, len(toks) + 1, dtype=torch.int64),
            )
        )
        return tree

    def test_memo_hit_matches_fresh_result(self):
        from sglang.srt.managers.schedule_policy import match_prefix_for_req

        tree = self._tree_with_prefix()
        req = _make_req("m1", "x", list(range(1, 20)))

        r1 = match_prefix_for_req(tree, req)
        len1 = len(req.prefix_indices)
        node1 = req.last_node
        num1 = req.num_matched_prefix_tokens
        self.assertGreater(len1, 0)
        self.assertEqual(req._match_epoch, tree.match_epoch())

        # Second call on an unchanged tree must be a memo hit with identical state.
        r2 = match_prefix_for_req(tree, req)
        self.assertIs(r2, r1)
        self.assertEqual(len(req.prefix_indices), len1)
        self.assertIs(req.last_node, node1)
        self.assertEqual(req.num_matched_prefix_tokens, num1)

    def test_memo_invalidated_after_mutation(self):
        import torch

        from sglang.srt.managers.schedule_policy import match_prefix_for_req
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        tree = self._tree_with_prefix()
        req = _make_req("m2", "x", list(range(1, 20)))
        match_prefix_for_req(tree, req)
        epoch_before = req._match_epoch

        # Insert a longer overlapping key: epoch bumps, memo must be invalidated.
        toks = array("q", list(range(1, 40)))
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=toks),
                value=torch.arange(1, len(toks) + 1, dtype=torch.int64),
            )
        )
        self.assertNotEqual(tree.match_epoch(), epoch_before)
        match_prefix_for_req(tree, req)
        self.assertEqual(req._match_epoch, tree.match_epoch())

    def test_calc_priority_identical_on_unchanged_tree(self):
        """LPM ordering with memoization must equal ordering across rounds."""
        import torch

        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        tree = RadixCache.create_simulated()
        for base in range(0, 5):
            toks = array("q", [base] + list(range(100, 100 + 20 * (base + 1))))
            tree.insert(
                InsertParams(
                    key=RadixKey(token_ids=toks),
                    value=torch.arange(1, len(toks) + 1, dtype=torch.int64),
                )
            )
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=tree,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        q = [
            _make_req(f"w{b}", "x", [b] + list(range(100, 100 + 20 * (b + 1))))
            for b in range(5)
        ]
        policy.calc_priority(q)
        order_round1 = [r.rid for r in q]
        matched_round1 = {r.rid: r.num_matched_prefix_tokens for r in q}
        # Second round, unchanged tree -> memo hits -> identical order & lengths.
        policy.calc_priority(q)
        order_round2 = [r.rid for r in q]
        matched_round2 = {r.rid: r.num_matched_prefix_tokens for r in q}
        self.assertEqual(order_round1, order_round2)
        self.assertEqual(matched_round1, matched_round2)


if __name__ == "__main__":
    unittest.main()
