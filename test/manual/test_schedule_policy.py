import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    PrefillAdder,
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


def _swa_adder(size_swa, sliding_window, page_size=16, rem_chunk_tokens=512):
    """A PrefillAdder carrying only the fields _swa_req_never_fits transitively
    reads, built via __new__ so the check is tested as pure logic (no KV pools,
    no GPU)."""
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.page_size = page_size
    adder.rem_chunk_tokens = rem_chunk_tokens
    adder.tree_cache = SimpleNamespace(sliding_window_size=sliding_window)
    adder.token_to_kv_pool_allocator = SimpleNamespace(size_swa=size_swa)
    return adder


class TestSwaChunkCapHatch(CustomTestCase):
    """The _swa_chunk_cap shrink-and-admit hatch must fire ONLY for a request
    whose SWA budget can never fit the drained pool (true head-of-line
    livelock). _swa_req_never_fits() is the gate: True => hatch, False => wait
    (NO_TOKEN). Admitting transient-pressure requests instead of waiting
    collapses the SWA evictable cushion and causes a retraction storm."""

    def test_transient_pressure_request_waits(self):
        # Small request against an ample pool: budget << pool, so it would fit
        # once running decodes drain -> must wait, not take the hatch.
        adder = _swa_adder(size_swa=1024, sliding_window=128)
        self.assertFalse(
            adder._swa_req_never_fits(extend_input_len=256, max_new_tokens=64)
        )

    def test_request_larger_than_whole_pool_takes_hatch(self):
        # A large host-hit load-back charge pushes the budget past the entire
        # pool: it can never fit however far the pool drains -> hatch (True).
        adder = _swa_adder(size_swa=1024, sliding_window=128)
        self.assertTrue(
            adder._swa_req_never_fits(
                extend_input_len=256, max_new_tokens=64, swa_host_hit_length=4096
            )
        )

    def test_decision_is_gated_by_pool_capacity(self):
        # Same request; only the pool size changes. Proves the check compares
        # the budget against size_swa (guards against a wrong-accessor bug):
        # never-fits on a small pool, fits on a large one.
        req = dict(extend_input_len=256, max_new_tokens=64, swa_host_hit_length=600)
        self.assertTrue(
            _swa_adder(size_swa=512, sliding_window=128)._swa_req_never_fits(**req)
        )
        self.assertFalse(
            _swa_adder(size_swa=4096, sliding_window=128)._swa_req_never_fits(**req)
        )


if __name__ == "__main__":
    unittest.main()
