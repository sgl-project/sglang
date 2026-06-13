import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_req(
    rid="r0",
    priority=0,
    max_new_tokens=100,
    output_len=0,
    wait_time=0.0,
    routing_key=None,
    origin_input_ids=None,
    prefix_indices=None,
    last_node=None,
):
    return SimpleNamespace(
        rid=rid,
        priority=priority,
        output_ids=[0] * output_len,
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        time_stats=SimpleNamespace(wait_queue_entry_time=wait_time),
        routing_key=routing_key,
        origin_input_ids=origin_input_ids if origin_input_ids is not None else [],
        extra_key=None,
        prefix_indices=(
            prefix_indices
            if prefix_indices is not None
            else torch.empty(0, dtype=torch.int64)
        ),
        last_node=last_node,
        last_host_node=None,
        host_hit_length=0,
    )


def _make_tree_cache(*, disable=False):
    cache = MagicMock()
    cache.disable = disable
    cache.supports_mamba.return_value = False
    cache.match_prefix.return_value = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        host_hit_length=0,
    )
    return cache


def _make_schedule_policy(
    policy_str,
    *,
    cache_disable=False,
    enable_priority=False,
    low_first=False,
):
    return SchedulePolicy(
        policy=policy_str,
        tree_cache=_make_tree_cache(disable=cache_disable),
        enable_hierarchical_cache=False,
        enable_priority_scheduling=enable_priority,
        schedule_low_priority_values_first=low_first,
    )


class TestValidateAndAdjustPolicy(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_lpm_with_enabled_cache(self):
        sp = _make_schedule_policy("lpm")
        self.assertEqual(sp.policy, CacheAwarePolicy.LPM)

    def test_dfs_weight_with_enabled_cache(self):
        sp = _make_schedule_policy("dfs-weight")
        self.assertEqual(sp.policy, CacheAwarePolicy.DFS_WEIGHT)

    def test_fcfs(self):
        sp = _make_schedule_policy("fcfs")
        self.assertEqual(sp.policy, CacheAgnosticPolicy.FCFS)

    def test_lof(self):
        sp = _make_schedule_policy("lof")
        self.assertEqual(sp.policy, CacheAgnosticPolicy.LOF)

    def test_random(self):
        sp = _make_schedule_policy("random")
        self.assertEqual(sp.policy, CacheAgnosticPolicy.RANDOM)

    def test_routing_key(self):
        sp = _make_schedule_policy("routing-key")
        self.assertEqual(sp.policy, CacheAgnosticPolicy.ROUTING_KEY)

    def test_lpm_with_disabled_cache_falls_back_to_fcfs(self):
        sp = _make_schedule_policy("lpm", cache_disable=True)
        self.assertEqual(sp.policy, CacheAgnosticPolicy.FCFS)

    def test_dfs_weight_with_disabled_cache_falls_back_to_fcfs(self):
        sp = _make_schedule_policy("dfs-weight", cache_disable=True)
        self.assertEqual(sp.policy, CacheAgnosticPolicy.FCFS)

    def test_unknown_policy_raises_value_error(self):
        with self.assertRaises(ValueError):
            _make_schedule_policy("not-a-real-policy")


class TestDetermineActivePolicy(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.sp = _make_schedule_policy("lpm")

    def test_lpm_small_queue_stays_lpm(self):
        queue = [_make_req(rid=str(i)) for i in range(10)]
        self.assertEqual(self.sp._determine_active_policy(queue), CacheAwarePolicy.LPM)

    def test_lpm_at_boundary_stays_lpm(self):
        queue = [_make_req(rid=str(i)) for i in range(128)]
        self.assertEqual(self.sp._determine_active_policy(queue), CacheAwarePolicy.LPM)

    def test_lpm_over_boundary_falls_back_to_fcfs(self):
        queue = [_make_req(rid=str(i)) for i in range(129)]
        self.assertEqual(
            self.sp._determine_active_policy(queue), CacheAgnosticPolicy.FCFS
        )

    def test_non_lpm_policy_not_affected_by_queue_size(self):
        sp = _make_schedule_policy("lof")
        queue = [_make_req(rid=str(i)) for i in range(200)]
        self.assertEqual(sp._determine_active_policy(queue), CacheAgnosticPolicy.LOF)


class TestSortByLongestPrefix(CustomTestCase):
    def test_longer_prefix_first(self):
        reqs = [
            _make_req(rid="r0", prefix_indices=torch.arange(5)),
            _make_req(rid="r1", prefix_indices=torch.arange(10)),
            _make_req(rid="r2", prefix_indices=torch.arange(2)),
        ]
        SchedulePolicy._sort_by_longest_prefix(reqs, set())
        self.assertEqual([len(r.prefix_indices) for r in reqs], [10, 5, 2])

    def test_deprioritized_go_to_end(self):
        reqs = [
            _make_req(rid="r0", prefix_indices=torch.arange(10)),
            _make_req(rid="r1", prefix_indices=torch.arange(5)),
        ]
        SchedulePolicy._sort_by_longest_prefix(reqs, {"r0"})
        self.assertEqual(reqs[0].rid, "r1")
        self.assertEqual(reqs[1].rid, "r0")

    def test_deprioritized_goes_to_end_even_with_longer_prefix(self):
        reqs = [
            _make_req(rid="short-a", prefix_indices=torch.arange(2)),
            _make_req(rid="long-deprio", prefix_indices=torch.arange(8)),
            _make_req(rid="medium-b", prefix_indices=torch.arange(4)),
        ]
        SchedulePolicy._sort_by_longest_prefix(reqs, {"long-deprio"})
        self.assertEqual(reqs[0].rid, "medium-b")
        self.assertEqual(reqs[1].rid, "short-a")
        self.assertEqual(reqs[2].rid, "long-deprio")

    def test_all_deprioritized_stable_order_preserved(self):
        reqs = [
            _make_req(rid="r0", prefix_indices=torch.arange(3)),
            _make_req(rid="r1", prefix_indices=torch.arange(1)),
        ]
        SchedulePolicy._sort_by_longest_prefix(reqs, {"r0", "r1"})
        self.assertEqual({reqs[0].rid, reqs[1].rid}, {"r0", "r1"})


class TestSortByLongestOutput(CustomTestCase):
    def test_without_priority_sorts_by_max_new_tokens_desc(self):
        reqs = [
            _make_req(rid="r0", max_new_tokens=50),
            _make_req(rid="r1", max_new_tokens=200),
            _make_req(rid="r2", max_new_tokens=100),
        ]
        SchedulePolicy._sort_by_longest_output(reqs, False, priority_sign=-1)
        self.assertEqual(
            [r.sampling_params.max_new_tokens for r in reqs], [200, 100, 50]
        )

    def test_with_priority_high_values_first(self):
        reqs = [
            _make_req(rid="r0", priority=1, max_new_tokens=100),
            _make_req(rid="r1", priority=3, max_new_tokens=50),
            _make_req(rid="r2", priority=2, max_new_tokens=200),
        ]
        SchedulePolicy._sort_by_longest_output(reqs, True, priority_sign=-1)
        self.assertEqual([r.priority for r in reqs], [3, 2, 1])

    def test_with_priority_low_values_first(self):
        reqs = [
            _make_req(rid="r0", priority=3, max_new_tokens=100),
            _make_req(rid="r1", priority=1, max_new_tokens=50),
            _make_req(rid="r2", priority=2, max_new_tokens=200),
        ]
        SchedulePolicy._sort_by_longest_output(reqs, True, priority_sign=1)
        self.assertEqual([r.priority for r in reqs], [1, 2, 3])

    def test_same_priority_tiebreak_by_max_new_tokens(self):
        reqs = [
            _make_req(rid="r0", priority=1, max_new_tokens=50),
            _make_req(rid="r1", priority=1, max_new_tokens=200),
        ]
        SchedulePolicy._sort_by_longest_output(reqs, True, priority_sign=-1)
        self.assertEqual(reqs[0].rid, "r1")
        self.assertEqual(reqs[1].rid, "r0")


class TestSortByPriorityAndFcfs(CustomTestCase):
    def test_low_values_first(self):
        reqs = [
            _make_req(rid="r0", priority=2, wait_time=1.0),
            _make_req(rid="r1", priority=0, wait_time=2.0),
            _make_req(rid="r2", priority=1, wait_time=3.0),
        ]
        SchedulePolicy._sort_by_priority_and_fcfs(reqs, priority_sign=1)
        self.assertEqual([r.priority for r in reqs], [0, 1, 2])

    def test_high_values_first(self):
        reqs = [
            _make_req(rid="r0", priority=0, wait_time=1.0),
            _make_req(rid="r1", priority=2, wait_time=2.0),
            _make_req(rid="r2", priority=1, wait_time=3.0),
        ]
        SchedulePolicy._sort_by_priority_and_fcfs(reqs, priority_sign=-1)
        self.assertEqual([r.priority for r in reqs], [2, 1, 0])

    def test_same_priority_tiebreak_by_wait_time(self):
        reqs = [
            _make_req(rid="r0", priority=1, wait_time=3.0),
            _make_req(rid="r1", priority=1, wait_time=1.0),
            _make_req(rid="r2", priority=1, wait_time=2.0),
        ]
        SchedulePolicy._sort_by_priority_and_fcfs(reqs, priority_sign=-1)
        self.assertEqual(
            [r.time_stats.wait_queue_entry_time for r in reqs], [1.0, 2.0, 3.0]
        )

    def test_priority_beats_wait_time(self):
        reqs = [
            _make_req(rid="early-low", priority=0, wait_time=0.0),
            _make_req(rid="late-high", priority=5, wait_time=10.0),
        ]
        SchedulePolicy._sort_by_priority_and_fcfs(reqs, priority_sign=-1)
        self.assertEqual(reqs[0].rid, "late-high")
        self.assertEqual(reqs[1].rid, "early-low")


class TestSortByRoutingKey(CustomTestCase):
    def _make_running_batch(self, routing_keys):
        batch = MagicMock()
        batch.reqs = [SimpleNamespace(routing_key=k) for k in routing_keys]
        return batch

    def test_empty_routing_key_counts_is_noop(self):
        reqs = [
            _make_req(rid="r0", routing_key="key-a"),
            _make_req(rid="r1", routing_key="key-b"),
        ]
        original_order = [r.rid for r in reqs]
        SchedulePolicy._sort_by_routing_key(
            reqs, self._make_running_batch([None, None])
        )
        self.assertEqual([r.rid for r in reqs], original_order)

    def test_matching_keys_sorted_by_frequency_descending(self):
        running_batch = self._make_running_batch(
            ["key-a", "key-a", "key-b", "key-a", "key-b"]
        )
        reqs = [
            _make_req(rid="r0", routing_key="key-b"),  # freq=2
            _make_req(rid="r1", routing_key="key-a"),  # freq=3
        ]
        SchedulePolicy._sort_by_routing_key(reqs, running_batch)
        self.assertEqual(reqs[0].rid, "r1")
        self.assertEqual(reqs[1].rid, "r0")

    def test_non_matching_keys_go_after_matching(self):
        reqs = [
            _make_req(rid="r0", routing_key="key-z"),
            _make_req(rid="r1", routing_key="key-a"),
            _make_req(rid="r2", routing_key=None),
        ]
        SchedulePolicy._sort_by_routing_key(reqs, self._make_running_batch(["key-a"]))
        self.assertEqual(reqs[0].rid, "r1")
        self.assertIn("r0", {r.rid for r in reqs[1:]})
        self.assertIn("r2", {r.rid for r in reqs[1:]})

    def test_no_matching_keys_in_waiting_queue(self):
        reqs = [
            _make_req(rid="r0", routing_key="key-a"),
            _make_req(rid="r1", routing_key="key-b"),
        ]
        original_order = [r.rid for r in reqs]
        SchedulePolicy._sort_by_routing_key(
            reqs, self._make_running_batch(["key-x", "key-x"])
        )
        self.assertEqual(sorted(r.rid for r in reqs), sorted(original_order))


class TestCalcPriority(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_fcfs_returns_false(self):
        sp = _make_schedule_policy("fcfs")
        reqs = [_make_req(rid=str(i), wait_time=float(i)) for i in range(5)]
        self.assertFalse(sp.calc_priority(reqs))

    def test_fcfs_without_priority_leaves_order_unchanged(self):
        sp = _make_schedule_policy("fcfs", enable_priority=False)
        reqs = [
            _make_req(rid="r0", priority=3, wait_time=0.0),
            _make_req(rid="r1", priority=1, wait_time=1.0),
        ]
        sp.calc_priority(reqs)
        self.assertEqual(reqs[0].rid, "r0")

    def test_fcfs_with_priority_high_first_sorts_by_priority(self):
        sp = _make_schedule_policy("fcfs", enable_priority=True, low_first=False)
        reqs = [
            _make_req(rid="r0", priority=1, wait_time=0.0),
            _make_req(rid="r1", priority=3, wait_time=1.0),
            _make_req(rid="r2", priority=2, wait_time=2.0),
        ]
        sp.calc_priority(reqs)
        self.assertEqual([r.priority for r in reqs], [3, 2, 1])

    def test_fcfs_with_priority_low_first_sorts_by_priority(self):
        sp = _make_schedule_policy("fcfs", enable_priority=True, low_first=True)
        reqs = [
            _make_req(rid="r0", priority=3, wait_time=0.0),
            _make_req(rid="r1", priority=1, wait_time=1.0),
            _make_req(rid="r2", priority=2, wait_time=2.0),
        ]
        sp.calc_priority(reqs)
        self.assertEqual([r.priority for r in reqs], [1, 2, 3])

    def test_lpm_large_queue_falls_back_no_prefix_computed(self):
        sp = _make_schedule_policy("lpm")
        reqs = [_make_req(rid=str(i), wait_time=float(i)) for i in range(129)]
        self.assertFalse(sp.calc_priority(reqs))

    def test_lpm_small_queue_returns_true_and_sorts_by_prefix(self):
        tree_cache = _make_tree_cache(disable=False)
        tree_cache.match_prefix.side_effect = [
            MatchResult(
                device_indices=torch.arange(1, dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                host_hit_length=0,
            ),
            MatchResult(
                device_indices=torch.arange(5, dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                host_hit_length=0,
            ),
        ]
        sp = SchedulePolicy(
            policy="lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        reqs = [
            _make_req(rid="short", origin_input_ids=[1]),
            _make_req(rid="long", origin_input_ids=[1, 2, 3, 4, 5]),
        ]
        self.assertTrue(sp.calc_priority(reqs))
        self.assertEqual(reqs[0].rid, "long")
        self.assertEqual(reqs[1].rid, "short")

    def test_lof_returns_false_and_sorts_by_max_new_tokens_desc(self):
        sp = _make_schedule_policy("lof")
        reqs = [
            _make_req(rid="r0", max_new_tokens=50),
            _make_req(rid="r1", max_new_tokens=200),
            _make_req(rid="r2", max_new_tokens=100),
        ]
        self.assertFalse(sp.calc_priority(reqs))
        self.assertEqual(reqs[0].rid, "r1")
        self.assertEqual(reqs[1].rid, "r2")
        self.assertEqual(reqs[2].rid, "r0")

    def test_routing_key_policy_returns_false(self):
        sp = _make_schedule_policy("routing-key")
        running_batch = MagicMock()
        running_batch.reqs = [SimpleNamespace(routing_key="key-a")]
        reqs = [
            _make_req(rid="r0", routing_key="key-a"),
            _make_req(rid="r1", routing_key="key-b"),
        ]
        self.assertFalse(sp.calc_priority(reqs, running_batch=running_batch))
        self.assertEqual(reqs[0].rid, "r0")

    def test_routing_key_policy_no_running_batch_is_noop(self):
        sp = _make_schedule_policy("routing-key")
        reqs = [
            _make_req(rid="r0", routing_key="key-b"),
            _make_req(rid="r1", routing_key="key-a"),
        ]
        original = [r.rid for r in reqs]
        sp.calc_priority(reqs, running_batch=None)
        self.assertEqual([r.rid for r in reqs], original)


if __name__ == "__main__":
    unittest.main()
