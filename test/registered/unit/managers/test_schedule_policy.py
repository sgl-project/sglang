"""Unit tests for SchedulePolicy sorting helpers.

Covers the static sorting methods in ``schedule_policy.py`` that decide the
ordering of the waiting queue. They are pure functions over a list of Req-like
objects, so we test them in isolation without a server or GPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_policy import SchedulePolicy


def make_req(
    rid,
    prefix_len=0,
    max_new_tokens=16,
    priority=0,
    entry_time=0.0,
    routing_key=None,
):
    """Build a lightweight Req-like object with only the fields touched by sorting."""
    return SimpleNamespace(
        rid=rid,
        prefix_indices=list(range(prefix_len)),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        priority=priority,
        time_stats=SimpleNamespace(wait_queue_entry_time=entry_time),
        routing_key=routing_key,
    )


class TestSortByLongestPrefix(unittest.TestCase):
    def test_longer_prefix_scheduled_first(self):
        q = [
            make_req("a", prefix_len=3),
            make_req("b", prefix_len=10),
            make_req("c", prefix_len=0),
        ]
        SchedulePolicy._sort_by_longest_prefix(q, set())
        self.assertEqual([r.rid for r in q], ["b", "a", "c"])

    def test_deprioritized_requests_go_last(self):
        # "a" has the longest prefix but is deprioritized -> must go last.
        q = [make_req("a", prefix_len=10), make_req("b", prefix_len=2)]
        SchedulePolicy._sort_by_longest_prefix(q, {"a"})
        self.assertEqual([r.rid for r in q], ["b", "a"])

    def test_empty_queue_is_noop(self):
        q = []
        SchedulePolicy._sort_by_longest_prefix(q, set())
        self.assertEqual(q, [])

    def test_equal_key_preserves_original_order(self):
        # Python's sort is stable; equal keys keep arrival order.
        q = [make_req("a", prefix_len=0), make_req("b", prefix_len=0)]
        SchedulePolicy._sort_by_longest_prefix(q, set())
        self.assertEqual([r.rid for r in q], ["a", "b"])


class TestSortByLongestOutput(unittest.TestCase):
    def test_longer_output_first_without_priority(self):
        q = [
            make_req("a", max_new_tokens=5),
            make_req("b", max_new_tokens=50),
            make_req("c", max_new_tokens=10),
        ]
        SchedulePolicy._sort_by_longest_output(
            q, enable_priority_scheduling=False, priority_sign=-1
        )
        self.assertEqual([r.rid for r in q], ["b", "c", "a"])

    def test_priority_dominates_then_output_length(self):
        # priority_sign=-1 (default) -> higher priority value first.
        q = [
            make_req("a", priority=1, max_new_tokens=100),
            make_req("b", priority=5, max_new_tokens=10),
            make_req("c", priority=5, max_new_tokens=20),
        ]
        SchedulePolicy._sort_by_longest_output(
            q, enable_priority_scheduling=True, priority_sign=-1
        )
        # priority-5 group first (ordered by longer output: c then b), then a.
        self.assertEqual([r.rid for r in q], ["c", "b", "a"])

    def test_low_priority_values_first_when_sign_flipped(self):
        # priority_sign=1 -> lower priority value first.
        q = [
            make_req("a", priority=5, max_new_tokens=10),
            make_req("b", priority=1, max_new_tokens=10),
        ]
        SchedulePolicy._sort_by_longest_output(
            q, enable_priority_scheduling=True, priority_sign=1
        )
        self.assertEqual([r.rid for r in q], ["b", "a"])


class TestSortByPriorityAndFcfs(unittest.TestCase):
    def test_higher_priority_value_first(self):
        # priority_sign=-1 (default) -> higher priority scheduled first.
        q = [
            make_req("a", priority=1, entry_time=0.0),
            make_req("b", priority=9, entry_time=0.0),
            make_req("c", priority=5, entry_time=0.0),
        ]
        SchedulePolicy._sort_by_priority_and_fcfs(q, priority_sign=-1)
        self.assertEqual([r.rid for r in q], ["b", "c", "a"])

    def test_fcfs_breaks_ties_on_equal_priority(self):
        # same priority -> earlier wait_queue_entry_time first.
        q = [
            make_req("a", priority=1, entry_time=3.0),
            make_req("b", priority=1, entry_time=1.0),
            make_req("c", priority=1, entry_time=2.0),
        ]
        SchedulePolicy._sort_by_priority_and_fcfs(q, priority_sign=-1)
        self.assertEqual([r.rid for r in q], ["b", "c", "a"])

    def test_low_priority_values_first_when_sign_flipped(self):
        # priority_sign=1 -> lower priority value first.
        q = [
            make_req("a", priority=9, entry_time=0.0),
            make_req("b", priority=1, entry_time=0.0),
        ]
        SchedulePolicy._sort_by_priority_and_fcfs(q, priority_sign=1)
        self.assertEqual([r.rid for r in q], ["b", "a"])


class TestSortByRoutingKey(unittest.TestCase):
    def _running_batch(self, keys):
        return SimpleNamespace(reqs=[SimpleNamespace(routing_key=k) for k in keys])

    def test_high_frequency_matching_key_first(self):
        running = self._running_batch(["k1", "k1", "k1", "k2"])
        q = [
            make_req("a", routing_key="k2"),
            make_req("b", routing_key="k1"),
            make_req("c", routing_key="kX"),
        ]
        SchedulePolicy._sort_by_routing_key(q, running)
        # k1 (count 3) first, then k2 (count 1), then unmatched kX last.
        self.assertEqual([r.rid for r in q], ["b", "a", "c"])

    def test_no_usable_keys_in_running_is_noop(self):
        # running batch has only falsy routing keys -> counter empty -> no sort.
        running = self._running_batch([None, None])
        q = [make_req("a", routing_key="k1"), make_req("b", routing_key="k2")]
        SchedulePolicy._sort_by_routing_key(q, running)
        self.assertEqual([r.rid for r in q], ["a", "b"])


class TestSortRandomly(unittest.TestCase):
    def test_shuffle_preserves_membership(self):
        q = [make_req("a"), make_req("b"), make_req("c")]
        SchedulePolicy._sort_randomly(q)
        self.assertEqual(len(q), 3)
        self.assertEqual({r.rid for r in q}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
