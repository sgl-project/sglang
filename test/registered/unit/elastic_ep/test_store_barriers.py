"""Unit tests for the elastic EP TCPStore barriers — no server, no model loading.

These run against a real in-process ``TCPStore`` rather than a fake, because the
properties under test (leader election, epoch reuse, round keying) are all about how
the counters behave under concurrent ``add``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import threading
import unittest
from datetime import timedelta

import torch.distributed as dist

import sglang.srt.elastic_ep.elastic_ep as eep
from sglang.srt.elastic_ep.elastic_ep import (
    _BARRIER_NS,
    _StoreBarrier,
    cohort_vote_via_store,
    share_expert_map_via_store,
)
from sglang.test.test_utils import CustomTestCase


class StoreBackedTestCase(CustomTestCase):
    """Gives each test a fresh TCPStore installed as the process-global one."""

    def setUp(self):
        super().setUp()
        self._store = dist.TCPStore(
            "127.0.0.1",
            0,
            1,
            True,
            timeout=timedelta(seconds=30),
            wait_for_workers=False,
        )
        self._orig = eep.get_global_tcp_store
        eep.get_global_tcp_store = lambda: self._store
        # Barrier epochs are tracked per process; reset so tests do not see each other.
        self._orig_cycles = dict(eep._last_local_cycle_id)
        for ns in eep._last_local_cycle_id:
            eep._last_local_cycle_id[ns] = 0

    def tearDown(self):
        eep.get_global_tcp_store = self._orig
        eep._last_local_cycle_id.update(self._orig_cycles)
        del self._store
        super().tearDown()


class TestStoreBarrier(StoreBackedTestCase):
    NS = "world"

    def test_single_rank_barrier_reaches_its_target(self):
        state = _StoreBarrier.post(self._store, 0, self.NS, 1)
        self.assertIsNotNone(state)
        reached, count = state.check(self._store, 0.0)
        self.assertTrue(reached)
        self.assertEqual(count, 1)
        state.consume()

    def test_barrier_does_not_complete_below_its_target(self):
        state = _StoreBarrier.post(self._store, 0, self.NS, 2)
        reached, count = state.check(self._store, 0.0)
        self.assertFalse(reached)
        self.assertEqual(count, 1)
        state.consume()

    def test_first_arrival_is_the_leader_and_peers_follow(self):
        first = _StoreBarrier.post(self._store, 0, self.NS, 2)
        self.assertEqual(first.arrival, 1, "first poster should lead")

        # The two ranks share a process here, so they also share the module-global
        # cycle marker the leader just advanced. A real follower is a separate
        # process that has not seen the new id yet, and it waits for one strictly
        # greater than what it holds. Rewind to stand in for that.
        eep._last_local_cycle_id[self.NS] = first.epoch - 1
        second = _StoreBarrier.post(self._store, 1, self.NS, 2)
        self.assertIsNotNone(second, "follower should adopt the leader's epoch")
        self.assertNotEqual(second.arrival, 1)
        # Both must agree on the epoch, or they wait on different ready keys.
        self.assertEqual(first.epoch, second.epoch)
        reached, count = first.check(self._store, 0.0)
        self.assertTrue(reached)
        self.assertEqual(count, 2)
        first.consume()

    def test_consume_lets_a_later_cycle_elect_a_leader_again(self):
        """This is the leak that wedges every later scale if consume is skipped."""
        first = _StoreBarrier.post(self._store, 0, self.NS, 1)
        self.assertEqual(first.arrival, 1)
        first.check(self._store, 0.0)
        first.consume()

        second = _StoreBarrier.post(self._store, 0, self.NS, 1)
        self.assertIsNotNone(second)
        self.assertEqual(second.arrival, 1, "ARRIVAL was not reset by consume")
        self.assertGreater(second.epoch, first.epoch, "epochs must not be reused")
        reached, _ = second.check(self._store, 0.0)
        self.assertTrue(reached)
        second.consume()

    def test_skipping_consume_blocks_the_next_leader_election(self):
        first = _StoreBarrier.post(self._store, 0, self.NS, 1)
        first.check(self._store, 0.0)
        # Deliberately no consume(), which is what abandon() exists to prevent.
        second = _StoreBarrier.post(self._store, 0, self.NS, 1)
        self.assertNotEqual(
            getattr(second, "arrival", None),
            1,
            "a leaked ARRIVAL should stop the next cycle from electing a leader",
        )

    def test_consecutive_cycles_use_distinct_ready_keys(self):
        first = _StoreBarrier.post(self._store, 0, self.NS, 1)
        first.check(self._store, 0.0)
        first.consume()
        second = _StoreBarrier.post(self._store, 0, self.NS, 1)
        self.assertNotEqual(
            first.ready_key, second.ready_key, "chained scales must not share a key"
        )
        second.consume()

    def test_namespaces_are_independent(self):
        world = _StoreBarrier.post(self._store, 0, "world", 1)
        nixl = _StoreBarrier.post(self._store, 0, "nixl", 1)
        self.assertEqual(world.arrival, 1)
        self.assertEqual(nixl.arrival, 1, "a nixl barrier must not see world's counter")
        self.assertNotEqual(world.ready_key, nixl.ready_key)
        world.consume()
        nixl.consume()

    def test_every_declared_namespace_can_post(self):
        for ns in _BARRIER_NS:
            state = _StoreBarrier.post(self._store, 0, ns, 1)
            self.assertIsNotNone(state, f"namespace {ns} failed to post")
            state.consume()


class TestCohortVote(StoreBackedTestCase):
    def test_single_voter_passes_through(self):
        self.assertTrue(cohort_vote_via_store(True, 1, tag="unit"))
        self.assertFalse(cohort_vote_via_store(False, 1, tag="unit"))

    def test_unanimous_yes_passes(self):
        results = self._vote_concurrently([True, True, True], tag="yes")
        self.assertEqual(results, [True, True, True])

    def test_one_no_fails_the_whole_cohort(self):
        """A per-rank bail would strand peers, so the verdict has to be collective."""
        results = self._vote_concurrently([True, False, True], tag="no")
        self.assertEqual(results, [False, False, False])

    def test_consecutive_votes_do_not_share_a_round(self):
        self.assertEqual(self._vote_concurrently([True, True], tag="r"), [True, True])
        # A second vote on the same tag must start a fresh round, not read the first.
        self.assertEqual(
            self._vote_concurrently([True, False], tag="r"), [False, False]
        )

    def _vote_concurrently(self, votes, *, tag):
        size = len(votes)
        results = [None] * size
        errors = []

        def run(index):
            try:
                results[index] = cohort_vote_via_store(
                    votes[index], size, tag=tag, timeout_s=30.0
                )
            except Exception as exc:  # surface instead of hanging the assertion
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [], f"vote raised: {errors}")
        return results


class TestExpertMapInbox(StoreBackedTestCase):
    def test_source_publishes_and_reader_receives(self):
        import torch

        payload = torch.tensor([[3, 1, 2, 0]], dtype=torch.int64)
        self.assertTrue(
            share_expert_map_via_store(
                payload, is_src=True, cohort_ranks=range(2), group_rank=0
            )
        )
        received = torch.zeros_like(payload)
        self.assertTrue(
            share_expert_map_via_store(
                received, is_src=False, cohort_ranks=range(2), group_rank=1
            )
        )
        self.assertTrue(torch.equal(received, payload))

    def test_sparse_cohort_is_addressed_by_rank_id(self):
        """After a fault the live ranks stop being 0..n-1, and the inbox is named for
        the reader's own id. Width 6 with rank 4 down leaves five live ranks and a top
        id of 5, so anything driven off the count writes to the rank that just died and
        never to rank 5, which then waits on an inbox nobody fills."""
        import torch

        live = (0, 1, 2, 3, 5)
        payload = torch.tensor([[3, 1, 2, 0]], dtype=torch.int64)
        self.assertTrue(
            share_expert_map_via_store(
                payload, is_src=True, cohort_ranks=live, group_rank=0
            )
        )
        self.assertTrue(self._store.check(["sglang_expert_map_to_r5"]))
        self.assertFalse(self._store.check(["sglang_expert_map_to_r4"]))

        received = torch.zeros_like(payload)
        self.assertTrue(
            share_expert_map_via_store(
                received, is_src=False, cohort_ranks=live, group_rank=5
            )
        )
        self.assertTrue(torch.equal(received, payload))

    def test_single_rank_cohort_needs_no_store_round(self):
        import torch

        payload = torch.tensor([[0, 1]], dtype=torch.int64)
        self.assertFalse(
            share_expert_map_via_store(
                payload, is_src=True, cohort_ranks=range(1), group_rank=0
            )
        )


if __name__ == "__main__":
    unittest.main()
