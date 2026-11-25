"""Elastic EP TCPStore barriers, against a real in-process TCPStore: leader election,
epoch reuse and round keying are about how counters behave under concurrent add."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import threading
import time
import unittest
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist

import sglang.srt.elastic_ep.elastic_ep as eep
from sglang.srt.elastic_ep.elastic_ep import (
    _BARRIER_NS,
    _close_vote_round,
    _StoreBarrier,
    cohort_vote_via_store,
    reset_barriers_after_failure,
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
        cycles = eep._last_local_cycle_id
        for patch in (
            mock.patch.object(eep, "get_global_tcp_store", lambda: self._store),
            # Barrier epochs are per process; reset so tests do not see each other.
            mock.patch.dict(cycles, dict.fromkeys(cycles, 0)),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        del self._store
        super().tearDown()


def _post(store, ns="world", target=1, rank=0):
    return _StoreBarrier.post(store, rank, ns, target)


class TestStoreBarrier(StoreBackedTestCase):
    def test_first_arrival_is_the_leader_and_peers_follow(self):
        first = _post(self._store, target=2)
        self.assertEqual(first.arrival, 1, "first poster should lead")
        self.assertEqual(first.check(self._store, 0.0), (False, 1), "below its target")
        # Both ranks share this process's cycle marker, which the leader just advanced;
        # a real follower has not seen the new id yet, so rewind to stand in for one.
        eep._last_local_cycle_id["world"] = first.epoch - 1
        second = _post(self._store, target=2, rank=1)
        self.assertIsNotNone(second, "follower should adopt the leader's epoch")
        self.assertNotEqual(second.arrival, 1)
        # Both must agree on the epoch, or they wait on different ready keys.
        self.assertEqual(first.epoch, second.epoch)
        self.assertEqual(first.check(self._store, 0.0), (True, 2))
        first.consume()

    def test_consume_lets_a_later_cycle_elect_a_leader_again(self):
        """This is the leak that wedges every later scale if consume is skipped."""
        first = _post(self._store)
        self.assertEqual(first.arrival, 1)
        self.assertEqual(first.check(self._store, 0.0), (True, 1))
        first.consume()
        second = _post(self._store)
        self.assertIsNotNone(second)
        self.assertEqual(second.arrival, 1, "ARRIVAL was not reset by consume")
        self.assertGreater(second.epoch, first.epoch, "epochs must not be reused")
        self.assertNotEqual(
            first.ready_key, second.ready_key, "chained scales must not share a key"
        )
        reached, _ = second.check(self._store, 0.0)
        self.assertTrue(reached)
        second.consume()

    def test_skipping_consume_blocks_the_next_leader_election(self):
        first = _post(self._store)
        first.check(self._store, 0.0)
        # Deliberately no consume(), which is what abandon() exists to prevent.
        second = _post(self._store)
        self.assertNotEqual(
            getattr(second, "arrival", None),
            1,
            "a leaked ARRIVAL should stop the next cycle from electing a leader",
        )

    def test_failure_reset_restores_leader_election_after_a_dead_leader(self):
        """A leader that dies after posting never consumes; the reset on a failed
        or recovered scale has to stand in for it."""
        first = _post(self._store)
        with mock.patch.object(dist, "get_rank", return_value=0):
            reset_barriers_after_failure()
        second = _post(self._store)
        self.assertIsNotNone(second)
        self.assertEqual(second.arrival, 1, "ARRIVAL was not reset")
        self.assertGreater(second.epoch, first.epoch, "epochs must not be reused")
        second.consume()

    def test_failure_reset_on_a_peer_leaves_the_counters(self):
        """Only rank 0 clears ARRIVAL, so a peer resetting late cannot wipe a round
        rank 0 has already started."""
        first = _post(self._store, target=2)
        with mock.patch.object(dist, "get_rank", return_value=1):
            reset_barriers_after_failure()
        self.assertEqual(int(self._store.add(_BARRIER_NS["world"][0], 0)), 1)
        self.assertEqual(eep._last_local_cycle_id["world"], first.epoch)

    def test_every_declared_namespace_posts_independently(self):
        states = [_post(self._store, ns) for ns in _BARRIER_NS]
        self.assertNotIn(None, states, "a namespace failed to post")
        # Each is its namespace's first arrival: none sees another's counter.
        self.assertEqual([s.arrival for s in states], [1] * len(states))
        self.assertEqual(len({s.ready_key for s in states}), len(states))
        for state in states:
            state.consume()


class TestCohortVote(StoreBackedTestCase):
    def test_single_voter_passes_through(self):
        self.assertTrue(cohort_vote_via_store(True, 1, tag="unit"))
        self.assertFalse(cohort_vote_via_store(False, 1, tag="unit"))

    def test_one_no_fails_the_whole_cohort(self):
        """A per-rank bail would strand peers, so the verdict has to be collective."""
        results = self._vote_concurrently([True, False, True], tag="no")
        self.assertEqual(results, [False, False, False])

    def test_unanimous_yes_passes_and_the_next_vote_starts_a_new_round(self):
        self.assertEqual(self._vote_concurrently([True, True], tag="r"), [True, True])
        # A second vote on the same tag must start a fresh round, not read the first.
        self.assertEqual(
            self._vote_concurrently([True, False], tag="r"), [False, False]
        )

    def test_a_timeout_aborts_the_round_for_the_late_voter_too(self):
        """A late voter must not read a finished round the timed-out rank gave up on."""
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            cohort_vote_via_store(True, 2, tag="late", timeout_s=0.2)
        # The abort bumped the epoch, so the late voter and the retry meet afresh.
        self.assertEqual(
            self._vote_concurrently([True, True], tag="late"), [True, True]
        )

    def test_voters_retrying_an_aborted_round_meet_the_late_voter(self):
        """N-1 voters abort, catch and retry before the Nth enters, as the grow admit
        does; a retry spending the aborted round's leftover slot splits the cohort."""
        results, errors = [None] * 3, []
        retried = threading.Barrier(3)

        def early(index):
            try:
                with self.assertRaisesRegex(RuntimeError, "aborted"):
                    cohort_vote_via_store(True, 3, tag="retry", timeout_s=0.2)
                retried.wait(timeout=30)
                results[index] = cohort_vote_via_store(
                    True, 3, tag="retry", timeout_s=30.0
                )
            except Exception as exc:  # surface instead of hanging the assertion
                errors.append(exc)

        threads = [threading.Thread(target=early, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        # The Nth voter enters only once both early voters are into their retry.
        retried.wait(timeout=30)
        time.sleep(0.5)
        results[2] = cohort_vote_via_store(True, 3, tag="retry", timeout_s=30.0)
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [], f"vote raised: {errors}")
        self.assertEqual(results, [True, True, True])

    def test_an_abort_after_the_last_vote_keeps_the_verdict(self):
        key = "unit_close"
        self._store.add(f"{key}_yes", 2)
        for _ in range(2):
            _close_vote_round(
                self._store, key, int(self._store.add(f"{key}_cast", 1)), 2
            )
        late = int(self._store.add(f"{key}_cast", eep._VOTE_ABORT_MARK))
        _close_vote_round(
            self._store, key, late, 2, abort=True, epoch_key=f"{key}_epoch"
        )
        self.assertEqual(self._store.get(f"{key}_verdict"), b"yes")
        # The round closed, so its voters stay in the epoch they agreed in.
        self.assertEqual(int(self._store.add(f"{key}_epoch", 0)), 0)

    def test_reshuffle_reads_a_vote_timeout_as_a_cohort_wide_no(self):
        from sglang.srt.eplb.eplb_manager import EPLBManager

        def timed_out(*args, **kwargs):
            raise RuntimeError("vote timed out")

        with mock.patch.object(eep, "cohort_vote_via_store", timed_out):
            self.assertFalse(EPLBManager._cohort_accepts(True, 2, "dims"))

    def _vote_concurrently(self, votes, *, tag):
        results, errors = [None] * len(votes), []

        def run(index):
            try:
                results[index] = cohort_vote_via_store(
                    votes[index], len(votes), tag=tag, timeout_s=30.0
                )
            except Exception as exc:  # surface instead of hanging the assertion
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(votes))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [], f"vote raised: {errors}")
        return results


def _share(tensor, cohort, rank):
    """Rank 0 publishes the map; any other rank reads its own inbox."""
    return share_expert_map_via_store(
        tensor, is_src=rank == 0, cohort_ranks=cohort, group_rank=rank
    )


class TestExpertMapInbox(StoreBackedTestCase):
    def test_source_publishes_and_reader_receives(self):
        payload = torch.tensor([[3, 1, 2, 0]], dtype=torch.int64)
        received = torch.zeros_like(payload)
        self.assertTrue(_share(payload, range(2), 0))
        self.assertTrue(_share(received, range(2), 1))
        self.assertTrue(torch.equal(received, payload))

    def test_sparse_cohort_is_addressed_by_rank_id(self):
        """With rank 4 of 6 down, a count-driven inbox writes to rank 4, not rank 5."""
        live = (0, 1, 2, 3, 5)
        payload = torch.tensor([[3, 1, 2, 0]], dtype=torch.int64)
        self.assertTrue(_share(payload, live, 0))
        self.assertTrue(self._store.check(["sglang_expert_map_to_r5"]))
        self.assertFalse(self._store.check(["sglang_expert_map_to_r4"]))
        received = torch.zeros_like(payload)
        self.assertTrue(_share(received, live, 5))
        self.assertTrue(torch.equal(received, payload))

    def test_single_rank_cohort_needs_no_store_round(self):
        self.assertFalse(_share(torch.tensor([[0, 1]], dtype=torch.int64), range(1), 0))


if __name__ == "__main__":
    unittest.main()
