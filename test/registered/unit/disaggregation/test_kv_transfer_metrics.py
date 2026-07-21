"""CPU unit tests for measured KV transfer speed accounting (CommonKVManager /
CommonKVSender). Covers accumulation, the measured-vs-fallback branch of
get_transfer_metric (incl. the dummy-CP short-circuit), record cleanup on
failure/abort resurrection, and the lock that serializes _record_transfer
against teardown."""

import threading
import unittest

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVSender,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_manager(kv_item_lens_sum=10, state_item_lens_sum=3, replica_factor=1):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.request_status = {}
    mgr._transfer_records = {}
    mgr.status_record_lock = threading.Lock()
    mgr.kv_item_lens_sum = kv_item_lens_sum
    mgr.state_item_lens_sum = state_item_lens_sum
    mgr._kv_replica_factor = replica_factor
    mgr.is_dummy_cp_rank = False
    return mgr


def _make_sender(mgr, room, num_kv_indices=5, num_state_indices=2):
    sender = CommonKVSender.__new__(CommonKVSender)
    sender.bootstrap_room = room
    sender.kv_mgr = mgr
    sender._transfer_metric = KVTransferMetric()
    sender._transfer_num_kv_indices = num_kv_indices
    sender._transfer_num_state_indices = num_state_indices
    return sender


class TestTransferMetric(unittest.TestCase):
    def test_transfer_bytes(self):
        mgr = _make_manager(kv_item_lens_sum=10, state_item_lens_sum=3)
        self.assertEqual(mgr._transfer_bytes(5, 2), 5 * 10 + 2 * 3)

    def test_record_transfer_accumulates(self):
        mgr = _make_manager()
        room = 1
        mgr.request_status[room] = KVPoll.Transferring
        mgr._record_transfer(room, mgr._transfer_bytes(5, 2), 0.1)
        mgr._record_transfer(room, mgr._transfer_bytes(3, 0), 0.2)
        record = mgr._transfer_records[room]
        self.assertAlmostEqual(record.total_elapsed_s, 0.3)
        self.assertEqual(
            record.total_bytes, mgr._transfer_bytes(5, 2) + mgr._transfer_bytes(3, 0)
        )

    def test_record_transfer_skips_nonpositive(self):
        mgr = _make_manager()
        room = 1
        mgr.request_status[room] = KVPoll.Transferring
        mgr._record_transfer(room, mgr._transfer_bytes(5, 2), 0.0)  # zero elapsed
        mgr._record_transfer(room, 0, 0.1)  # zero bytes
        self.assertNotIn(room, mgr._transfer_records)

    def test_measured_record_pops_and_ignores_replica_factor(self):
        # Regression: on the measured path _record_transfer accumulates
        # bytes once per destination rank (called inside the per-dst-rank send
        # loop), so replication is already baked into record.total_bytes. For
        # MLA (replica_factor = required_dst_info_num > 1) the buggy code
        # multiplied by the factor again, over-counting bytes/speed by that
        # factor. The measured total must equal the raw recorded bytes, and the
        # record must be popped so a reused room can't inherit it.
        mgr = _make_manager(replica_factor=4)
        room = 42
        mgr.request_status[room] = KVPoll.Transferring
        # Two dst ranks recorded -> replication already summed into the record.
        mgr._record_transfer(room, mgr._transfer_bytes(5, 2), 0.1)
        mgr._record_transfer(room, mgr._transfer_bytes(5, 2), 0.15)
        sender = _make_sender(mgr, room)

        metric = sender.get_transfer_metric()

        self.assertAlmostEqual(metric.transfer_latency_s, 0.25)
        self.assertEqual(metric.transfer_total_bytes, 2 * mgr._transfer_bytes(5, 2))
        self.assertNotIn(room, mgr._transfer_records)

    def test_fallback_scales_by_replica_factor(self):
        # Fallback path: _transfer_num_kv/state_indices is a single source-side
        # count with no dst-rank dimension, so it must be scaled by the replica
        # factor to reflect the KV being replicated to each dst rank (MLA).
        mgr = _make_manager(replica_factor=4)
        room = 7
        sender = _make_sender(mgr, room, num_kv_indices=4, num_state_indices=1)

        metric = sender.get_transfer_metric()

        self.assertIsNone(metric.transfer_latency_s)
        self.assertEqual(metric.transfer_total_bytes, 4 * mgr._transfer_bytes(4, 1))

    def test_dummy_cp_rank_skips_estimate_and_replica_lookup(self):
        # Regression: dummy CP ranks transfer no KV and never bootstrap, so
        # _kv_replica_factor never resolves. The pre-guard code fell through to
        # the fallback estimate and called get_kv_replica_factor(), logging a
        # spurious "called before resolve_kv_replica_factor" warning on every
        # such rank. The guard must short-circuit to the zero-default metric
        # without touching the replica factor.
        mgr = _make_manager(replica_factor=None)
        mgr.is_dummy_cp_rank = True

        def _must_not_call():
            raise AssertionError(
                "get_kv_replica_factor must not be called on dummy CP ranks"
            )

        mgr.get_kv_replica_factor = _must_not_call
        sender = _make_sender(mgr, room=1, num_kv_indices=4, num_state_indices=1)

        metric = sender.get_transfer_metric()

        self.assertIsNone(metric.transfer_latency_s)
        self.assertIsNone(metric.transfer_total_bytes)

    def test_status_record_lock_serializes_teardown_against_transfer(self):
        # Regression: a send-worker transfer that passed _record_transfer's
        # guard must not run its setdefault after a concurrent
        # update_status(Failed) flips the status and pops the record, or it
        # leaks a record across bootstrap_room reuse. Forced deterministically:
        # the worker is parked inside its critical section (at setdefault) while
        # update_status runs; the lock must make update_status block until the
        # worker releases.
        worker_in_cs = threading.Event()
        release_worker = threading.Event()

        class _PausingRecords(dict):
            def setdefault(self, *args, **kwargs):
                worker_in_cs.set()
                release_worker.wait(timeout=5)
                return super().setdefault(*args, **kwargs)

        mgr = _make_manager()
        room = 55
        mgr._transfer_records = _PausingRecords()
        mgr.request_status[room] = KVPoll.Transferring

        worker = threading.Thread(
            target=mgr._record_transfer, args=(room, mgr._transfer_bytes(5, 2), 0.1)
        )
        worker.start()
        self.assertTrue(worker_in_cs.wait(timeout=5))  # parked mid-critical-section

        update_done = threading.Event()

        def _teardown():
            mgr.update_status(room, KVPoll.Failed)
            update_done.set()

        updater = threading.Thread(target=_teardown)
        updater.start()
        # Fixed: update_status is blocked on the lock. Buggy: it completes now.
        self.assertFalse(update_done.wait(timeout=0.3))

        release_worker.set()
        worker.join(timeout=5)
        updater.join(timeout=5)
        self.assertTrue(update_done.is_set())

        # Whatever the interleaving, the failure teardown leaves no record.
        self.assertNotIn(room, mgr._transfer_records)

    def test_clear_drains_transfer_record(self):
        # clear() tears the room down and must not leave a stale transfer record
        # behind; a reused bootstrap_room would otherwise inherit its bytes/time.
        mgr = _make_manager()
        room = 99
        mgr.request_status[room] = KVPoll.Transferring
        mgr._record_transfer(room, mgr._transfer_bytes(5, 2), 0.1)
        self.assertIn(room, mgr._transfer_records)
        sender = _make_sender(mgr, room)

        sender.clear()

        self.assertNotIn(room, mgr.request_status)
        self.assertNotIn(room, mgr._transfer_records)

    def test_record_transfer_does_not_resurrect_untracked_room(self):
        # A late in-flight transfer must not re-create a record (via setdefault) for
        # a room that is no longer tracked, which nothing would drain -- leaking
        # _transfer_records across bootstrap_room reuse. Both guard branches:
        #   Failed: abort/failure race popped the record on another thread.
        #   None:   on Success, clear() popped request_status before
        #           get_transfer_metric drained the record, so status reads None.
        for status in (KVPoll.Failed, None):
            with self.subTest(status=status):
                mgr = _make_manager()
                room = 13
                if status is not None:
                    mgr.request_status[room] = status
                else:
                    self.assertNotIn(room, mgr.request_status)

                mgr._record_transfer(room, mgr._transfer_bytes(5, 2), 0.1)

                self.assertNotIn(room, mgr._transfer_records)


if __name__ == "__main__":
    unittest.main()
