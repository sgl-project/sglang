"""Unit tests for orphan room lifecycle, admission guard, TTL sweeping, and thread-safe purge.

Tests Issue #39428 fix:
- Prevents unbounded memory growth from orphan bootstrap_room injection.
- Verifies admission pre-authorization guard.
- Verifies O(1) min-heap TTL sweep for stale / orphan rooms.
- Verifies bounded terminal status LRU cache to eliminate KeyError during client poll.
- Verifies centralized thread-safe _purge_room_state.
"""

import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_bare_manager(ttl: float = 15.0, max_pending: int = 4096):
    """Create a bare CommonKVManager carrying room lifecycle attributes without full __init__."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.request_status = {}
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = {}
    mgr._staging_ctx = None
    mgr.deferred_bootstrap = None
    mgr.failure_lock = threading.Lock()
    mgr.failure_records = {}
    mgr._send_records = None
    mgr._room_lock = threading.RLock()
    mgr._room_expiry_heap = []
    mgr._room_expiries = {}
    mgr._terminal_status_cache = OrderedDict()
    mgr._terminal_status_cache_max_size = 2048
    mgr._scheduled_prefill_rooms = set()
    mgr._prefill_admission_enabled = False
    mgr.room_ttl = ttl
    mgr.max_pending_rooms = max_pending
    return mgr


class TestOrphanRoomLifecycle(CustomTestCase):
    def test_admission_guard_default_backward_compatible(self):
        mgr = _make_bare_manager()
        # When guard is disabled, any room is accepted
        self.assertTrue(mgr.is_room_scheduled(12345))
        self.assertTrue(mgr.is_room_scheduled(99999))

    def test_admission_guard_strict_filtering(self):
        mgr = _make_bare_manager()
        mgr.enable_prefill_admission_guard()

        # Unscheduled room must be rejected
        self.assertFalse(mgr.is_room_scheduled(100))
        self.assertFalse(mgr.is_room_scheduled(200))

        # Register legitimate scheduled prefill request
        mgr.register_prefill_room(100)
        self.assertTrue(mgr.is_room_scheduled(100))
        self.assertFalse(mgr.is_room_scheduled(200))

        # Purging room also revokes its scheduled status
        mgr._purge_room_state(100, terminal_status=KVPoll.Success)
        self.assertFalse(mgr.is_room_scheduled(100))

    def test_purge_room_state_cleans_all_dictionaries(self):
        mgr = _make_bare_manager()
        room = 42

        mgr.register_prefill_room(room)
        mgr.request_status[room] = KVPoll.Transferring
        mgr.transfer_infos[room] = {"worker_0": SimpleNamespace()}
        mgr.req_to_decode_prefix_len[room] = 128
        mgr._deferred_ack_targets[room] = ("127.0.0.1", 9999)

        # Purge room
        mgr._purge_room_state(room, terminal_status=KVPoll.Success)

        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, mgr._deferred_ack_targets)
        self.assertNotIn(room, mgr.request_status)
        self.assertNotIn(room, mgr._scheduled_prefill_rooms)

        # check_status should gracefully return the terminal status from LRU cache without KeyError
        self.assertEqual(mgr.check_status(room), KVPoll.Success)

    def test_terminal_status_cache_lru_bounding(self):
        mgr = _make_bare_manager()
        mgr._terminal_status_cache_max_size = 3
        # Import OrderedDict for exact LRU behavior
        from collections import OrderedDict

        mgr._terminal_status_cache = OrderedDict()

        for room in [1, 2, 3]:
            mgr._record_terminal_status(room, KVPoll.Success)

        self.assertEqual(len(mgr._terminal_status_cache), 3)
        self.assertIn(1, mgr._terminal_status_cache)

        # Adding 4th room should evict oldest (room 1)
        mgr._record_terminal_status(4, KVPoll.Failed)
        self.assertEqual(len(mgr._terminal_status_cache), 3)
        self.assertNotIn(1, mgr._terminal_status_cache)
        self.assertIn(2, mgr._terminal_status_cache)
        self.assertIn(3, mgr._terminal_status_cache)
        self.assertIn(4, mgr._terminal_status_cache)

    def test_min_heap_ttl_eviction(self):
        mgr = _make_bare_manager(ttl=10.0)
        base_time = 1000.0

        # Inject two rooms at base_time
        room1 = 101
        room2 = 102
        mgr.transfer_infos[room1] = {"w": 1}
        mgr.transfer_infos[room2] = {"w": 1}
        mgr.request_status[room1] = KVPoll.WaitingForInput
        mgr.request_status[room2] = KVPoll.Bootstrapping

        with unittest.mock.patch("time.time", return_value=base_time):
            mgr.record_room_active(room1, ttl=10.0)
            mgr.record_room_active(room2, ttl=20.0)

        # At base_time + 5s, neither should expire
        evicted = mgr.sweep_stale_rooms(now=base_time + 5.0)
        self.assertEqual(evicted, 0)
        self.assertIn(room1, mgr.transfer_infos)
        self.assertIn(room2, mgr.transfer_infos)

        # At base_time + 12s, room1 should expire and be purged
        evicted = mgr.sweep_stale_rooms(now=base_time + 12.0)
        self.assertEqual(evicted, 1)
        self.assertNotIn(room1, mgr.transfer_infos)
        self.assertEqual(mgr.check_status(room1), KVPoll.Failed)
        self.assertIn(room2, mgr.transfer_infos)

        # At base_time + 25s, room2 should expire
        evicted = mgr.sweep_stale_rooms(now=base_time + 25.0)
        self.assertEqual(evicted, 1)
        self.assertNotIn(room2, mgr.transfer_infos)
        self.assertEqual(mgr.check_status(room2), KVPoll.Failed)

    def test_completed_rooms_not_erroneously_purged_by_sweep(self):
        mgr = _make_bare_manager(ttl=10.0)
        base_time = 1000.0
        room = 201

        mgr.transfer_infos[room] = {"w": 1}
        mgr.request_status[room] = KVPoll.WaitingForInput

        with unittest.mock.patch("time.time", return_value=base_time):
            mgr.record_room_active(room, ttl=10.0)

        # Request succeeds before TTL expiry
        mgr.update_status(room, KVPoll.Success)
        # Clear/purge on success
        mgr._purge_room_state(room, terminal_status=KVPoll.Success)

        # At base_time + 15s, sweep runs
        evicted = mgr.sweep_stale_rooms(now=base_time + 15.0)
        self.assertEqual(evicted, 0)
        # Status remains Success in terminal cache
        self.assertEqual(mgr.check_status(room), KVPoll.Success)

    def test_scheduled_room_not_evicted_by_orphan_ttl_sweep(self):
        mgr = _make_bare_manager(ttl=10.0)
        base_time = 1000.0
        room = 202

        # Decode arrives before prefill creates the sender
        with patch("time.time", return_value=base_time):
            self.assertTrue(mgr.admit_bootstrap_room(room))
        mgr.request_status[room] = KVPoll.WaitingForInput

        # Prefill schedules the room shortly after
        mgr.register_prefill_room(room)

        # Even past orphan TTL, a scheduled room must not be evicted by sweep
        evicted = mgr.sweep_stale_rooms(now=base_time + 20.0)
        self.assertEqual(evicted, 0)
        self.assertIn(room, mgr.transfer_infos)

    def test_max_pending_rooms_blocks_orphans_but_allows_scheduled_rooms(self):
        mgr = _make_bare_manager(ttl=10.0, max_pending=2)
        base_time = 1000.0

        with patch("time.time", return_value=base_time):
            self.assertTrue(mgr.admit_bootstrap_room(1))
            self.assertTrue(mgr.admit_bootstrap_room(2))
            # Unscheduled 3rd room is rejected while cap is full
            self.assertFalse(mgr.admit_bootstrap_room(3))
            self.assertNotIn(3, mgr.transfer_infos)

            # A legitimate scheduled prefill room is still admitted
            mgr.register_prefill_room(4)
            self.assertTrue(mgr.admit_bootstrap_room(4))
            self.assertIn(4, mgr.transfer_infos)

        # Once orphan rooms expire, admitting a new room sweeps them first
        with patch("time.time", return_value=base_time + 15.0):
            self.assertTrue(mgr.admit_bootstrap_room(5))
            self.assertNotIn(1, mgr.transfer_infos)
            self.assertNotIn(2, mgr.transfer_infos)
            self.assertIn(4, mgr.transfer_infos)
            self.assertIn(5, mgr.transfer_infos)

    def test_room_reuse_not_evicted_by_stale_heap_entry(self):
        mgr = _make_bare_manager(ttl=10.0)
        room = 203

        with patch("time.time", return_value=1000.0):
            self.assertTrue(mgr.admit_bootstrap_room(room))
        mgr._purge_room_state(room, terminal_status=KVPoll.Success)

        # Room ID is reused before the first lifecycle's expiry (1010.0)
        with patch("time.time", return_value=1008.0):
            self.assertTrue(mgr.admit_bootstrap_room(room))

        # At 1011.0, the stale (1010.0, 203) heap entry pops but must not evict
        # the reused room whose new expiry is 1018.0
        evicted = mgr.sweep_stale_rooms(now=1011.0)
        self.assertEqual(evicted, 0)
        self.assertIn(room, mgr.transfer_infos)

        # At 1019.0, the reused room expires normally
        evicted = mgr.sweep_stale_rooms(now=1019.0)
        self.assertEqual(evicted, 1)
        self.assertNotIn(room, mgr.transfer_infos)


class _TestSender(CommonKVSender):
    def failure_exception(self):
        pass

    def poll(self):
        pass


class TestOrphanRoomLifecycleSender(CustomTestCase):
    def test_sender_clear_purges_room_state(self):
        mgr = _make_bare_manager()
        room = 301
        sender = _TestSender.__new__(_TestSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = room
        sender.conclude_state = KVPoll.Success

        mgr.transfer_infos[room] = {"w": 1}
        mgr.req_to_decode_prefix_len[room] = 64
        mgr.request_status[room] = KVPoll.Success

        sender.clear()

        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        self.assertEqual(mgr.check_status(room), KVPoll.Success)

    def test_sender_clear_preserves_deferred_ack_target_until_drained(self):
        mgr = _make_bare_manager()
        room = 302
        sender = _TestSender.__new__(_TestSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = room
        sender.conclude_state = KVPoll.Failed

        mgr.transfer_infos[room] = {"w": 1}
        mgr.request_status[room] = KVPoll.Failed
        mgr._staging_outstanding[room] = 1
        target = SimpleNamespace(ip="127.0.0.1", port=9999, generation=1)
        mgr._deferred_ack_targets[room] = {("127.0.0.1", 9999): target}
        mgr._send_abort_acks = Mock()

        # While a write is still in flight, clear() must keep the deferred ACK target
        sender.clear()
        self.assertIn(room, mgr._deferred_ack_targets)
        mgr._send_abort_acks.assert_not_called()

        # Once the write drains, _maybe_ack_drained_abort sends the ACK and clears it
        mgr._staging_outstanding[room] = 0
        mgr._maybe_ack_drained_abort(room)
        mgr._send_abort_acks.assert_called_once_with(
            room, {("127.0.0.1", 9999): target}
        )
        self.assertNotIn(room, mgr._deferred_ack_targets)


if __name__ == "__main__":
    unittest.main()
