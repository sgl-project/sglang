import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 7


def _make_manager():
    mgr = object.__new__(CommonKVManager)
    mgr.request_status = {}
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = {}
    mgr.bootstrap_timeout = 300
    mgr._abandoned_rooms = {}
    mgr._abandoned_rooms_lock = threading.Lock()
    mgr.send_kv_status_message = MagicMock()
    return mgr


def _info(endpoint, is_dummy=False):
    return SimpleNamespace(endpoint=endpoint, dst_port=1, is_dummy=is_dummy)


def _register(mgr, room, required=1):
    # What a bootstrap thread does as each decode peer registers the room.
    infos = mgr.transfer_infos.setdefault(room, {})
    infos[len(infos)] = _info(f"decode{len(infos)}")
    if len(infos) == required:
        mgr.req_to_decode_prefix_len[room] = 0
        mgr.update_status(room, KVPoll.WaitingForInput)


def _failed_targets(mgr):
    return [
        c.kwargs["targets"]
        for c in mgr.send_kv_status_message.call_args_list
        if c.kwargs["status"] == KVPoll.Failed and c.kwargs["targets"]
    ]


class TestAbandonRoom(unittest.TestCase):
    def assert_dropped(self, mgr, room):
        self.assertNotIn(room, mgr.request_status)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)

    def test_abandon_after_registration(self):
        mgr = _make_manager()
        _register(mgr, ROOM)
        mgr.abandon_room(ROOM)
        self.assert_dropped(mgr, ROOM)
        self.assertEqual(_failed_targets(mgr), [[("decode0", 1)]])

    def test_registration_after_abandon(self):
        mgr = _make_manager()
        mgr.abandon_room(ROOM)
        _register(mgr, ROOM)
        self.assert_dropped(mgr, ROOM)
        self.assertEqual(_failed_targets(mgr), [[("decode0", 1)]])

    def test_registration_completing_after_abandon(self):
        mgr = _make_manager()
        _register(mgr, ROOM, required=2)
        mgr.abandon_room(ROOM)
        # The bootstrap thread still owns the partial entry.
        self.assertIn(ROOM, mgr.transfer_infos)
        _register(mgr, ROOM, required=2)
        self.assert_dropped(mgr, ROOM)
        self.assertEqual(_failed_targets(mgr), [[("decode0", 1), ("decode1", 1)]])

    def test_dummy_peers_not_notified(self):
        mgr = _make_manager()
        mgr.transfer_infos[ROOM] = {0: _info("dummy", is_dummy=True)}
        mgr.update_status(ROOM, KVPoll.WaitingForInput)
        mgr.abandon_room(ROOM)
        self.assert_dropped(mgr, ROOM)
        self.assertEqual(_failed_targets(mgr), [])

    def test_other_rooms_unaffected(self):
        mgr = _make_manager()
        mgr.abandon_room(ROOM)
        _register(mgr, 8)
        self.assertEqual(mgr.request_status[8], KVPoll.WaitingForInput)
        self.assertIn(8, mgr.transfer_infos)

    def test_expired_rooms_pruned_with_leftovers(self):
        mgr = _make_manager()
        mgr.abandon_room(ROOM)
        mgr.abandon_room(8)
        # Re-abandoning moves a room to the back.
        mgr.abandon_room(ROOM)
        self.assertEqual(list(mgr._abandoned_rooms), [8, ROOM])
        _register(mgr, 8, required=2)
        mgr._abandoned_rooms[8] = time.monotonic() - mgr.bootstrap_timeout - 1
        mgr.abandon_room(9)
        self.assertEqual(list(mgr._abandoned_rooms), [ROOM, 9])
        self.assert_dropped(mgr, 8)

    def test_deferred_bootstrap_endpoint_fails(self):
        for registered_first in (True, False):
            with self.subTest(registered_first=registered_first):
                mgr = _make_manager()
                mgr.deferred_bootstrap = DeferredBootstrap(10)
                endpoint = ("decode", 1234)
                if registered_first:
                    mgr.deferred_bootstrap.register(ROOM, endpoint)
                mgr.abandon_room(ROOM)
                if not registered_first:
                    self.assertEqual(
                        mgr.deferred_bootstrap.register(ROOM, endpoint),
                        (endpoint, True),
                    )
                else:
                    self.assertEqual(_failed_targets(mgr), [[endpoint]])

    def test_without_abandon_state_update_status_unchanged(self):
        # Decode managers, and tests that bypass __init__.
        mgr = _make_manager()
        mgr._abandoned_rooms = None
        del mgr._abandoned_rooms_lock
        mgr.update_status(ROOM, KVPoll.WaitingForInput)
        self.assertEqual(mgr.request_status[ROOM], KVPoll.WaitingForInput)


if __name__ == "__main__":
    unittest.main()
