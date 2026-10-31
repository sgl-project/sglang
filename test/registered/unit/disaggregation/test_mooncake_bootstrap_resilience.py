"""Malformed metadata must not strand subsequent, valid PD handshakes."""

import queue
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOM = 42


def _metadata(room=ROOM, session=b"decode-session", destinations=1):
    return [
        str(room).encode("ascii"),
        b"127.0.0.1",
        b"6000",
        session,
        np.array([1, 2], dtype=np.int32).tobytes(),
        b"0",
        pack_int_lists([[3]], "i"),
        str(destinations).encode("ascii"),
        b"0",
        b"",
    ]


def _manager():
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager._stopped = False
    manager._worker_threads = []
    manager.server_socket = None
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.request_status = {ROOM: KVPoll.Bootstrapping}
    manager.update_status = lambda room, status: manager.request_status.__setitem__(
        room, status
    )
    manager.resolve_kv_replica_factor = Mock()
    manager._handle_abort_notification = Mock(return_value=False)
    manager.requires_dcp_relayout = Mock(return_value=False)
    manager._publish_peer_registration = Mock(return_value=None)
    manager._staging_ctx = SimpleNamespace(
        watermark_cv=threading.Condition(), remote_watermarks={}
    )
    return manager


class TestMooncakeBootstrapResilience(unittest.TestCase):
    def test_malformed_message_does_not_publish_partial_room_state(self):
        malformed = [
            [],
            [b"\xd7"],
            [b"0\x9a"],
            [b"42"],
            [b"None"],
            [b"WATERMARK"],
            [b"STAGING_RSP"],
            [b"not-a-room", *_metadata()[1:]],
            _metadata(destinations=0),
            _metadata(room=-1),
        ]
        for index, value in [(4, b"x"), (5, b"invalid"), (6, b"x"), (9, b"x")]:
            message = _metadata()
            message[index] = value
            malformed.append(message)
        for message in malformed:
            with self.subTest(message_index=malformed.index(message)):
                manager = _manager()
                with self.assertLogs(
                    "sglang.srt.disaggregation.mooncake.conn", level="WARNING"
                ):
                    manager._handle_prefill_bootstrap_message(message)
                self.assertEqual(manager.transfer_infos, {})
                self.assertEqual(manager.req_to_decode_prefix_len, {})
                self.assertEqual(manager.request_status[ROOM], KVPoll.Bootstrapping)
                manager._publish_peer_registration.assert_not_called()
                manager._handle_prefill_bootstrap_message(_metadata())
                self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)

    def test_receiver_continues_after_bad_message(self):
        manager = _manager()
        messages = queue.Queue()
        manager._make_worker_recv = lambda socket: messages.get
        received = threading.Event()

        def update(room, status):
            manager.request_status[room] = status
            received.set()

        manager.update_status = update
        manager.start_prefill_thread()
        receiver = manager._worker_threads[-1]
        try:
            messages.put([b"\xd7"])
            messages.put(_metadata())
            self.assertTrue(received.wait(timeout=2))
            self.assertTrue(receiver.is_alive())
            self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)
        finally:
            manager._stopped = True
            messages.put(None)
            receiver.join(timeout=2)
        self.assertFalse(receiver.is_alive())

    def test_waits_for_all_decode_destinations(self):
        manager = _manager()
        manager._handle_prefill_bootstrap_message(_metadata(destinations=2))
        self.assertEqual(manager.request_status[ROOM], KVPoll.Bootstrapping)
        manager._handle_prefill_bootstrap_message(
            _metadata(session=b"another-session", destinations=2)
        )
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)
        self.assertEqual(len(manager.transfer_infos[ROOM]), 2)

    def test_legacy_metadata_without_optional_fields(self):
        manager = _manager()
        manager._handle_prefill_bootstrap_message(_metadata()[:8])
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)
        self.assertEqual(manager.req_to_decode_prefix_len[ROOM], 0)

    def test_registration_and_watermark_still_work(self):
        manager = _manager()
        manager._handle_prefill_bootstrap_message(
            [
                b"None",
                b"127.0.0.1",
                b"6000",
                b"decode-session",
                struct.pack("Q", 0x1000),
                struct.pack("Q", 0x2000),
                pack_int_lists([], "Q"),
                b"0",
                b"1",
                b"16",
            ]
        )
        manager._publish_peer_registration.assert_called_once()
        manager._handle_prefill_bootstrap_message(
            [b"WATERMARK", b"1", b"10", b"decode-session"]
        )
        self.assertEqual(
            manager._staging_ctx.remote_watermarks["decode-session"], (1, 10)
        )

    def test_internal_error_is_not_silently_treated_as_bad_metadata(self):
        manager = _manager()
        manager.resolve_kv_replica_factor.side_effect = ValueError("internal failure")
        with self.assertRaisesRegex(ValueError, "internal failure"):
            manager._handle_prefill_bootstrap_message(_metadata())

    def test_staging_control_is_dropped_when_staging_disabled(self):
        manager = _manager()
        manager._staging_ctx = None
        for message in (
            [b"WATERMARK", b"1", b"10", b"decode-session"],
            [b"STAGING_RSP", b"42", b"0", b"0", b"1", b"10", b"decode-session"],
        ):
            with self.assertLogs(
                "sglang.srt.disaggregation.mooncake.conn", level="WARNING"
            ):
                manager._handle_prefill_bootstrap_message(message)
        manager._handle_prefill_bootstrap_message(_metadata())
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)

    def test_dead_receiver_is_fatal_except_during_teardown(self):
        manager = _manager()
        manager._bootstrap_thread = Mock(is_alive=Mock(return_value=False))
        with self.assertRaisesRegex(RuntimeError, "bootstrap receiver exited"):
            manager.check_worker_health()
        manager._stopped = True
        manager.check_worker_health()

    def test_transport_error_is_not_swallowed(self):
        manager = _manager()
        manager._make_worker_recv = lambda socket: Mock(
            side_effect=OSError("socket failed")
        )
        errors = []
        with patch(
            "threading.excepthook",
            side_effect=lambda args: errors.append(args.exc_value),
        ):
            manager.start_prefill_thread()
            manager._bootstrap_thread.join(timeout=2)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], OSError)
        with self.assertRaises(RuntimeError):
            manager.check_worker_health()


if __name__ == "__main__":
    unittest.main()
