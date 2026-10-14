"""Unit tests for the mooncake early-registration room lifecycle.

PD routers can dispatch the decode leg of a request concurrently with (or
even before) the prefill /generate reaches its scheduler: a room
registration can therefore arrive at the prefill bootstrap_thread while the
room's sender does not exist yet (burst tokenization, a multi-second running
batch delaying ingest). decode registers exactly once and never re-sends.

These tests pin the lifecycle invariants:

- an early registration is accepted and the room may complete bootstrap
  before the sender exists (the sender's later Bootstrapping status must
  not demote the already-recorded WaitingForInput);
- a registration for an already-concluded room (Failed, tombstoned, or
  table-absent with a recorded failure) is dropped instead of recreating
  its transfer targets -- dead-end transfers for a request that no longer
  exists;
- rooms whose early registration is never claimed by a prefill request are
  reclaimed by the sweep after early_room_ttl_s, and claimed/fresh rooms
  are left alone.
"""

import threading
import time
import types
import unittest
from typing import Callable

import numpy as np
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


class _LifecycleTestManager(MooncakeKVManager):
    """A MooncakeKVManager shell built without the engine/registration stack.

    Everything the bootstrap_thread and the lifecycle helpers touch is plain
    dict bookkeeping, so the shell is sufficient for control-flow coverage.
    Real ZMQ IO happens only against a private PULL socket created by
    with_bind_socket().
    """

    def __init__(self):
        # Intentionally skip MooncakeKVManager.__init__: no transfer engine,
        # no real sockets, no worker pool.
        self.request_status = {}
        self.transfer_infos = {}
        self.req_to_decode_prefix_len = {}
        self.failure_records = {}
        self.failure_lock = threading.Lock()
        self.is_mla_backend = False
        # update_status() drops deferred-ACK state when a room lifecycle
        # opens; the shell mirrors the real manager's tables.
        self._deferred_ack_targets = {}
        self._deferred_ack_poisoned_rooms = set()
        self._staging_outstanding = {}
        # Early-registration lifecycle bookkeeping mirroring the real
        # MooncakeKVManager PREFILL branch.
        self._room_lifecycle_lock = threading.Lock()
        self._concluded_rooms = {}
        self._early_rooms = {}
        self.tombstone_ttl_s = 360.0
        self.early_room_ttl_s = 150.0
        self._stopped = False
        self._worker_threads = []
        # _make_worker_recv() reads server_args.enable_pd_role_switch; the
        # shell keeps the plain blocking recv.
        self.server_args = types.SimpleNamespace(enable_pd_role_switch=False)

    def with_bind_socket(self) -> None:
        self._zmq_ctx = zmq.Context()
        self.server_socket = self._zmq_ctx.socket(zmq.PULL)
        self.port = self.server_socket.bind_to_random_port("tcp://127.0.0.1")
        # One persistent PUSH connection: ZMQ FIFO ordering is per-socket,
        # and several test flows depend on send order.
        self._push_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._push_socket.setsockopt(zmq.LINGER, 0)
        self._push_socket.connect(f"tcp://127.0.0.1:{self.port}")

    def push_message(self, frames) -> None:
        self._push_socket.send_multipart(frames)

    def shutdown(self) -> None:
        # Ask the consumer loop to exit, then wake any thread parked in a
        # blocking recv with a benign message so it finishes its current
        # iteration and observes _stopped (a room absent from every table
        # with no failure record is early-accepted, which is inert here).
        self._stopped = True
        housekeeping_shutdown = getattr(self, "_housekeeping_shutdown", None)
        if housekeeping_shutdown is not None:
            housekeeping_shutdown.set()
        if getattr(self, "_push_socket", None) is not None:
            try:
                self._push_socket.send_multipart(self._registration_frames(0))
            except Exception:
                pass
        for t in getattr(self, "_worker_threads", []):
            t.join(timeout=5.0)
        if getattr(self, "_push_socket", None) is not None:
            self._push_socket.close(linger=0)
        if self.server_socket is not None:
            self.server_socket.close(linger=0)
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except zmq.ZMQError:
                pass

    def _registration_frames(
        self,
        room: int,
        session: bytes = b"sess-1",
        required_dst_info_num: int = 1,
    ):
        return [
            str(room).encode("ascii"),
            b"127.0.0.1",
            b"12349",
            session,
            np.array([0, 1, 2], dtype=np.int32).tobytes(),
            b"5",
            pack_int_lists([], "i"),
            str(required_dst_info_num).encode("ascii"),
            b"0",  # decode_prefix_len
        ]


class TestEarlyRegistrationLifecycle(CustomTestCase):
    def setUp(self):
        self.mgr = _LifecycleTestManager()
        self.mgr.with_bind_socket()
        self.mgr.start_prefill_thread()

    def tearDown(self):
        self.mgr.shutdown()

    def _simulate_sender_creation(self, room: int):
        # One-line equivalent of MooncakeKVSender.__init__: claim the room,
        # then let the base class insert its opening Bootstrapping status.
        self.mgr.on_sender_created(room)
        self.mgr.update_status(room, KVPoll.Bootstrapping)

    def test_early_registration_accepted_before_sender(self):
        # Decode registered the room while the prefill leg is still behind
        # tokenization / a running batch (no request_status entry yet and no
        # recorded failure). The registration must be accepted.
        self.mgr.push_message(self.mgr._registration_frames(300))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(300) == KVPoll.WaitingForInput
            ),
            "early registration was not accepted",
        )
        self.assertIn(300, self.mgr.transfer_infos)

    def test_early_registration_survives_late_sender_creation(self):
        self.mgr.push_message(self.mgr._registration_frames(301))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(301) == KVPoll.WaitingForInput
            ),
            "early registration was not accepted",
        )

        self._simulate_sender_creation(301)

        self.assertEqual(
            self.mgr.request_status.get(301),
            KVPoll.WaitingForInput,
            "sender's Bootstrapping status demoted the early registration",
        )
        self.assertNotIn(301, self.mgr._early_rooms)

    def test_partial_early_registration_completes_after_sender_claim(self):
        # With required_dst_info_num=2 the first decode rank's registration
        # parks the room as pending; the sender claims the room; the second
        # decode rank's registration completes the bootstrap.
        self.mgr.push_message(
            self.mgr._registration_frames(
                302, session=b"sess-a", required_dst_info_num=2
            )
        )
        self.assertTrue(
            _wait_for(lambda: 302 in self.mgr._early_rooms),
            "first early registration was not tracked",
        )
        self.assertIsNone(self.mgr.request_status.get(302))
        self.assertEqual(len(self.mgr.transfer_infos.get(302, {})), 1)

        self._simulate_sender_creation(302)

        self.mgr.push_message(
            self.mgr._registration_frames(
                302, session=b"sess-b", required_dst_info_num=2
            )
        )
        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(302) == KVPoll.WaitingForInput
            ),
            "room did not bootstrap after its registrations completed",
        )
        self.assertEqual(len(self.mgr.transfer_infos[302]), 2)

    def test_failed_room_registration_dropped(self):
        # Failed is terminal: a registration for it must not resurrect or
        # re-arm any state. A live room's registration afterwards is the
        # consumption barrier (per-socket FIFO).
        self.mgr.request_status[303] = KVPoll.Failed
        self.mgr.request_status[305] = KVPoll.Bootstrapping
        self.mgr.push_message(self.mgr._registration_frames(303))
        self.mgr.push_message(self.mgr._registration_frames(305))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(305) == KVPoll.WaitingForInput
            ),
            "barrier room did not bootstrap",
        )
        self.assertEqual(self.mgr.request_status.get(303), KVPoll.Failed)
        self.assertNotIn(303, self.mgr.transfer_infos)

    def test_concluded_room_registration_dropped(self):
        # Room concluded and its sender cleared the tables -- modeled by a
        # tombstone. The late registration must not recreate the transfer
        # targets. A live room's registration afterwards is the barrier.
        self.mgr.note_room_concluded(123)
        self.mgr.request_status[124] = KVPoll.Bootstrapping
        self.mgr.push_message(self.mgr._registration_frames(123))
        self.mgr.push_message(self.mgr._registration_frames(124))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(124) == KVPoll.WaitingForInput
            ),
            "barrier room did not bootstrap",
        )
        self.assertNotIn(123, self.mgr.request_status)
        self.assertNotIn(123, self.mgr.transfer_infos)

    def test_failed_then_cleared_room_registration_dropped(self):
        # The sender concluded and popped the tables before the tombstone
        # was written (e.g. a sibling-rank bootstrap timeout concluded the
        # room first): the surviving recorded failure is enough evidence
        # that the room is concluded, never early.
        with self.mgr.failure_lock:
            self.mgr.failure_records[306] = "Request 306 timed out ..."
        self.mgr.request_status[307] = KVPoll.Bootstrapping

        self.mgr.push_message(self.mgr._registration_frames(306))
        self.mgr.push_message(self.mgr._registration_frames(307))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(307) == KVPoll.WaitingForInput
            ),
            "barrier room did not bootstrap",
        )
        self.assertNotIn(306, self.mgr.request_status)
        self.assertNotIn(306, self.mgr.transfer_infos)
        self.assertNotIn(306, self.mgr._early_rooms)

    def test_live_registration_still_bootstraps(self):
        self.mgr.request_status[789] = KVPoll.Bootstrapping
        self.mgr.push_message(self.mgr._registration_frames(789))

        self.assertTrue(
            _wait_for(
                lambda: self.mgr.request_status.get(789) == KVPoll.WaitingForInput
            ),
            "healthy room did not leave KVPoll.Bootstrapping",
        )
        self.assertIn(789, self.mgr.transfer_infos)


class TestSweepStaleRooms(CustomTestCase):
    def setUp(self):
        self.mgr = _LifecycleTestManager()

    def test_sweep_reclaims_unclaimed_rooms_and_expired_tombstones(self):
        self.mgr._early_rooms[306] = time.monotonic()
        self.mgr.early_room_ttl_s = 0.0  # expire immediately
        self.mgr.note_room_concluded(307)
        self.mgr.tombstone_ttl_s = 0.0

        self.mgr.sweep_stale_rooms()

        self.assertNotIn(306, self.mgr._early_rooms)
        self.assertNotIn(307, self.mgr._concluded_rooms)
        with self.mgr.failure_lock:
            self.assertIn(306, self.mgr.failure_records)

    def test_sweep_preserves_fresh_and_claimed_rooms(self):
        self.mgr._early_rooms[308] = time.monotonic()
        self.mgr.sweep_stale_rooms()
        self.assertIn(308, self.mgr._early_rooms)

        self.mgr.early_room_ttl_s = 0.0
        self.mgr.on_sender_created(308)
        self.mgr.sweep_stale_rooms()
        self.assertNotIn(308, self.mgr._early_rooms)


if __name__ == "__main__":
    unittest.main()
