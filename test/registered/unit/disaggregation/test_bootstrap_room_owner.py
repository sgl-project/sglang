"""A PD bootstrap room belongs to the prefill request that claims it; another
request's decode destination, prefill or ABORT on that room must not act on it."""

import contextlib
import threading
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation import prefill as prefill_module
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.srt.disaggregation.common.conn import (
    ABORT_TAG,
    AbortNotification,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl import conn as nixl_conn
from sglang.srt.disaggregation.utils import TransferBackend, kv_transfer_fingerprint
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

ROOM = 1000
FP_A = kv_transfer_fingerprint([11, 12, 13])
FP_B = kv_transfer_fingerprint([21, 22, 23])
A_ENDPOINT = ("10.0.0.1", 7001)
B_ENDPOINT = ("10.0.0.2", 7002)
BACKENDS = (nixl_conn.NixlKVManager, MooncakeKVManager)


def _prefill_manager(manager_type, *, deferred_release=False):
    manager = manager_type.__new__(manager_type)
    manager.is_mla_backend = False
    manager.is_dummy_cp_rank = False
    manager.request_status = {}
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.room_owners = {}
    manager._room_owner_lock = threading.Lock()
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = {}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.enable_deferred_decode_kv_release = deferred_release
    manager.bootstrap_timeout = 300
    manager.send_kv_status_message = MagicMock()
    manager._send_abort_ack = MagicMock()
    manager._send_multipart_locked = MagicMock()
    manager.server_args = SimpleNamespace(enable_pd_role_switch=False)
    manager._stopped = False
    manager._worker_threads = []
    return manager


def _run_bootstrap_loop(manager, *messages):
    """Run the backend's prefill bootstrap thread inline over the messages."""
    manager.server_socket = MagicMock()
    manager.server_socket.recv_multipart.side_effect = [*messages, SystemExit()]
    with patch.object(threading, "Thread") as thread:
        if isinstance(manager, MooncakeKVManager):
            manager.start_prefill_thread()
        else:
            manager._start_bootstrap_thread()
    with contextlib.suppress(SystemExit):
        thread.call_args.kwargs["target"]()


def _sender(manager, fingerprint):
    with get_context().override_server_args(dp_size=1, enable_trace=False):
        return CommonKVSender(manager, "unused", ROOM, prompt_fingerprint=fingerprint)


def _metadata(manager_type, endpoint, prompt_fingerprint):
    """The metadata frames a decode receiver sends to a prefill bootstrap thread."""
    decode = manager_type.__new__(manager_type)
    decode.local_ip, decode.rank_port = endpoint
    decode.agent = SimpleNamespace(name=f"agent-{endpoint[1]}")
    decode.engine = SimpleNamespace(get_session_id=lambda: f"session-{endpoint[1]}")
    return decode.build_receiver_metadata_message(
        bootstrap_room=ROOM,
        required_dst_info_num=1,
        kv_indices=np.array([5, 6], dtype=np.int32),
        aux_index=3,
        state_indices=None,
        decode_prefix_len=None,
        is_dummy=False,
        prompt_fingerprint=prompt_fingerprint,
    )


def _abort(endpoint, prompt_fingerprint, generation=None):
    """The ABORT frames a decode receiver sends to a prefill bootstrap thread."""
    receiver = CommonKVReceiver.__new__(CommonKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        enable_deferred_decode_kv_release=False,
        local_ip=endpoint[0],
        rank_port=endpoint[1],
    )
    receiver.bootstrap_room = ROOM
    receiver.prompt_fingerprint = prompt_fingerprint
    receiver.init_time = None
    receiver._abort_generation = generation
    receiver.bootstrap_infos = [{"rank_ip": "prefill", "rank_port": 1}]
    sent = []
    sock = SimpleNamespace(send_multipart=sent.append)
    receiver._connect_to_bootstrap_server = lambda info: (sock, threading.Lock())
    receiver._send_abort_notification()
    return sent[0]


class TestBootstrapRoomOwner(CustomTestCase):
    def _rejected_targets(self, manager):
        calls = manager.send_kv_status_message.call_args_list
        self.assertTrue(all(c.kwargs["status"] == KVPoll.Failed for c in calls))
        return [tuple(t) for c in calls for t in c.kwargs["targets"]]

    def test_metadata_from_another_request_is_rejected_after_claim(self):
        """A decode sent on another request's room before that request's own
        decode must neither take its KV nor block its decode."""
        for manager_type in BACKENDS:
            with self.subTest(backend=manager_type.__name__):
                manager = _prefill_manager(manager_type)
                _sender(manager, FP_A)
                _run_bootstrap_loop(manager, _metadata(manager_type, B_ENDPOINT, FP_B))
                self.assertNotIn(ROOM, manager.transfer_infos)
                self.assertEqual(manager.request_status[ROOM], KVPoll.Bootstrapping)
                self.assertEqual(self._rejected_targets(manager), [B_ENDPOINT])

                _run_bootstrap_loop(manager, _metadata(manager_type, A_ENDPOINT, FP_A))
                (info,) = manager.transfer_infos[ROOM].values()
                self.assertEqual((info.endpoint, info.dst_port), A_ENDPOINT)
                self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)

    def test_claim_drops_metadata_that_arrived_before_the_owner(self):
        """Metadata from another request that beat the prefill request goes
        back to waiting, and the owner's own metadata still bootstraps it."""
        for manager_type in BACKENDS:
            with self.subTest(backend=manager_type.__name__):
                manager = _prefill_manager(manager_type)
                _run_bootstrap_loop(manager, _metadata(manager_type, B_ENDPOINT, FP_B))
                self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)

                _sender(manager, FP_A)
                self.assertNotIn(ROOM, manager.transfer_infos)
                self.assertNotIn(ROOM, manager.req_to_decode_prefix_len)
                self.assertEqual(manager.request_status[ROOM], KVPoll.Bootstrapping)
                self.assertEqual(self._rejected_targets(manager), [B_ENDPOINT])

                _run_bootstrap_loop(manager, _metadata(manager_type, A_ENDPOINT, FP_A))
                self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)

    def test_second_prefill_on_a_held_room_fails_alone(self):
        """A prefill request on a room that another live request holds fails
        and leaves the holder's destination and status untouched."""
        for manager_type in BACKENDS:
            with self.subTest(backend=manager_type.__name__):
                manager = _prefill_manager(manager_type)
                owner = _sender(manager, FP_A)
                _run_bootstrap_loop(manager, _metadata(manager_type, A_ENDPOINT, FP_A))

                other = _sender(manager, FP_B)
                self.assertEqual(other.poll(), KVPoll.Failed)
                with self.assertRaisesRegex(KVTransferError, "in use by another"):
                    other.failure_exception()
                other.abort()
                other.clear()
                self.assertIn(ROOM, manager.transfer_infos)
                self.assertEqual(owner.poll(), KVPoll.WaitingForInput)

                owner.clear()
                self.assertEqual(manager.room_owners, {})

    def test_room_stays_held_until_every_sender_is_cleared(self):
        """The samples of a parallel-sampling request reuse one room, and an
        aborted sender keeps it until clear() drops the room's state; another
        request cannot take the room before then."""
        manager_type = nixl_conn.NixlKVManager
        manager = _prefill_manager(manager_type)
        first, second = _sender(manager, FP_A), _sender(manager, FP_A)
        first.clear()
        second.abort()
        self.assertEqual(_sender(manager, FP_B).poll(), KVPoll.Failed)
        _run_bootstrap_loop(manager, _metadata(manager_type, B_ENDPOINT, FP_B))
        self.assertEqual(self._rejected_targets(manager), [B_ENDPOINT])

        second.clear()
        self.assertEqual(_sender(manager, FP_B).poll(), KVPoll.Bootstrapping)

    def test_unverifiable_metadata_keeps_the_old_behavior(self):
        """An older decode sends no fingerprint, and a released room has no
        owner: both are filed as before."""
        for manager_type in BACKENDS:
            with self.subTest(backend=manager_type.__name__):
                manager = _prefill_manager(manager_type)
                owner = _sender(manager, FP_A)
                legacy = _metadata(manager_type, B_ENDPOINT, None)[:-1]
                _run_bootstrap_loop(manager, legacy)
                self.assertIn(ROOM, manager.transfer_infos)

                owner.clear()
                _run_bootstrap_loop(manager, _metadata(manager_type, B_ENDPOINT, FP_B))
                self.assertIn(ROOM, manager.transfer_infos)
                manager.send_kv_status_message.assert_not_called()

    def test_abort_from_another_request_leaves_the_room_running(self):
        """The decode of a rejected destination sends an ABORT for the room;
        it must not fail the owner, and its ACK must still go out."""
        cases = ((FP_B, False), (FP_A, True), (None, True))
        for manager_type in BACKENDS:
            for deferred_release in (False, True):
                for sender_fp, aborted in cases:
                    with self.subTest(
                        backend=manager_type.__name__,
                        deferred_release=deferred_release,
                        sender=sender_fp,
                    ):
                        manager = _prefill_manager(
                            manager_type, deferred_release=deferred_release
                        )
                        _sender(manager, FP_A)
                        _run_bootstrap_loop(
                            manager, _abort(B_ENDPOINT, sender_fp, generation=7)
                        )
                        self.assertEqual(
                            manager.request_status[ROOM],
                            KVPoll.Failed if aborted else KVPoll.Bootstrapping,
                        )
                        if deferred_release:
                            manager._send_abort_ack.assert_called_once()

    def test_deferred_bootstrap_abort_from_another_request_is_ignored(self):
        """Under prefill_complete the control-message path applies the same
        rule before it cancels the room."""
        for sender_fp, aborted in ((FP_B, False), (FP_A, True)):
            with self.subTest(sender=sender_fp):
                manager = _prefill_manager(MooncakeKVManager)
                manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
                _sender(manager, FP_A)
                _run_bootstrap_loop(manager, _abort(B_ENDPOINT, sender_fp))
                self.assertEqual(manager.deferred_bootstrap.rooms[ROOM].failed, aborted)
                self.assertEqual(
                    manager.request_status[ROOM],
                    KVPoll.Failed if aborted else KVPoll.Bootstrapping,
                )

    def test_decode_and_prefill_fingerprint_the_same_ids(self):
        """Decode's receiver and the prefill request for the same transfer
        agree, also for a rebootstrap, whose prefill prompt carries outputs."""
        sampling_params = SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            ignore_eos=False,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        for output_ids in ([], [4, 5, 6]):
            with self.subTest(output_ids=output_ids):
                decode_req = SimpleNamespace(
                    origin_input_ids=array("q", [1, 2, 3]),
                    output_ids=output_ids,
                    sampling_params=sampling_params,
                    rid="rid",
                    bootstrap_host="10.0.0.9",
                    bootstrap_port=8998,
                    bootstrap_room=ROOM,
                    priority=None,
                    extra_key=None,
                    cache_salt=None,
                    routing_key=None,
                    disagg_prefill_dp_rank=None,
                )
                prefill_ids = (
                    Req.build_rebootstrap_payload(decode_req)["input_ids"]
                    if output_ids
                    else decode_req.origin_input_ids
                )
                prefill_req = SimpleNamespace(
                    origin_input_ids=prefill_ids,
                    bootstrap_host="10.0.0.9",
                    bootstrap_room=ROOM,
                    disagg_prefill_dp_rank=None,
                )
                kv_class = MagicMock()
                decode_queue = decode_module.DecodePreallocQueue.__new__(
                    decode_module.DecodePreallocQueue
                )
                decode_queue.transfer_backend = TransferBackend.NIXL
                decode_queue.kv_manager = decode_queue.queue = None
                prefill_queue = prefill_module.PrefillBootstrapQueue.__new__(
                    prefill_module.PrefillBootstrapQueue
                )
                prefill_queue.transfer_backend = TransferBackend.NIXL
                prefill_queue.kv_manager = None
                prefill_queue.bootstrap_port = 8998
                with (
                    patch.object(decode_module, "get_kv_class", return_value=kv_class),
                    patch.object(decode_module, "move"),
                    patch.object(prefill_module, "get_kv_class", return_value=kv_class),
                ):
                    decode_queue._create_receiver_and_enqueue(
                        decode_req, is_rebootstrap=bool(output_ids)
                    )
                    prefill_queue.create_sender(prefill_req, num_kv_heads=1)
                decode_fp, prefill_fp = (
                    c.kwargs["prompt_fingerprint"] for c in kv_class.call_args_list
                )
                self.assertEqual(decode_fp, prefill_fp)

    def test_abort_notification_round_trip(self):
        for generation in (None, 7):
            for fingerprint in (None, FP_B):
                with self.subTest(generation=generation, fingerprint=fingerprint):
                    sent = AbortNotification(
                        room=ROOM,
                        decode_ip=B_ENDPOINT[0],
                        decode_port=B_ENDPOINT[1],
                        generation=generation,
                        prompt_fingerprint=fingerprint,
                    )
                    self.assertEqual(AbortNotification.from_zmq(sent.to_zmq()), sent)
        # The frames an older decode sends.
        legacy = [ABORT_TAG, b"1000", b"10.0.0.2", b"7002", b"7"]
        self.assertIsNone(AbortNotification.from_zmq(legacy).prompt_fingerprint)


if __name__ == "__main__":
    unittest.main()
