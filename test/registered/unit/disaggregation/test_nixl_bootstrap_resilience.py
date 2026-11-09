"""CPU contracts for NIXL prefill bootstrap metadata isolation.

The real builders, parsers, receive loop and room-status methods run. Peer
registration and staging application are mocked; no NIXL agent or GPU starts.
"""

import queue
import struct
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import AbortNotification
from sglang.srt.disaggregation.nixl import conn as nixl_conn
from sglang.srt.disaggregation.nixl.conn import (
    GUARD,
    KVArgsRegisterInfo,
    NixlKVManager,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestNixlBootstrapResilience(CustomTestCase):
    def manager(self):
        manager = object.__new__(NixlKVManager)
        manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = False
        manager.deferred_bootstrap = None
        manager.is_mla_backend = False
        manager.transfer_infos = {}
        manager.request_status = {}
        manager.req_to_decode_prefix_len = {}
        manager._deferred_ack_targets = {}
        manager._deferred_ack_poisoned_rooms = set()
        manager._staging_outstanding = {}
        manager._staging_ctx = SimpleNamespace(allocator=None)
        manager._add_remote_peer = Mock()
        manager.record_failure = Mock()
        manager.kv_args = SimpleNamespace(engine_rank=0)
        return manager

    def producer(self, agent="decode-a"):
        manager = self.manager()
        manager.local_ip = "127.0.0.1"
        manager.rank_port = 12345
        manager.agent = SimpleNamespace(
            name=agent, get_agent_metadata=lambda: b"opaque-agent-metadata"
        )
        manager.attn_tp_size = manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.customized_info_enabled = False
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[4096, 8192],
            kv_data_lens=[256, 256],
            kv_item_lens=[16, 16],
            kv_data_mem_kinds=["VRAM", "VRAM"],
            kv_layer_ids=[0, 1],
            aux_data_ptrs=[12288],
            aux_item_lens=[8],
            state_data_ptrs=[],
            state_item_lens=[],
            state_data_lens=[],
            state_dim_per_tensor=[],
            state_layer_ids=[],
            gpu_id=0,
            engine_rank=0,
        )
        return manager

    def transfer(self, *, room=11, agent="decode-a", destinations=1, prefix=0):
        return self.producer(agent).build_receiver_metadata_message(
            bootstrap_room=room,
            required_dst_info_num=destinations,
            kv_indices=np.array([3, 7], dtype=np.int32),
            aux_index=2,
            state_indices=[[5, 9]],
            decode_prefix_len=prefix,
            is_dummy=False,
        )

    def target(self, manager):
        with patch.object(nixl_conn.threading, "Thread") as thread:
            manager._start_bootstrap_thread()
        thread.return_value.start.assert_called_once_with()
        return thread.call_args.kwargs["target"]

    def receive(self, manager, messages):
        # Stop only after the finite message sequence. SystemExit is outside
        # the production Exception guard, matching existing NIXL unit fixtures.
        manager.server_socket = Mock()
        manager.server_socket.recv_multipart.side_effect = [*messages, SystemExit()]
        with self.assertRaises(SystemExit):
            self.target(manager)()

    def assert_transfer(self, manager, *, room=11, agent="decode-a"):
        self.assertEqual(manager.request_status[room], KVPoll.WaitingForInput)
        info = manager.transfer_infos[room][agent]
        self.assertIsInstance(info, TransferInfo)
        np.testing.assert_array_equal(info.dst_kv_indices, [3, 7])
        self.assertEqual(info.dst_aux_index, 2)
        return info

    def test_receiver_continues_after_bad_message_on_real_thread(self):
        manager = self.manager()
        messages = queue.Queue()
        for message in ([b"foreign-traffic"], self.transfer(), SystemExit()):
            messages.put(message)

        def recv():
            value = messages.get(timeout=1)
            if isinstance(value, BaseException):
                raise value
            return value

        manager.server_socket = SimpleNamespace(recv_multipart=recv)
        real_thread = threading.Thread
        threads, errors = [], []

        def start_thread(*, target):
            def run():
                try:
                    target()
                except SystemExit:
                    pass
                except BaseException as exc:
                    errors.append(exc)

            thread = real_thread(target=run)
            threads.append(thread)
            return thread

        with patch.object(nixl_conn.threading, "Thread", side_effect=start_thread):
            manager._start_bootstrap_thread()
        self.assertEqual(len(threads), 1)
        threads[0].join(timeout=3)
        self.assertFalse(threads[0].is_alive(), "owned bootstrap thread did not stop")
        self.assertEqual(errors, [])
        self.assert_transfer(manager)

    def test_malformed_transfer_leaves_no_room_and_valid_followup_progresses(self):
        bad = self.transfer(room=99)
        bad[5] = b"\x01"  # An incomplete int32 destination index.
        manager = self.manager()
        self.receive(manager, [bad, self.transfer()])
        self.assertEqual(set(manager.transfer_infos), {11})
        self.assertNotIn(99, manager.request_status)
        self.assertNotIn(99, manager.req_to_decode_prefix_len)
        manager._add_remote_peer.assert_not_called()
        self.assert_transfer(manager)

    def test_malformed_registration_does_not_register_peer(self):
        valid = self.producer().build_receiver_registration_message()
        bad = valid.copy()
        bad[6] = b"\x01"  # An incomplete uint64 pointer.
        manager = self.manager()
        self.receive(manager, [bad, valid])
        manager._add_remote_peer.assert_called_once()
        info = manager._add_remote_peer.call_args.args[0]
        self.assertIsInstance(info, KVArgsRegisterInfo)
        self.assertEqual(info.dst_kv_ptrs, [4096, 8192])
        self.assertEqual(info.dst_aux_ptrs, [12288])
        self.assertEqual(info.dst_kv_layer_ids, [0, 1])
        self.assertEqual(manager.transfer_infos, {})
        self.assertEqual(manager.request_status, {})

    def test_truncated_and_invalid_encodings_do_not_poison_existing_room(self):
        transfer = self.transfer(room=99)
        registration = self.producer().build_receiver_registration_message()
        cases = [[], [GUARD], [b"foreign"], transfer[:7], registration[:13]]
        for original, index, value in (
            (transfer, 1, b"not-a-room"),
            (transfer, 4, b"\xff"),
            (transfer, 5, b"\x01"),
            (transfer, 8, struct.pack("<I", 1)),
            (transfer, 9, b"not-a-prefix"),
            (transfer, 10, b"not-a-dummy-flag"),
            (registration, 6, b"\x01"),
            (registration, 7, b"\x01"),
            (registration, 9, struct.pack("<I", 1)),
            (registration, 10, b"\xff"),
        ):
            bad = original.copy()
            bad[index] = value
            cases.append(bad)
        for index, bad in enumerate(cases):
            with self.subTest(case=index):
                manager = self.manager()
                self.receive(manager, [self.transfer(), bad, self.transfer(room=12)])
                self.assertEqual(set(manager.transfer_infos), {11, 12})
                self.assert_transfer(manager)
                self.assert_transfer(manager, room=12)
                manager._add_remote_peer.assert_not_called()

    def test_legacy_mandatory_frames_remain_accepted(self):
        manager = self.manager()
        self.receive(
            manager,
            [
                self.producer().build_receiver_registration_message()[:14],
                self.transfer()[:8],
            ],
        )
        manager._add_remote_peer.assert_called_once()
        registration = manager._add_remote_peer.call_args.args[0]
        self.assertEqual(registration.dst_kv_item_lens, [16, 16])
        self.assertEqual(registration.dst_kv_mem_kinds, ["VRAM", "VRAM"])
        self.assertEqual((registration.dst_dcp_size, registration.dst_dcp_rank), (1, 0))
        info = self.assert_transfer(manager)
        self.assertEqual(info.dst_state_indices, [])
        self.assertIsNone(info.decode_prefix_len)
        self.assertFalse(info.is_dummy)

    def test_multiple_destinations_publish_readiness_only_when_complete(self):
        manager = self.manager()
        manager.request_status[11] = KVPoll.Bootstrapping
        self.receive(manager, [self.transfer(destinations=2, prefix=4)])
        self.assertEqual(manager.request_status[11], KVPoll.Bootstrapping)
        self.assertNotIn(11, manager.req_to_decode_prefix_len)
        self.receive(manager, [self.transfer(agent="decode-b", destinations=2, prefix=4)])
        self.assertEqual(set(manager.transfer_infos[11]), {"decode-a", "decode-b"})
        self.assertEqual(manager.req_to_decode_prefix_len[11], 4)
        self.assertEqual(self.assert_transfer(manager).dst_state_indices, [[5, 9]])

    def test_rejected_payload_is_not_logged(self):
        secret = b"private-peer-frame-do-not-log"
        bad = self.transfer()
        bad[6] = secret
        manager = self.manager()
        with self.assertLogs(nixl_conn.logger, level="WARNING") as logs:
            self.receive(manager, [bad, self.transfer()])
        self.assertTrue(logs.records)
        self.assertNotIn(secret.decode(), "\n".join(logs.output))
        self.assert_transfer(manager)

    def test_registration_application_errors_still_propagate(self):
        manager = self.manager()
        error = ValueError("peer registration application failed")
        manager._add_remote_peer.side_effect = error
        with patch.object(nixl_conn.logger, "exception") as fatal:
            with self.assertRaises(ValueError) as caught:
                self.receive(
                    manager,
                    [self.producer().build_receiver_registration_message(), self.transfer()],
                )
        self.assertIs(caught.exception, error)
        fatal.assert_called_once()
        self.assertEqual(manager.server_socket.recv_multipart.call_count, 1)
        self.assertEqual(manager.transfer_infos, {})

    def test_room_application_errors_still_propagate(self):
        manager = self.manager()
        error = ValueError("room state application failed")
        manager.resolve_kv_replica_factor = Mock(side_effect=error)
        with patch.object(nixl_conn.logger, "exception"):
            with self.assertRaises(ValueError) as caught:
                self.receive(manager, [self.transfer(), self.transfer(room=12)])
        self.assertIs(caught.exception, error)
        self.assertEqual(manager.server_socket.recv_multipart.call_count, 1)
        self.assertNotIn(12, manager.transfer_infos)

    def test_unexpected_parser_errors_still_propagate(self):
        manager = self.manager()
        error = RuntimeError("unexpected parser implementation failure")
        with (
            patch.object(TransferInfo, "from_zmq", side_effect=error),
            patch.object(nixl_conn.logger, "exception"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.receive(manager, [self.transfer()])
        self.assertIs(caught.exception, error)
        self.assertEqual(manager.transfer_infos, {})

    def test_socket_errors_still_propagate(self):
        manager = self.manager()
        error = zmq.ZMQError(zmq.ETERM)
        manager.server_socket = Mock()
        manager.server_socket.recv_multipart.side_effect = error
        with patch.object(nixl_conn.logger, "exception"):
            with self.assertRaises(zmq.ZMQError) as caught:
                self.target(manager)()
        self.assertIs(caught.exception, error)

    def test_abort_and_staging_routing_remain_unchanged(self):
        for enabled in (False, True):
            with self.subTest(staging=enabled):
                manager = self.manager()
                manager.enable_staging = enabled
                manager.request_status[11] = KVPoll.Bootstrapping
                watermark = [b"WATERMARK", b"staging-data"]
                response = [b"STAGING_RSP", b"staging-data"]
                with (
                    patch.object(nixl_conn, "handle_watermark_msg") as water,
                    patch.object(nixl_conn, "handle_staging_rsp") as staging,
                ):
                    self.receive(
                        manager, [watermark, response, AbortNotification(room=11).to_zmq()]
                    )
                if enabled:
                    water.assert_called_once_with(manager._staging_ctx, watermark)
                    staging.assert_called_once_with(response, manager.transfer_infos)
                else:
                    water.assert_not_called()
                    staging.assert_not_called()
                self.assertEqual(manager.request_status[11], KVPoll.Failed)
                manager.record_failure.assert_called_once()
                self.assertEqual(manager.transfer_infos, {})

    def test_staging_application_errors_still_propagate(self):
        manager = self.manager()
        manager.enable_staging = True
        error = ValueError("staging application failed")
        with (
            patch.object(nixl_conn, "handle_watermark_msg", side_effect=error),
            patch.object(nixl_conn.logger, "exception"),
        ):
            with self.assertRaises(ValueError) as caught:
                self.receive(manager, [[b"WATERMARK"], self.transfer()])
        self.assertIs(caught.exception, error)
        self.assertEqual(manager.server_socket.recv_multipart.call_count, 1)

    def test_bad_then_valid_metadata_over_real_zmq(self):
        manager = self.manager()
        context = zmq.Context()
        receiver, sender = context.socket(zmq.PULL), context.socket(zmq.PUSH)
        try:
            for socket in (receiver, sender):
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 1000)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
            endpoint = f"inproc://nixl-bootstrap-{uuid.uuid4().hex}"
            receiver.bind(endpoint)
            sender.connect(endpoint)
            sender.send_multipart([b"foreign-traffic"])
            sender.send_multipart(self.transfer())
            received = 0

            def recv():
                nonlocal received
                if received == 2:
                    raise SystemExit
                received += 1
                return receiver.recv_multipart()

            manager.server_socket = SimpleNamespace(recv_multipart=recv)
            with self.assertRaises(SystemExit):
                self.target(manager)()
            self.assertEqual(received, 2)
            self.assert_transfer(manager)
        finally:
            sender.close(linger=0)
            receiver.close(linger=0)
            context.term()


if __name__ == "__main__":
    unittest.main()
