"""CPU tests for typed AFD control messages and startup handshakes."""

import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from unittest.mock import patch
from uuid import uuid4

import msgspec
import zmq

from sglang.srt.afd.afd_handshake import (
    AFD_HANDSHAKE_PORT_DELTA,
    AFDHandshakeInfo,
    AFDHandshakeServer,
    _handshake_reply_decoder,
    _HandshakeError,
    run_handshake_attn,
)
from sglang.srt.afd.afd_sync import (
    AFDDirectCommonPlan,
    AFDDirectControlClient,
    AFDDirectControlServer,
    AFDDirectProposal,
    AFDSyncMsg,
    AFDSyncMsgReceiver,
    AFDSyncMsgSender,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _handshake_info(role, backend="nccl"):
    return AFDHandshakeInfo(
        role=role,
        model_identity="test-model:123",
        num_ranks=1,
        rank_node_ids=[role],
        afd_backend=backend,
        afd_ffn_moe_backend="none",
        wire_dtype="bfloat16",
        moe_layers=[{"layer_id": 0, "hidden_size": 128}],
        direct_backend_version="test-version",
        direct_deepep_fp8_scale_ue8m0=False,
        decode_token_buckets=[1, 2, 4],
        rank_nvshmem_fabric_keys=[(b"\x00\xfffabric", 7)],
    )


def _free_port():
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port <= 65535 - 4:
            return port


def _accept_handshakes(port, info, num_sessions=1):
    with closing(AFDHandshakeServer(port)) as server:
        server._socket.setsockopt(zmq.RCVTIMEO, 5000)
        return server.accept_all(info, num_sessions=num_sessions)


class TestAFDControlWireProtocol(CustomTestCase):
    def _sync_channel(self):
        endpoint = f"inproc://afd-sync-test-{uuid4()}"
        receiver = AFDSyncMsgReceiver(endpoint)
        sender = AFDSyncMsgSender(endpoint)
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        receiver.socket.setsockopt(zmq.RCVTIMEO, 5000)
        return sender, receiver

    def _direct_channel(self):
        endpoint = f"inproc://afd-direct-test-{uuid4()}"
        server = AFDDirectControlServer(endpoint, session_id=0)
        client = AFDDirectControlClient(endpoint, session_id=0)
        self.addCleanup(server.close)
        self.addCleanup(client.close)
        server.socket.setsockopt(zmq.RCVTIMEO, 5000)
        client._socket.setsockopt(zmq.RCVTIMEO, 5000)
        return client, server

    def test_sync_roundtrip_preserves_types_and_step_order(self):
        sender, receiver = self._sync_channel()
        for step_id in range(2):
            msg = AFDSyncMsg(step_id, [3, 4], 1, 8, True, (4, 4))
            sender.send(msg)
            actual = receiver.recv()
            self.assertEqual(actual, msg)
            self.assertIsInstance(actual.tokens_per_rank, list)
            self.assertIsInstance(actual.child_wire_capacities, tuple)
            self.assertEqual(actual.effective_child_wire_capacities, (4, 4))
        sender.send(AFDSyncMsg(3, [0, 0], 4, 0))
        with self.assertRaisesRegex(RuntimeError, "step_id out of order"):
            receiver.recv()

    def test_sync_rejects_invalid_payloads(self):
        sender, receiver = self._sync_channel()
        valid = msgspec.to_builtins(AFDSyncMsg(0, [1], 1, 1))
        invalid_payloads = [
            b"\xc1",
            msgspec.msgpack.encode({**valid, "type": "proposal"}),
            msgspec.msgpack.encode({**valid, "tokens_per_rank": ["1"]}),
            msgspec.msgpack.encode({**valid, "communication_graph": 1}),
            msgspec.msgpack.encode({**valid, "unexpected": True}),
            msgspec.msgpack.encode({"type": "sync"}),
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                sender._socket.send(payload)
                with self.assertRaises(msgspec.DecodeError):
                    receiver.recv()
        sender.send(AFDSyncMsg(0, [1], 1, 1))
        self.assertEqual(receiver.recv().step_id, 0)

    def test_direct_idle_replacement_preserves_work_and_wire_types(self):
        client, server = self._direct_channel()
        idle = AFDDirectProposal(0, (0, 0), int(ForwardMode.IDLE), 0)
        active = AFDDirectProposal(1, (3, 4), int(ForwardMode.DECODE), 8, True, (4, 4))
        client.send_proposal(idle)
        self.assertEqual(server.recv_proposal(), idle)
        client.send_proposal(active)
        self.assertEqual(client.next_proposal_id, 2)

        old_plan = AFDDirectCommonPlan(
            union_step_id=0,
            observed_proposal_ids=(0, 2),
            active_mask=(False, True),
            forward_mode=int(ForwardMode.DECODE),
            parent_wire_capacity=8,
            wire_capacity=8,
            communication_graph=True,
            child_wire_capacities=(4, 4),
        )
        server.send_plan(old_plan)
        self.assertEqual(client.recv_plan(), old_plan)
        self.assertTrue(client.has_outstanding_proposal)

        actual_proposal = server.recv_proposal()
        self.assertEqual(actual_proposal, active)
        self.assertIsInstance(actual_proposal.tokens_per_rank, tuple)
        self.assertIsInstance(actual_proposal.child_wire_capacities, tuple)
        self.assertIs(actual_proposal.can_run_graph, True)
        active_plan = AFDDirectCommonPlan(
            union_step_id=1,
            observed_proposal_ids=(1, 2),
            active_mask=(True, False),
            forward_mode=int(ForwardMode.DECODE),
            parent_wire_capacity=8,
            wire_capacity=8,
            communication_graph=True,
            child_wire_capacities=(4, 4),
        )
        server.send_plan(active_plan)
        actual_plan = client.recv_plan()
        self.assertEqual(actual_plan, active_plan)
        self.assertIsInstance(actual_plan.active_mask, tuple)
        self.assertIs(actual_plan.active_mask[0], True)
        self.assertIs(actual_plan.active_mask[1], False)
        self.assertIsInstance(actual_plan.child_wire_capacities, tuple)
        self.assertIs(actual_plan.communication_graph, True)
        self.assertFalse(client.has_outstanding_proposal)
        self.assertEqual(client.next_proposal_id, 2)

    def test_direct_channels_reject_the_wrong_message_type(self):
        client, server = self._direct_channel()
        client._socket.send(msgspec.msgpack.encode(AFDSyncMsg(0, [1], 1, 1)))
        with self.assertRaises(msgspec.ValidationError):
            server.recv_proposal()
        proposal = AFDDirectProposal(0, (1,), 1, 1)
        client.send_proposal(proposal)
        server.recv_proposal()
        server.socket.send(msgspec.msgpack.encode(proposal))
        with self.assertRaises(msgspec.ValidationError):
            client.recv_plan()
        self.assertTrue(client.has_outstanding_proposal)


class TestAFDHandshakeWireProtocol(CustomTestCase):
    def test_handshake_roundtrip_preserves_topology_and_binary_ids(self):
        for backend in ("nccl", "nvshmem", "direct_deepep", "direct_megamoe"):
            with self.subTest(backend=backend):
                port = _free_port()
                ffn_info = _handshake_info("ffn", backend)
                attn_infos = [_handshake_info("attn", backend) for _ in range(2)]
                for index, info in enumerate(attn_infos):
                    info.rank_node_ids = [f"attn-{index}"]
                uid = b"\x00\xffnvshmem-uid"
                with (
                    envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.override(5),
                    patch(
                        "sglang.srt.afd.afd_handshake._get_nvshmem_unique_id",
                        return_value=uid,
                    ),
                    ThreadPoolExecutor(max_workers=3) as pool,
                ):
                    server = pool.submit(_accept_handshakes, port, ffn_info, 2)
                    clients = [
                        pool.submit(
                            run_handshake_attn,
                            f"127.0.0.1:{port - AFD_HANDSHAKE_PORT_DELTA}",
                            info,
                        )
                        for info in attn_infos
                    ]
                    results = [client.result(timeout=10) for client in clients]
                    server_results = server.result(timeout=10)
                self.assertEqual({result.session_id for result in results}, {0, 1})
                for result in results:
                    self.assertEqual(result.peer, ffn_info)
                    self.assertEqual(
                        result.peer.rank_nvshmem_fabric_keys, [(b"\x00\xfffabric", 7)]
                    )
                    self.assertEqual(result.dist_host, "127.0.0.1")
                    self.assertEqual(result.sync_port, port + 1 + 2 * result.session_id)
                    self.assertEqual(
                        result.nvshmem_uid, uid if backend == "nvshmem" else None
                    )
                    if backend == "nccl":
                        self.assertIsNone(result.attn_members)
                    else:
                        self.assertEqual(
                            result.attn_members,
                            tuple(item.peer for item in server_results),
                        )
                        self.assertEqual(
                            result.attn_members[result.session_id], result.local
                        )
                        self.assertEqual(result.total_attn_ranks, 2)
                    peer_result = server_results[result.session_id]
                    expected_dist_port = port + 2
                    if backend in ("nccl", "nvshmem"):
                        expected_dist_port += 2 * result.session_id
                    self.assertEqual(result.dist_port, expected_dist_port)
                    self.assertEqual(result.sync_port, peer_result.sync_port)
                    self.assertEqual(result.dist_port, peer_result.dist_port)
                    self.assertEqual(result.local, peer_result.peer)

    def test_handshake_rejects_model_mismatch_on_both_peers(self):
        port = _free_port()
        ffn_info = _handshake_info("ffn")
        attn_info = _handshake_info("attn")
        attn_info.model_identity = "other-model"
        with (
            envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.override(5),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            server = pool.submit(_accept_handshakes, port, ffn_info)
            with self.assertRaisesRegex(RuntimeError, "rejected.*model mismatch"):
                run_handshake_attn(
                    f"127.0.0.1:{port - AFD_HANDSHAKE_PORT_DELTA}", attn_info
                )
            with self.assertRaisesRegex(RuntimeError, "model mismatch"):
                server.result(timeout=10)

    def test_handshake_rejects_malformed_request_and_replies_with_error(self):
        invalid_info = asdict(_handshake_info("attn"))
        invalid_info["num_ranks"] = "one"
        port = _free_port()
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            closing(zmq.Context.instance().socket(zmq.REQ)) as client,
        ):
            server = pool.submit(_accept_handshakes, port, _handshake_info("ffn"))
            client.setsockopt(zmq.RCVTIMEO, 5000)
            client.setsockopt(zmq.LINGER, 0)
            client.connect(f"tcp://127.0.0.1:{port}")
            client.send(
                msgspec.msgpack.encode(
                    {"type": "request", "info": invalid_info, "dist_host": "127.0.0.1"}
                )
            )
            reply = _handshake_reply_decoder.decode(client.recv())
            self.assertIsInstance(reply, _HandshakeError)
            self.assertIn("num_ranks", reply.error)
            with self.assertRaises(msgspec.ValidationError):
                server.result(timeout=10)


if __name__ == "__main__":
    unittest.main()
