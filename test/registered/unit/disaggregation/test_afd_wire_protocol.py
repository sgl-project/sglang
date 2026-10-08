"""CPU tests for typed AFD control messages and startup handshakes."""

import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from itertools import product
from unittest.mock import patch
from uuid import uuid4

import msgspec
import zmq

from sglang.srt.afd.afd_handshake import (
    AFD_HANDSHAKE_PORT_DELTA,
    AFDHandshakeInfo,
    AFDHandshakeResult,
    AFDHandshakeServer,
    _handshake_reply_decoder,
    _HandshakeError,
    build_afd_graph_capture_plan,
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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _handshake_info(role):
    return AFDHandshakeInfo(
        role=role,
        model_identity="test-model:123",
        num_ranks=1,
        rank_node_ids=["node-0"],
        afd_backend="nccl",
        afd_ffn_moe_backend="none",
        wire_dtype="bfloat16",
        moe_layers=[{"layer_id": 0, "hidden_size": 128, "top_k": 2}],
    )


def _free_port():
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port <= 65535 - 4:
            return port


def _accept_handshake(port, info, num_sessions=1):
    with closing(AFDHandshakeServer(port)) as server:
        server._socket.setsockopt(zmq.RCVTIMEO, 5000)
        return server.accept_all(info, num_sessions)


class TestAFDControlWireProtocol(unittest.TestCase):
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
            msg = AFDSyncMsg(
                step_id,
                [3, 4],
                1,
                8,
                communication_graph=step_id == 0,
                child_wire_capacities=(4, 4),
            )
            sender.send(msg)
            actual = receiver.recv()
            self.assertEqual(actual, msg)
            self.assertIsInstance(actual.tokens_per_rank, list)
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
            msgspec.msgpack.encode({**valid, "wire_capacity": "1"}),
            msgspec.msgpack.encode({**valid, "communication_graph": "true"}),
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


class TestAFDHandshakeWireProtocol(unittest.TestCase):
    def test_incompatible_session_rejects_all_collected_peers(self):
        port = _free_port()
        infos = [_handshake_info("attn") for _ in range(2)]
        infos[1].num_ranks = 2
        with (
            envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.override(5),
            ThreadPoolExecutor(max_workers=3) as pool,
        ):
            server = pool.submit(_accept_handshake, port, _handshake_info("ffn"), 2)
            clients = [
                pool.submit(
                    run_handshake_attn,
                    f"127.0.0.1:{port - AFD_HANDSHAKE_PORT_DELTA}",
                    info,
                )
                for info in infos
            ]
            for future in [server, *clients]:
                with self.assertRaisesRegex(
                    RuntimeError, "homogeneous attn instances.*num_ranks"
                ):
                    future.result(timeout=10)

    def test_microbatch_graph_plan_uses_child_fanin_and_keeps_tiny_batches_serial(self):
        for count in (2, 3):
            with self.subTest(count=count):
                attn, ffn = _handshake_info("attn"), _handshake_info("ffn")
                attn.num_ranks, ffn.num_ranks = 4, 2
                attn.afd_ffn_moe_backend = ffn.afd_ffn_moe_backend = "deepep"
                attn.decode_num_microbatches = count
                attn.decode_token_buckets = [1, 32, 64]
                attn.decode_graph_token_buckets = [1, 64]
                ffn.decode_graph_token_buckets = [2, 64]
                plan = build_afd_graph_capture_plan(
                    AFDHandshakeResult(attn, ffn, "localhost", 1, 2)
                )
                parent, tiny = plan.attn_graph_variants
                self.assertEqual(parent.child_wire_capacities, (32,) * count)
                self.assertEqual(parent.child_compute_capacities, (64,) * count)
                self.assertEqual(tiny.child_wire_capacities, (1,))
                self.assertEqual(tiny.child_compute_capacities, (2,))
                self.assertEqual(plan.ffn_graph_variants, ())

    def test_asymmetric_graph_plan_agrees_on_wire_and_compute_capacities(self):
        for layout, num_attn, num_ffn in (
            ("none", 3, 2),
            ("none", 2, 3),
            ("deepep", 3, 2),
            ("deepep", 2, 3),
        ):
            with self.subTest(layout=layout, num_attn=num_attn, num_ffn=num_ffn):
                attn, ffn = _handshake_info("attn"), _handshake_info("ffn")
                attn.num_ranks, ffn.num_ranks = num_attn, num_ffn
                attn.afd_ffn_moe_backend = ffn.afd_ffn_moe_backend = layout
                multiplier = 1 if layout == "deepep" else num_attn
                fan_in = (
                    (num_attn + num_ffn - 1) // num_ffn if layout == "deepep" else 1
                )
                attn.decode_token_buckets = [4 * multiplier, 8 * multiplier]
                attn.decode_graph_token_buckets = attn.decode_token_buckets
                ffn.decode_graph_token_buckets = [
                    capacity * fan_in for capacity in attn.decode_token_buckets
                ]
                handshake = AFDHandshakeResult(ffn, attn, "localhost", 1, 2)
                plan = build_afd_graph_capture_plan(handshake)
                self.assertEqual(
                    [
                        (
                            v.attn_batch_size,
                            v.attn_wire_capacity,
                            v.child_compute_capacities,
                        )
                        for v in plan.attn_graph_variants
                    ],
                    [
                        (8, 8 * multiplier, (8 * multiplier * fan_in,)),
                        (4, 4 * multiplier, (4 * multiplier * fan_in,)),
                    ],
                )
                self.assertEqual(
                    plan.ffn_graph_variants,
                    (
                        (8 * multiplier, 8 * multiplier * fan_in),
                        (4 * multiplier, 4 * multiplier * fan_in),
                    ),
                )
                ffn.disable_cuda_graph_padding = True
                if layout == "deepep":
                    with self.assertRaisesRegex(RuntimeError, "non-uniform SCATTERED"):
                        build_afd_graph_capture_plan(handshake)
                ffn.disable_cuda_graph_padding = False
                ffn.decode_graph_token_buckets = []
                with self.assertRaisesRegex(RuntimeError, "every attn graph capacity"):
                    build_afd_graph_capture_plan(handshake)

    def test_handshake_roundtrip_agrees_on_endpoints_and_model(self):
        for transport, (num_attn, num_ffn, backend, num_sessions) in product(
            ("nccl", "nvshmem", "direct_deepep", "direct_megamoe"),
            (
                (1, 1, "none", 1),
                (3, 2, "none", 1),
                (2, 3, "deepep", 1),
                (1, 1, "none", 2),
                (3, 2, "none", 2),
                (2, 3, "deepep", 2),
            ),
        ):
            is_direct = transport in ("direct_deepep", "direct_megamoe")
            if is_direct and backend != "deepep":
                continue
            with self.subTest(
                transport=transport,
                num_attn=num_attn,
                num_ffn=num_ffn,
                backend=backend,
                num_sessions=num_sessions,
            ):
                port = _free_port()
                ffn_info = _handshake_info("ffn")
                attn_info = _handshake_info("attn")
                ffn_info.afd_backend = attn_info.afd_backend = transport
                if is_direct:
                    ffn_info.direct_backend_version = (
                        attn_info.direct_backend_version
                    ) = "test-version"
                    ffn_info.direct_deepep_fp8_scale_ue8m0 = False
                ffn_info.rank_node_ids = ["node-0"] * num_ffn
                attn_info.rank_node_ids = ["node-0"] * num_attn
                uid = b"\x00\xffnvshmem-uid"
                ffn_info.num_ranks = num_ffn
                attn_info.num_ranks = num_attn
                ffn_info.afd_ffn_moe_backend = backend
                attn_info.afd_ffn_moe_backend = backend
                graph_capacity = 32 if backend == "deepep" else 32 * num_attn
                attn_info.decode_token_buckets = [graph_capacity, 96 * num_attn]
                attn_info.decode_graph_token_buckets = [graph_capacity]
                ffn_info.decode_graph_token_buckets = [1]
                with (
                    envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.override(5),
                    patch(
                        "sglang.srt.afd.afd_handshake._get_nvshmem_unique_id",
                        return_value=uid,
                    ),
                    ThreadPoolExecutor(max_workers=num_sessions + 1) as pool,
                ):
                    server = pool.submit(
                        _accept_handshake, port, ffn_info, num_sessions
                    )
                    clients = [
                        pool.submit(
                            run_handshake_attn,
                            f"127.0.0.1:{port - AFD_HANDSHAKE_PORT_DELTA}",
                            attn_info,
                        )
                        for _ in range(num_sessions)
                    ]
                    peer_results = server.result(timeout=10)
                    results = sorted(
                        (client.result(timeout=10) for client in clients),
                        key=lambda result: result.session_id,
                    )
                self.assertEqual(len(peer_results), num_sessions)
                for session_id, (result, peer_result) in enumerate(
                    zip(results, peer_results)
                ):
                    self.assertEqual(result.session_id, session_id)
                    self.assertEqual(peer_result.session_id, session_id)
                    self.assertEqual(
                        result.nvshmem_uid, uid if transport == "nvshmem" else None
                    )
                    self.assertEqual(
                        result.attn_members,
                        tuple(peer.peer for peer in peer_results)
                        if transport != "nccl"
                        else None,
                    )
                    self.assertEqual(result.local, attn_info)
                    self.assertEqual(result.peer, ffn_info)
                    self.assertEqual(peer_result.local, ffn_info)
                    self.assertEqual(peer_result.peer, attn_info)
                    self.assertEqual(result.dist_host, "127.0.0.1")
                    self.assertEqual(result.sync_port, port + 1 + 2 * session_id)
                    self.assertEqual(
                        result.dist_port,
                        port + 2 + (0 if is_direct else 2 * session_id),
                    )
                    if transport != "nccl":
                        self.assertEqual(
                            result.total_attn_ranks, num_attn * num_sessions
                        )
                        self.assertEqual(
                            result.attn_instance_rank_offset, num_attn * session_id
                        )
                        self.assertEqual(result.attn_members[session_id], result.local)
                    self.assertEqual(result.sync_port, peer_result.sync_port)
                    self.assertEqual(result.dist_port, peer_result.dist_port)
                    if is_direct:
                        self.assertEqual(
                            result.peer.direct_backend_version, "test-version"
                        )
                        self.assertIs(result.peer.direct_deepep_fp8_scale_ue8m0, False)
                        continue
                    fan_in = (
                        (num_attn + num_ffn - 1) // num_ffn
                        if backend == "deepep"
                        else 1
                    )
                    buckets = result.peer.decode_graph_token_buckets
                    self.assertIn(graph_capacity * fan_in, buckets)
                    self.assertEqual(
                        buckets[-1], graph_capacity * fan_in * num_sessions
                    )

    def test_handshake_rejects_invalid_model_or_graph_capacity_on_both_peers(self):
        for invalid_graph in (False, True):
            with self.subTest(invalid_graph=invalid_graph):
                port = _free_port()
                ffn_info = _handshake_info("ffn")
                attn_info = _handshake_info("attn")
                if invalid_graph:
                    attn_info.decode_token_buckets = [-1]
                    attn_info.decode_graph_token_buckets = [-1]
                    ffn_info.decode_graph_token_buckets = [1]
                    error_type, message = ValueError, "bucket bounds must be positive"
                else:
                    attn_info.model_identity = "other-model"
                    error_type, message = RuntimeError, "model mismatch"
                with (
                    envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.override(5),
                    ThreadPoolExecutor(max_workers=1) as pool,
                ):
                    server = pool.submit(_accept_handshake, port, ffn_info)
                    with self.assertRaisesRegex(RuntimeError, f"rejected.*{message}"):
                        run_handshake_attn(
                            f"127.0.0.1:{port - AFD_HANDSHAKE_PORT_DELTA}", attn_info
                        )
                    with self.assertRaisesRegex(error_type, message):
                        server.result(timeout=10)

    def test_handshake_rejects_malformed_request_and_replies_with_error(self):
        invalid_info = asdict(_handshake_info("attn"))
        invalid_info["num_ranks"] = "one"
        port = _free_port()
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            closing(zmq.Context.instance().socket(zmq.REQ)) as client,
        ):
            server = pool.submit(_accept_handshake, port, _handshake_info("ffn"))
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
