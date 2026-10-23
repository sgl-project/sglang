"""CPU tests for typed AFD control messages and startup handshakes."""

import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
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
    AFDSyncMsg,
    AFDSyncMsgReceiver,
    AFDSyncMsgSender,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _handshake_info(role):
    return AFDHandshakeInfo(
        role=role,
        model_identity="test-model:123",
        num_ranks=1,
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
        if port <= 65535 - 2:
            return port


def _accept_handshake(port, info):
    with closing(AFDHandshakeServer(port)) as server:
        server._socket.setsockopt(zmq.RCVTIMEO, 5000)
        return server.accept(info)


class TestAFDControlWireProtocol(CustomTestCase):
    def _sync_channel(self):
        endpoint = f"inproc://afd-sync-test-{uuid4()}"
        receiver = AFDSyncMsgReceiver(endpoint)
        sender = AFDSyncMsgSender(endpoint)
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        receiver.socket.setsockopt(zmq.RCVTIMEO, 5000)
        return sender, receiver

    def test_sync_roundtrip_preserves_types_and_step_order(self):
        sender, receiver = self._sync_channel()
        for step_id in range(2):
            msg = AFDSyncMsg(step_id, [3, 4], 1, 8)
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


class TestAFDHandshakeWireProtocol(CustomTestCase):
    def test_handshake_roundtrip_agrees_on_endpoints_and_model(self):
        for num_attn, num_ffn, backend in (
            (1, 1, "none"),
            (3, 2, "none"),
            (2, 3, "deepep"),
        ):
            with self.subTest(num_attn=num_attn, num_ffn=num_ffn, backend=backend):
                port = _free_port()
                ffn_info = _handshake_info("ffn")
                attn_info = _handshake_info("attn")
                ffn_info.num_ranks = num_ffn
                attn_info.num_ranks = num_attn
                ffn_info.afd_ffn_moe_backend = backend
                attn_info.afd_ffn_moe_backend = backend
                attn_info.decode_token_buckets = [96]
                with (
                    envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.override(5),
                    ThreadPoolExecutor(max_workers=1) as pool,
                ):
                    server = pool.submit(_accept_handshake, port, ffn_info)
                    result = run_handshake_attn(
                        f"127.0.0.1:{port - AFD_HANDSHAKE_PORT_DELTA}", attn_info
                    )
                    peer_result = server.result(timeout=10)
                self.assertEqual(result.local, attn_info)
                self.assertEqual(result.peer, ffn_info)
                self.assertEqual(peer_result.local, ffn_info)
                self.assertEqual(peer_result.peer, attn_info)
                self.assertEqual(result.dist_host, "127.0.0.1")
                self.assertEqual(result.sync_port, port + 1)
                self.assertEqual(result.dist_port, port + 2)
                self.assertEqual(result.sync_port, peer_result.sync_port)
                self.assertEqual(result.dist_port, peer_result.dist_port)

    def test_handshake_rejects_model_mismatch_on_both_peers(self):
        port = _free_port()
        ffn_info = _handshake_info("ffn")
        attn_info = _handshake_info("attn")
        attn_info.model_identity = "other-model"
        with (
            envs.SGLANG_AFD_HANDSHAKE_TIMEOUT_S.override(5),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            server = pool.submit(_accept_handshake, port, ffn_info)
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
