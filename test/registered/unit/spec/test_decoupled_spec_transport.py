"""CPU unit tests for the decoupled-spec transport layer.

Exercises the transport interface + factory through the in-process fake backend
(``FakeTransport`` over a shared ``FakeTransportMesh``): round-trip, ordering,
multi-peer routing, idle wait, dead-peer injection, and the factory dispatch.
No GPU, no real sockets.
"""

import threading
import unittest

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    FakeTransport,
    FakeTransportMesh,
    ZmqTransport,
    build_transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

EP_A = "ipc:///tmp/decoupled-spec-fake-a"
EP_B = "ipc:///tmp/decoupled-spec-fake-b"
EP_C = "ipc:///tmp/decoupled-spec-fake-c"


def _control_msg(rid="r", drafter_rank=0) -> DraftMeshMessage:
    batch = DraftControlBatch(
        dst_drafter_rank=drafter_rank,
        sync_messages=[
            DraftSync(
                request_id=rid, src_verifier_rank=0, dst_drafter_rank=drafter_rank
            )
        ],
    )
    return DraftMeshMessage.from_control_batch(batch)


def _tail_msg(rid="r", verifier_rank=0, tok=7) -> DraftMeshMessage:
    out = DraftTailStreamOutput(
        src_drafter_rank=0,
        dst_verifier_rank=verifier_rank,
        request_id=rid,
        base_committed_len=0,
        new_token_pos=0,
        new_token=tok,
    )
    return DraftMeshMessage.from_tail_stream_output_batch(
        DraftTailStreamOutputBatch(outputs=[out])
    )


def _fake(mesh, bind, connect):
    t = build_transport(
        kind=DecoupledSpecTransportKind.FAKE,
        bind_endpoint=bind,
        connect_endpoints=connect,
        mesh=mesh,
    )
    t.start()
    return t


class TestTransportFactory(CustomTestCase):
    def test_factory_returns_fake(self):
        mesh = FakeTransportMesh()
        t = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=EP_A,
            connect_endpoints=[EP_B],
            mesh=mesh,
        )
        self.assertIsInstance(t, FakeTransport)

    def test_factory_coerces_string_kind(self):
        mesh = FakeTransportMesh()
        t = build_transport(
            kind="fake", bind_endpoint=EP_A, connect_endpoints=[EP_B], mesh=mesh
        )
        self.assertIsInstance(t, FakeTransport)

    def test_factory_returns_zmq_without_importing_or_starting(self):
        # Constructing the ZMQ backend must not require pyzmq (import is deferred
        # to start()), so this works even on a box without zmq installed.
        t = build_transport(
            kind=DecoupledSpecTransportKind.ZMQ,
            bind_endpoint=EP_A,
            connect_endpoints=[EP_B],
        )
        self.assertIsInstance(t, ZmqTransport)

    def test_fake_without_mesh_raises(self):
        with self.assertRaises(ValueError):
            build_transport(
                kind=DecoupledSpecTransportKind.FAKE,
                bind_endpoint=EP_A,
                connect_endpoints=[EP_B],
            )

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            build_transport(kind="rdma", bind_endpoint=EP_A, connect_endpoints=[EP_B])


class TestFakeTransport(CustomTestCase):
    def _pair(self):
        mesh = FakeTransportMesh()
        a = _fake(mesh, EP_A, [EP_B])
        b = _fake(mesh, EP_B, [EP_A])
        return mesh, a, b

    def test_roundtrip_both_directions(self):
        _mesh, a, b = self._pair()
        a.send(0, _control_msg(rid="r0"))
        got = b.try_recv()
        self.assertIsNotNone(got)
        self.assertEqual(got.message_type, DraftMeshMessageType.CONTROL_BATCH)
        self.assertEqual(got.control_batch.sync_messages[0].request_id, "r0")

        b.send(0, _tail_msg(rid="r0", tok=42))
        got2 = a.try_recv()
        self.assertIsNotNone(got2)
        self.assertEqual(
            got2.message_type, DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH
        )
        self.assertEqual(got2.tail_stream_output_batch.outputs[0].new_token, 42)

    def test_try_recv_empty_returns_none(self):
        _mesh, a, _b = self._pair()
        self.assertIsNone(a.try_recv())

    def test_fifo_order_preserved(self):
        _mesh, a, b = self._pair()
        for i in range(3):
            a.send(0, _control_msg(rid=f"r{i}"))
        rids = []
        while (msg := b.try_recv()) is not None:
            rids.append(msg.control_batch.sync_messages[0].request_id)
        self.assertEqual(rids, ["r0", "r1", "r2"])

    def test_multi_peer_routing(self):
        mesh = FakeTransportMesh()
        a = _fake(mesh, EP_A, [EP_B, EP_C])  # peer rank 0 -> B, rank 1 -> C
        b = _fake(mesh, EP_B, [EP_A])
        c = _fake(mesh, EP_C, [EP_A])
        a.send(0, _control_msg(rid="to_b"))
        a.send(1, _control_msg(rid="to_c"))
        self.assertEqual(b.try_recv().control_batch.sync_messages[0].request_id, "to_b")
        self.assertEqual(c.try_recv().control_batch.sync_messages[0].request_id, "to_c")
        self.assertIsNone(b.try_recv())
        self.assertIsNone(c.try_recv())

    def test_send_out_of_range_rank_raises(self):
        _mesh, a, _b = self._pair()
        with self.assertRaises(RuntimeError):
            a.send(5, _control_msg())

    def test_wait_for_input(self):
        _mesh, a, b = self._pair()
        # Nothing pending -> times out -> False.
        self.assertFalse(b.wait_for_input(0.01))
        # After a delivery -> True.
        a.send(0, _control_msg())
        self.assertTrue(b.wait_for_input(0.01))

    def test_close_unregisters_inbound(self):
        _mesh, a, b = self._pair()
        b.send(0, _control_msg())  # a now has a pending inbound message
        a.close()  # unregister drops a's inbox along with the pending message
        self.assertIsNone(a.try_recv())

    def test_dead_peer_drops_send_and_reports_unreachable(self):
        mesh, a, b = self._pair()
        self.assertTrue(a.is_peer_alive(0))
        mesh.kill(EP_B)  # b's inbound is now dead
        self.assertFalse(a.is_peer_alive(0))
        a.send(0, _control_msg())  # dropped silently
        self.assertIsNone(b.try_recv())

    def test_wait_wakes_when_own_endpoint_killed(self):
        mesh, _a, b = self._pair()
        mesh.kill(EP_B)  # b's own endpoint
        # A killed endpoint must not leave a waiter blocked forever.
        self.assertTrue(b.wait_for_input(0.01))


class TestFakeTransportThreaded(CustomTestCase):
    def test_background_waiter_wakes_on_delivery(self):
        mesh = FakeTransportMesh()
        a = _fake(mesh, EP_A, [EP_B])
        b = _fake(mesh, EP_B, [EP_A])
        result = {}

        def waiter():
            if b.wait_for_input(5.0):
                result["msg"] = b.try_recv()

        t = threading.Thread(target=waiter)
        t.start()
        a.send(0, _control_msg(rid="async"))
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        self.assertIsNotNone(result.get("msg"))
        self.assertEqual(
            result["msg"].control_batch.sync_messages[0].request_id, "async"
        )


if __name__ == "__main__":
    unittest.main()
