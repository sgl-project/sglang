"""In-process integration test for the decoupled enumeration-spec IPC layer.

Wires the verifier-side VerifierIpcThread (recv daemon) + drafter-side
DrafterIpcThread over the fake transport, all in one process, and drives them
deterministically via ``_step()`` (no background threads, no GPU, no sockets).
The verifier lands received enumeration blocks into a CPU stand-in for the GPU
DecoupledEnumBuffer (real ``plan_landing`` routing, no torch scatter); the
drafter drains its control inbox. Exercises the whole open -> enumerate ->
land -> commit -> close loop end to end.
"""

import unittest

from sglang.srt.speculative.decoupled_slot_table import (
    DecoupledSlotTable,
    plan_landing,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftEnumerationBufferBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    VerifyCommit,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    FakeTransportMesh,
    build_transport,
)
from sglang.srt.speculative.drafter_ipc_thread import DrafterIpcThread
from sglang.srt.speculative.verifier_ipc_thread import VerifierIpcThread
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

V_EP = "ipc:///tmp/decoupled-spec-itest-v"
D_EP = "ipc:///tmp/decoupled-spec-itest-d"


def _block(rid="r", dst=0, tok=100) -> DraftEnumerationBufferBatch:
    # Minimal K=1, F=1 enumeration block: row_stride = (K+1)*F*K = 2, one row.
    return DraftEnumerationBufferBatch(
        src_drafter_rank=0,
        dst_verifier_rank=dst,
        num_steps=1,
        fanout=1,
        rids=[rid],
        base_committed_lens=[0],
        tokens=(tok, tok + 1),
    )


class _FakeEnumBuffer:
    """CPU stand-in for DecoupledEnumBuffer's ``land`` (the box has no torch).

    Runs the real host-side prologue -- the K/F dims guard and ``plan_landing``,
    which holds ``verifier_rank`` and raises on a misrouted block -- then records
    the routed plan instead of doing the GPU scatter. Faithful for the IPC layer:
    the wrong-verifier raise the daemon tests assert originates in
    ``plan_landing``, exactly as in the real ``DecoupledEnumBuffer.land``.
    """

    def __init__(self, *, verifier_rank=0, num_steps=1, fanout=1):
        self.verifier_rank = int(verifier_rank)
        self.num_steps = int(num_steps)
        self.fanout = int(fanout)
        self.landed = []  # list[(block, LandingPlan)]

    def land(self, block, slot_table):
        if int(block.num_steps) != self.num_steps or int(block.fanout) != self.fanout:
            raise RuntimeError("enumeration block dims differ from the buffer's config")
        plan = plan_landing(block, slot_table, verifier_rank=self.verifier_rank)
        self.landed.append((block, plan))


def _drain_sync_and_close(token):
    # Simulate the drafter scheduler draining its inbox; commit segments are not
    # consumed here (consumable length 0).
    return token.collect_ready_draft_controls(
        lambda inbox: inbox.extract_ready_controls_locked(lambda seg: 0)
    )


def _drain_all(token):
    # Drain everything including full commit segments.
    return token.collect_ready_draft_controls(
        lambda inbox: inbox.extract_ready_controls_locked(
            lambda seg: len(seg.committed_tokens)
        )
    )


class TestDecoupledSpecIpcIntegration(CustomTestCase):
    def _wire(self):
        mesh = FakeTransportMesh()
        v_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=V_EP,
            connect_endpoints=[D_EP],
            mesh=mesh,
        )
        d_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=D_EP,
            connect_endpoints=[V_EP],
            mesh=mesh,
        )
        v_tp.start()
        d_tp.start()
        slot_table = DecoupledSlotTable()
        enum_buffer = _FakeEnumBuffer(verifier_rank=0, num_steps=1, fanout=1)
        proxy = VerifierIpcThread(
            transport=v_tp, enum_buffer=enum_buffer, slot_table=slot_table
        )
        token = DrafterIpcThread(transport=d_tp, drafter_rank=0)
        return mesh, v_tp, d_tp, slot_table, enum_buffer, proxy, token

    def test_full_loop_open_enumerate_land_commit_close(self):
        _mesh, v_tp, d_tp, slot_table, enum_buffer, proxy, token = self._wire()
        try:
            # 1. verifier opens a draft request (DraftSync). The scheduler (stood
            #    in for here) binds the request's seat at prefill so a landing
            #    block can be routed to it.
            slot_table.assign("r", 0)
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    sync_messages=[
                        DraftSync(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            committed_outputs=[],
                        )
                    ],
                )
            )
            proxy._step()  # forward over the transport

            # 2. drafter receives the control into its inbox.
            token._step()
            ready = _drain_sync_and_close(token)
            self.assertEqual([m.request_id for m in ready.sync_messages], ["r"])

            # 3. drafter pushes one enumeration block back.
            token.submit_draft_results(_block(rid="r", dst=0, tok=100))
            token._step()  # send to verifier

            # 4. verifier lands the block into its (fake) enum buffer, routed to
            #    the request's assigned seat.
            proxy._step()
            self.assertEqual(len(enum_buffer.landed), 1)
            landed_block, plan = enum_buffer.landed[0]
            self.assertEqual(landed_block.rids, ["r"])
            self.assertEqual([w.pool_idx for w in plan.writes], [0])
            self.assertEqual(plan.dropped_rids, [])

            # 5. verifier commits the token; drafter sees the committed segment.
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    verify_commit_messages=[
                        VerifyCommit(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            pre_verify_committed_len=0,
                            committed_tokens=[100],
                        )
                    ],
                )
            )
            proxy._step()
            token._step()
            ready2 = _drain_all(token)
            self.assertEqual(len(ready2.ready_commit_segments), 1)
            self.assertEqual(ready2.ready_commit_segments[0].committed_tokens, [100])

            # 6. verifier closes the request; the scheduler drops the seat so a
            #    late block can no longer land in it.
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    close_messages=[
                        DraftClose(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            reason="finished",
                        )
                    ],
                )
            )
            slot_table.remove("r")
            self.assertIsNone(slot_table.lookup("r"))
            proxy._step()
            token._step()
            ready3 = _drain_sync_and_close(token)
            self.assertEqual(len(ready3.close_keys), 1)
        finally:
            v_tp.close()
            d_tp.close()

    def test_block_for_unbound_rid_is_dropped_not_landed(self):
        # A block whose rid has no seat (request finished / not yet opened) is
        # dropped by plan_landing rather than scattered into a stale seat.
        _mesh, v_tp, d_tp, _slot_table, enum_buffer, proxy, token = self._wire()
        try:
            token.submit_draft_results(_block(rid="ghost", dst=0, tok=7))
            token._step()
            proxy._step()
            self.assertEqual(len(enum_buffer.landed), 1)
            _landed_block, plan = enum_buffer.landed[0]
            self.assertEqual(plan.writes, [])
            self.assertEqual(plan.dropped_rids, ["ghost"])
        finally:
            v_tp.close()
            d_tp.close()

    def test_token_drops_control_for_wrong_drafter_rank(self):
        # token is drafter rank 0.
        _mesh, v_tp, d_tp, _slot_table, _enum_buffer, _proxy, token = self._wire()
        try:
            # Inject a control batch addressed to drafter rank 5 onto drafter 0's wire.
            v_tp.send(
                0,
                DraftMeshMessage.from_control_batch(
                    DraftControlBatch(
                        dst_drafter_rank=5,
                        sync_messages=[
                            DraftSync(
                                request_id="x",
                                src_verifier_rank=0,
                                dst_drafter_rank=5,
                                committed_outputs=[],
                            )
                        ],
                    )
                ),
            )
            token._step()
            ready = _drain_sync_and_close(token)
            self.assertEqual(ready.sync_messages, [])  # dropped: wrong drafter rank
        finally:
            v_tp.close()
            d_tp.close()

    def test_drafter_sends_each_block_to_its_verifier(self):
        # An enumeration block carries a single dst_verifier_rank, so a drafter
        # serving two verifiers submits one block per verifier and each routes to
        # its own peer (no per-row grouping).
        mesh = FakeTransportMesh()
        v0_ep = "ipc:///tmp/ds-fanout-v0"
        v1_ep = "ipc:///tmp/ds-fanout-v1"
        d_ep = "ipc:///tmp/ds-fanout-d"
        d_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=d_ep,
            connect_endpoints=[v0_ep, v1_ep],
            mesh=mesh,
        )
        v0_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=v0_ep,
            connect_endpoints=[d_ep],
            mesh=mesh,
        )
        v1_tp = build_transport(
            kind=DecoupledSpecTransportKind.FAKE,
            bind_endpoint=v1_ep,
            connect_endpoints=[d_ep],
            mesh=mesh,
        )
        d_tp.start()
        v0_tp.start()
        v1_tp.start()
        token = DrafterIpcThread(transport=d_tp, drafter_rank=0)
        try:
            token.submit_draft_results(_block(rid="a", dst=0, tok=10))
            token.submit_draft_results(_block(rid="b", dst=1, tok=20))
            token._step()
            m0 = v0_tp.try_recv()
            m1 = v1_tp.try_recv()
            self.assertEqual(m0.enumeration_buffer_batch.rids, ["a"])
            self.assertEqual(m0.enumeration_buffer_batch.tokens[0], 10)
            self.assertEqual(m1.enumeration_buffer_batch.rids, ["b"])
            self.assertEqual(m1.enumeration_buffer_batch.tokens[0], 20)
        finally:
            d_tp.close()
            v0_tp.close()
            v1_tp.close()

    def test_proxy_rejects_block_for_wrong_verifier_rank(self):
        # proxy is verifier rank 0.
        _mesh, v_tp, d_tp, _slot_table, _enum_buffer, proxy, _token = self._wire()
        try:
            # Inject a block addressed to verifier rank 9 onto verifier 0's wire.
            d_tp.send(
                0,
                DraftMeshMessage.from_enumeration_buffer_batch(
                    _block(rid="r", dst=9, tok=1)
                ),
            )
            # Router-level invariant: land() -> plan_landing (driven here via
            # _step) rejects a block for the wrong verifier. Production containment
            # of that raise by the daemon loop is covered by the _run test below.
            with self.assertRaises(RuntimeError):
                proxy._step()
        finally:
            v_tp.close()
            d_tp.close()

    def test_proxy_run_terminates_loudly_on_wrong_verifier_rank(self):
        # The daemon loop must NOT let a router RuntimeError escape and silently
        # kill the proxy for all requests: _run logs it and breaks cleanly.
        # (Phase 5c will quarantine the offending request instead.)
        _mesh, v_tp, d_tp, _slot_table, _enum_buffer, proxy, _token = self._wire()
        try:
            d_tp.send(
                0,
                DraftMeshMessage.from_enumeration_buffer_batch(
                    _block(rid="r", dst=9, tok=1)
                ),
            )
            # _run returns (breaks) instead of propagating, and logs loudly.
            with self.assertLogs(
                "sglang.srt.speculative.verifier_ipc_thread", level="ERROR"
            ):
                proxy._run()
        finally:
            v_tp.close()
            d_tp.close()

    def test_token_run_terminates_loudly_on_malformed_control(self):
        # Mirror of the proxy test on the drafter side: a malformed control
        # envelope makes _route_control_message raise; _run must contain it.
        _mesh, v_tp, d_tp, _slot_table, _enum_buffer, _proxy, token = self._wire()
        try:
            # CONTROL_BATCH message_type with a None payload is malformed.
            v_tp.send(
                0,
                DraftMeshMessage(
                    message_type=DraftMeshMessageType.CONTROL_BATCH,
                    control_batch=None,
                ),
            )
            with self.assertLogs(
                "sglang.srt.speculative.drafter_ipc_thread", level="ERROR"
            ):
                token._run()
        finally:
            v_tp.close()
            d_tp.close()


if __name__ == "__main__":
    unittest.main()
