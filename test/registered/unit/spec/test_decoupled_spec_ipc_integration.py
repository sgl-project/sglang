"""In-process integration test for the decoupled-spec IPC layer.

Wires the verifier-side DraftProxyThread + drafter-side TokenSyncThread + a
DraftTailBuffer over the fake transport, all in one process, and drives them
deterministically via ``_step()`` (no background threads, no GPU, no sockets).
This is the test class that the fake transport unblocks: it exercises the whole
open -> stream -> commit -> close loop end to end.
"""

import unittest

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftMeshMessage,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    DecoupledSpecTransportKind,
    FakeTransportMesh,
    build_transport,
)
from sglang.srt.speculative.draft_proxy import DraftProxyThread
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.srt.speculative.token_sync_thread import TokenSyncThread
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

V_EP = "ipc:///tmp/decoupled-spec-itest-v"
D_EP = "ipc:///tmp/decoupled-spec-itest-d"


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid


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
            lambda seg: len(seg.committed_token_ids)
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
        buf = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        proxy = DraftProxyThread(transport=v_tp, verifier_rank=0, draft_tail_buffer=buf)
        token = TokenSyncThread(transport=d_tp, drafter_rank=0)
        return mesh, v_tp, d_tp, buf, proxy, token

    def test_full_loop_open_stream_commit_close(self):
        _mesh, v_tp, d_tp, buf, proxy, token = self._wire()
        try:
            # 1. verifier opens a draft request (DraftSync).
            proxy.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    sync_messages=[
                        DraftSync(
                            request_id="r",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            committed_output_ids=[],
                        )
                    ],
                )
            )
            self.assertTrue(buf.has_request("r"))  # applied to the verifier mirror now
            proxy._step()  # forward over the transport

            # 2. drafter receives the control into its inbox.
            token._step()
            ready = _drain_sync_and_close(token)
            self.assertEqual([m.request_id for m in ready.sync_messages], ["r"])

            # 3. drafter streams one draft tail token back.
            token.submit_draft_results(
                DraftTailStreamOutputBatch(
                    outputs=[
                        DraftTailStreamOutput(
                            src_drafter_rank=0,
                            dst_verifier_rank=0,
                            request_id="r",
                            base_committed_len=0,
                            new_token_pos=0,
                            new_token_id=100,
                        )
                    ]
                )
            )
            token._step()  # send to verifier

            # 4. verifier ingests the tail into the buffer.
            proxy._step()
            snap = buf.get_draft_snapshots([_FakeReq("r")])[0]
            self.assertEqual(snap.tail_tokens, [100])

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
                            committed_token_ids=[100],
                        )
                    ],
                )
            )
            self.assertEqual(buf.get_committed_len("r"), 1)  # matched the buffered tail
            proxy._step()
            token._step()
            ready2 = _drain_all(token)
            self.assertEqual(len(ready2.ready_commit_segments), 1)
            self.assertEqual(ready2.ready_commit_segments[0].committed_token_ids, [100])

            # 6. verifier closes the request.
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
            self.assertFalse(buf.has_request("r"))  # verifier mirror dropped
            proxy._step()
            token._step()
            ready3 = _drain_sync_and_close(token)
            self.assertEqual(len(ready3.close_keys), 1)
        finally:
            v_tp.close()
            d_tp.close()

    def test_token_drops_control_for_wrong_drafter_rank(self):
        _mesh, v_tp, d_tp, _buf, _proxy, token = self._wire()  # token is drafter rank 0
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
                                committed_output_ids=[],
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

    def test_drafter_fans_out_to_multiple_verifiers(self):
        # The drafter groups its outgoing tail outputs by dst_verifier_rank and
        # routes each group to the right verifier peer.
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
        token = TokenSyncThread(transport=d_tp, drafter_rank=0)
        try:
            token.submit_draft_results(
                DraftTailStreamOutputBatch(
                    outputs=[
                        DraftTailStreamOutput(
                            src_drafter_rank=0,
                            dst_verifier_rank=0,
                            request_id="a",
                            base_committed_len=0,
                            new_token_pos=0,
                            new_token_id=10,
                        ),
                        DraftTailStreamOutput(
                            src_drafter_rank=0,
                            dst_verifier_rank=1,
                            request_id="b",
                            base_committed_len=0,
                            new_token_pos=0,
                            new_token_id=20,
                        ),
                    ]
                )
            )
            token._step()
            m0 = v0_tp.try_recv()
            m1 = v1_tp.try_recv()
            self.assertEqual(m0.tail_stream_output_batch.outputs[0].new_token_id, 10)
            self.assertEqual(m1.tail_stream_output_batch.outputs[0].new_token_id, 20)
        finally:
            d_tp.close()
            v0_tp.close()
            v1_tp.close()

    def test_proxy_rejects_tail_for_wrong_verifier_rank(self):
        _mesh, v_tp, d_tp, _buf, proxy, _token = (
            self._wire()
        )  # proxy is verifier rank 0
        try:
            # Inject a tail output addressed to verifier rank 9 onto verifier 0's wire.
            d_tp.send(
                0,
                DraftMeshMessage.from_tail_stream_output_batch(
                    DraftTailStreamOutputBatch(
                        outputs=[
                            DraftTailStreamOutput(
                                src_drafter_rank=0,
                                dst_verifier_rank=9,
                                request_id="r",
                                base_committed_len=0,
                                new_token_pos=0,
                                new_token_id=1,
                            )
                        ]
                    )
                ),
            )
            with self.assertRaises(RuntimeError):
                proxy._step()
        finally:
            v_tp.close()
            d_tp.close()


if __name__ == "__main__":
    unittest.main()
