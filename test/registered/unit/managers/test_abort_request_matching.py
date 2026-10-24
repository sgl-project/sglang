"""Targeted aborts must not treat caller-supplied IDs as request families."""

import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    WaitingMMRequestStatus,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.session.session_controller import SessionReqNode

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RIDS = ["job-1", "job-10", "job-11", "job-2", "job-1_0", "job-1_0_extra"]


def make_req(rid):
    req = Mock(rid=rid)
    req.finished.return_value = False
    req.to_finish = None
    req.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)
    req.grammar = Future()
    return req


def make_scheduler():
    sched = Scheduler.__new__(Scheduler)
    sched.chunked_req = None
    sched._pending_chunked_abort_req = None
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = DisaggregationMode.NULL
    sched.mm_receiver = None
    sched.running_batch = SimpleNamespace(reqs=[])
    sched.last_batch = None
    sched._release_aborted_request = Mock()
    sched.beam_coordinator = Mock()
    sched.ipc_channels = Mock()
    sched.tree_cache = Mock()
    return sched


class TestAbortRequestMatching(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def cases(self):
        for target, abort_all in [
            ("job-1", False),
            ("job-1_0", False),
            ("missing", False),
            ("", True),
        ]:
            yield (
                AbortReq(rid=target, abort_all=abort_all),
                {rid for rid in RIDS if abort_all or rid == target},
            )

    def test_waiting_and_running(self):
        for queue in ("waiting", "running", "last_batch"):
            for abort, expected in self.cases():
                with self.subTest(queue=queue, abort=abort):
                    sched = make_scheduler()
                    reqs = [make_req(rid) for rid in RIDS]
                    if queue == "waiting":
                        sched.waiting_queue = list(reqs)
                    elif queue == "running":
                        sched.running_batch.reqs = reqs
                    else:
                        sched.last_batch = SimpleNamespace(reqs=reqs)
                    with patch("sglang.srt.managers.scheduler._make_abort_req"):
                        sched.abort_request(abort)
                    if queue == "waiting":
                        self.assertEqual(
                            {r.rid for r in sched.waiting_queue}, set(RIDS) - expected
                        )
                        self.assertEqual(
                            {
                                c.args[0].rid
                                for c in sched._release_aborted_request.call_args_list
                            },
                            expected,
                        )
                        self.assertEqual(
                            sched.ipc_channels.send_to_tokenizer.send_output.call_count,
                            len(expected),
                        )
                    else:
                        self.assertEqual(
                            {r.rid for r in reqs if r.to_finish is not None}, expected
                        )

    def test_finished_running_request_is_ignored(self):
        sched = make_scheduler()
        req = make_req("job-1")
        req.finished.return_value = True
        sched.running_batch.reqs = [req]
        sched.abort_request(AbortReq(rid="job-1"))
        self.assertIsNone(req.to_finish)

    def test_chunked_prefill(self):
        for abort, expected in self.cases():
            for rid in RIDS:
                with self.subTest(abort=abort, rid=rid):
                    sched = make_scheduler()
                    req = sched.chunked_req = make_req(rid)
                    sched.abort_request(abort)
                    self.assertIs(
                        sched._pending_chunked_abort_req,
                        req if rid in expected else None,
                    )

    def test_grammar_queue(self):
        for abort, expected in self.cases():
            with self.subTest(abort=abort):
                reqs = [make_req(rid) for rid in RIDS]
                manager = SimpleNamespace(grammar_queue=reqs)
                GrammarManager.abort_requests(manager, abort)
                self.assertEqual(
                    {r.rid for r in reqs if r.grammar.cancelled()}, expected
                )
                self.assertEqual(
                    {r.rid for r in reqs if r.set_finish_with_abort.called}, expected
                )

    def test_pd_prefill_queues(self):
        for abort, expected in self.cases():
            with self.subTest(abort=abort):
                sched = make_scheduler()
                sched.disaggregation_mode = DisaggregationMode.PREFILL
                bootstrap = [make_req(rid) for rid in RIDS]
                inflight = [make_req(rid) for rid in RIDS]
                sched.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=bootstrap)
                sched.disagg_prefill_inflight_queue = inflight
                sched.abort_request(abort)
                for reqs in (bootstrap, inflight):
                    self.assertEqual(
                        {r.rid for r in reqs if r.disagg_kv_sender.abort.called},
                        expected,
                    )

    def test_pd_decode_queues(self):
        for abort, expected in self.cases():
            with self.subTest(abort=abort):
                sched = make_scheduler()
                sched.disaggregation_mode = DisaggregationMode.DECODE
                prealloc = [Mock(req=make_req(rid)) for rid in RIDS]
                transfer = [Mock(req=make_req(rid), host_staged=False) for rid in RIDS]
                retracted = [make_req(rid) for rid in RIDS]
                sched.disagg_decode_prealloc_queue = SimpleNamespace(
                    queue=prealloc, retracted_queue=retracted
                )
                sched.disagg_decode_transfer_queue = SimpleNamespace(queue=transfer)
                with (
                    patch(
                        "sglang.srt.managers.scheduler.discard_kv_cache_backup"
                    ) as discard,
                    patch("sglang.srt.managers.scheduler._make_abort_req"),
                ):
                    sched.abort_request(abort)
                for reqs in (prealloc, transfer):
                    self.assertEqual(
                        {r.req.rid for r in reqs if r.kv_receiver.abort.called},
                        expected,
                    )
                self.assertEqual(
                    {c.args[0].rid for c in discard.call_args_list}, expected
                )
                self.assertEqual(
                    {r.rid for r in sched.disagg_decode_prealloc_queue.retracted_queue},
                    set(RIDS) - expected,
                )

    def test_dllm_waiting_and_staging(self):
        for abort, expected in self.cases():
            with self.subTest(abort=abort):
                reqs = [make_req(rid) for rid in RIDS]
                manager = SimpleNamespace(
                    waiting_queue=list(reqs), staging_queue=list(reqs)
                )
                aborted = DllmManager.pop_aborted_reqs(
                    manager, abort.abort_all, abort.rid
                )
                self.assertEqual({r.rid for r in aborted}, expected)
                self.assertEqual(len(aborted), len(expected))
                for queue in (manager.waiting_queue, manager.staging_queue):
                    self.assertEqual({r.rid for r in queue}, set(RIDS) - expected)

    def test_session_clear_uses_explicit_children(self):
        # Session replacement/close has its own explicit tree; child names
        # need not resemble the parent's RID at all.
        parent, child, grandchild, unrelated = [
            make_req(rid) for rid in ("job-1", "child", "grandchild", "job-1_0")
        ]
        nodes = {}
        for req, parent_node in (
            (parent, None),
            (child, parent),
            (grandchild, child),
            (unrelated, None),
        ):
            req.finished_reason = None
            nodes[req.rid] = SessionReqNode(
                req, parent=nodes[parent_node.rid] if parent_node else None
            )
        nodes[parent.rid].clear(nodes)
        self.assertEqual(set(nodes), {unrelated.rid})
        for req in (parent, child, grandchild):
            self.assertEqual(req.to_finish.to_json()["type"], "abort")
        self.assertIsNone(unrelated.to_finish)

    def test_encoder_waiting_queue(self):
        for status in (WaitingMMRequestStatus.PENDING, WaitingMMRequestStatus.SUCCESS):
            for abort, expected in self.cases():
                with self.subTest(status=status, abort=abort):
                    reqs = [Mock(rid=rid, status=status) for rid in RIDS]
                    receiver = SimpleNamespace(waiting_list=reqs)
                    MMReceiverBase.abort_waiting_requests(receiver, abort)
                    self.assertEqual(
                        {r.rid for r in reqs if r._fail_and_release.called}, expected
                    )
                    self.assertEqual(
                        {r.rid for r in reqs if r.release_resources.called}, expected
                    )


if __name__ == "__main__":
    unittest.main()
