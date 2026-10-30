"""A finish decided outside a step commit is recorded; the drain finishes it
once per loop iteration, and the request's memory waits for any launched step
that carries it."""

import unittest
from collections import deque
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

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.pending_finishes import (  # noqa: E402
    PendingFinishes,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the finish paths touch."""

    request_finish = Req.request_finish
    finish_outside_step = Req.finish_outside_step
    finished = Req.finished

    def __init__(self, rid: str):
        self.rid = rid
        self.to_finish = None
        self.finished_reason = None
        self.beam_group = None
        self.inflight_middle_chunks = 0
        self.return_logprob = False
        self.output_ids = []
        self.weight_version_events = []
        self.time_stats = Mock()
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)


def _batch(*reqs):
    return SimpleNamespace(reqs=list(reqs))


def _scheduler(*, chunked_req=None) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.pending_finishes = PendingFinishes()
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.batch_result_processor = Mock()
    s.output_streamer = Mock()
    s.beam_coordinator = Mock()
    s._release_aborted_request = Mock()
    s.tree_cache = SimpleNamespace(supports_mamba=lambda: False)
    s.enable_skip_finishing_decode = False
    s.enable_continuous_input_polling = False
    s.result_queue = deque()
    return s


class TestPendingFinishes(CustomTestCase):
    def test_first_reason_wins(self):
        req, done = _FakeReq("r"), _FakeReq("done")
        done.finished_reason = FINISH_LENGTH(length=1)
        abort, failed = FINISH_ABORT("client"), FINISH_ABORT("load failed", 503)
        finishes = PendingFinishes()

        finishes.add(req, abort)
        finishes.add(req, failed)
        finishes.add(done, abort)

        self.assertEqual(finishes.reqs(), [req])
        self.assertIs(finishes.pop(req), abort)

    def test_drain_takes_every_finish_but_a_finished_request(self):
        first, gone, last = (_FakeReq(rid) for rid in "abc")
        finishes = PendingFinishes()
        for req in (first, gone, last):
            finishes.add(req, FINISH_ABORT())
        gone.finished_reason = FINISH_LENGTH(length=1)

        taken = finishes.take_all()

        self.assertEqual([req for req, _ in taken], [first, last])
        self.assertFalse(finishes)


class TestSchedulerPendingFinishes(CustomTestCase):
    def test_drain_finishes_a_running_request_and_retires_it(self):
        # No step result says whether a mamba tree could cache the state.
        for supports_mamba, checkpoint in ((False, True), (True, False)):
            with self.subTest(supports_mamba=supports_mamba):
                req, reason = _FakeReq("running"), FINISH_ABORT()
                s = _scheduler()
                s.tree_cache = SimpleNamespace(supports_mamba=lambda: supports_mamba)
                s.pending_finishes.add(req, reason)

                s.drain_pending_finishes()

                self.assertIs(req.finished_reason, reason)
                retire = s.batch_result_processor.retire_outside_step
                retire.assert_called_once_with(req, checkpoint=checkpoint)
                s.output_streamer.stream_output.assert_called_once_with([req], False)
                self.assertFalse(s.pending_finishes)


class TestAbortRunningRequest(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_abort_skips_requests_a_queue_branch_took(self):
        running, requeued = _FakeReq("running"), _FakeReq("requeued")
        requeued.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)
        s = _scheduler()
        s._deferred_responses = None
        s.mm_receiver = None
        s.dllm_config = None
        s.grammar_manager = Mock()
        s.rust_server = None
        s.ipc_channels = Mock()
        # A request requeued after its step ran is still in the last batch.
        s.waiting_queue = [requeued]
        s.running_batch = _batch(running)
        s.last_batch = _batch(requeued)

        with patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        ):
            # Two aborts in one intake: a client abort, then an abort_all.
            for rid, abort_all in (("requeued", False), ("", True)):
                s.abort_request(
                    SimpleNamespace(
                        abort_all=abort_all,
                        rid=rid,
                        abort_message=None,
                        finished_reason=None,
                    )
                )

        self.assertEqual(s.waiting_queue, [])
        self.assertEqual(s.pending_finishes.reqs(), [running])
        s._release_aborted_request.assert_called_once_with(requeued)
        self.assertEqual(s.ipc_channels.send_to_tokenizer.send_output.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
