"""Tests for deferred chunked-prefill aborts."""

import unittest
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
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    def __init__(self, rid: str, *, inflight_middle_chunks: int = 0):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these two predicates.
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.inflight_middle_chunks = inflight_middle_chunks
        self.pending_bootstrap = True
        self.disagg_kv_sender = Mock()
        self.time_stats = SimpleNamespace(trace_ctx=Mock())
        self.return_logprob = False
        self.to_finish = None
        self._finished = False

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.chunked_req = chunked_req
    sched._pending_chunked_abort_req = pending_req
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.running_batch = SimpleNamespace(reqs=running_reqs)
    sched.last_batch = None
    sched.disagg_prefill_pending_chunk_rids = {pending_req.rid}
    sched.req_to_metadata_buffer_idx_allocator = Mock()
    sched.tree_cache = Mock()
    sched.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=Mock())
    )
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        sched.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_finished_req_only_clears_marker(self):
        req = _FakeReq("done_rid")
        req._finished = True
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[])

        sched.process_pending_chunked_abort()

        self.assertIsNone(req.to_finish)
        self.assertIsNone(sched._pending_chunked_abort_req)

    @patch("sglang.srt.managers.scheduler.release_kv_cache")
    def test_inflight_chunked_abort_defers_resource_release(self, release_kv_cache):
        req = _FakeReq("inflight_rid", inflight_middle_chunks=2)
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[req])
        sched.disaggregation_mode = DisaggregationMode.PREFILL
        sched._release_aborted_request = Mock()

        sched.process_pending_chunked_abort()

        self.assertIsNone(sched.chunked_req)
        self.assertIsNone(sched._pending_chunked_abort_req)
        req.disagg_kv_sender.abort.assert_called_once_with()
        release_kv_cache.assert_not_called()
        sched.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        sched._release_aborted_request.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
