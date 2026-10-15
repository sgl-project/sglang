"""Tests for deferred chunked-prefill aborts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.request_lists import CoreLists  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.mode_owners import (  # noqa: E402
    ModeOwners,
)
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402
from sglang.test.scheduler_request_lists import request_registry_of  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    def __init__(self, rid: str):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these two predicates.
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.to_finish = None
        self._finished = False
        self.discard_output_reason = None
        self.return_logprob = False
        self.output_ids = []
        self.weight_version_events = []
        self.time_stats = Mock()

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = partial_scheduler()
    sched.parts.mode_owners = ModeOwners()
    sched.parts.lists = CoreLists()
    sched.parts.lists.chunked_req = chunked_req
    sched.parts.lists._pending_chunked_abort_req = pending_req
    sched.parts.lists.waiting_queue = []
    sched.parts.mode_owners.dllm = None
    sched.parts.grammar_manager = Mock()
    sched.mode.disaggregation_mode = None
    sched.parts.mode_owners.disagg_prefill = None
    sched.parts.mode_owners.disagg_decode = None
    sched.mm_receiver = None
    sched.parts.lists.decoding = list(running_reqs)
    sched.parts.mode_owners.hisparse_coordinator = None
    sched.hub.request_registry = request_registry_of(sched)
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        sched.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(sched.parts.lists._pending_chunked_abort_req)

    def test_finished_req_only_clears_marker(self):
        req = _FakeReq("done_rid")
        req._finished = True
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[])

        sched.process_pending_chunked_abort()

        self.assertIsNone(req.to_finish)
        self.assertIsNone(sched.parts.lists._pending_chunked_abort_req)

    def test_discarded_chunk_keeps_its_abort_reason(self):
        req = _FakeReq("failed_load")
        req.discard_output_reason = FINISH_ABORT("External KV cache load failed", 503)
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])
        sched.parts.request_retirer = Mock()
        sched.parts.kv.tree_cache = None
        sched.parts.ipc_channels = Mock()

        sched.process_pending_chunked_abort()

        self.assertEqual(req.finished_reason.status_code, 503)
        (abort_req, _), _ = (
            sched.parts.ipc_channels.send_to_tokenizer.send_output.call_args
        )
        self.assertEqual(abort_req.finished_reason, req.discard_output_reason.to_json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
