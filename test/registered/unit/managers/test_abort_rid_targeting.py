"""An AbortReq must only abort the request it names.

``POST /abort_request {"rid": "job-1"}`` used to also abort ``job-10``,
``job-11``, ... because every scheduler queue matched abort targets with a bare
``req.rid.startswith(abort_rid)``. Caller-supplied rids are an ordinary way to
name requests (sequential ids, per-tenant prefixes), so the abort has to match
exactly -- plus the ``f"{rid}_{i}"`` sub-request ids the engine derives from a
parent rid for batch requests and parallel sampling.
"""

import unittest
from collections import deque
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

from sglang.srt.managers.io_struct import AbortReq, matches_abort_rid  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    def __init__(self, rid: str):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these two predicates.
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.to_finish = None
        self._finished = False
        self.finished_reason = None
        self.cache_request_handle = None
        self.weight_version_events = []
        self.output_ids = []

    def finished(self):
        return self._finished


def _make_scheduler(*, waiting_reqs=(), running_reqs=()) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.enable_continuous_input_polling = False
    sched.result_queue = deque()
    sched.chunked_req = None
    sched._pending_chunked_abort_req = None
    sched.waiting_queue = list(waiting_reqs)
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.running_batch = SimpleNamespace(reqs=list(running_reqs))
    sched.last_batch = None
    sched.ipc_channels = Mock()
    sched.beam_coordinator = Mock()
    sched._release_aborted_request = Mock()
    return sched


class TestAbortRidMatching(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_matches_exact_rid_and_derived_children_only(self):
        self.assertTrue(matches_abort_rid("job-1", "job-1"))
        # Batch requests / parallel sampling expand "job" to "job_0", "job_1".
        self.assertTrue(matches_abort_rid("job_0", "job"))
        self.assertFalse(matches_abort_rid("job-10", "job-1"))
        self.assertFalse(matches_abort_rid("job-1x", "job-1"))
        self.assertFalse(matches_abort_rid("job", "job_0"))

    def test_running_sibling_with_shared_prefix_survives(self):
        target, sibling = _FakeReq("job-1"), _FakeReq("job-10")
        sched = _make_scheduler(running_reqs=[target, sibling])

        sched.abort_request(AbortReq(rid="job-1"))

        self.assertIsNotNone(target.to_finish)
        self.assertIsNone(sibling.to_finish, "unrelated request was aborted")

    def test_running_children_are_aborted_by_parent_rid(self):
        reqs = [_FakeReq(f"job_{i}") for i in range(3)]
        sched = _make_scheduler(running_reqs=reqs)

        sched.abort_request(AbortReq(rid="job"))

        for req in reqs:
            self.assertIsNotNone(req.to_finish, req.rid)

    def test_waiting_sibling_with_shared_prefix_survives(self):
        target, sibling = _FakeReq("job-1"), _FakeReq("job-10")
        sched = _make_scheduler(waiting_reqs=[target, sibling])

        sched.abort_request(AbortReq(rid="job-1"))

        self.assertEqual(["job-10"], [req.rid for req in sched.waiting_queue])


if __name__ == "__main__":
    unittest.main(verbosity=2)
