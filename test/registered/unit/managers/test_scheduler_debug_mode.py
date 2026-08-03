"""Unit tests for the dev-only --debug-mode failed-batch handling.

Two layers:

1. ``SchedulerDebugFaultHandler.discard_batch`` -- the teardown primitive, built with
   narrow fakes (no Scheduler, no model, no GPU).
2. ``Scheduler._discard_failed_batch`` -- the orchestration around it: which scheduler
   fields it is allowed to touch, and the best-effort re-raise contract.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.debug_fault_handler import (
    ABORT_MESSAGE,
    SchedulerDebugFaultHandler,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReq:
    """Only the fields the teardown path reads or writes."""

    def __init__(self, rid: str, finished: bool = False):
        self.rid = rid
        self.finished_reason = "done" if finished else None
        self.to_finish = "staged"
        self.time_stats = MagicMock()
        self.return_logprob = False
        self.req_pool_idx = 0

    def finished(self) -> bool:
        return self.finished_reason is not None


def _fake_batch(rids, finished_rids=()):
    return MagicMock(reqs=[_FakeReq(r, finished=r in finished_rids) for r in rids])


def _make_handler(**overrides):
    kwargs = dict(
        tree_cache=MagicMock(),
        hisparse_coordinator=None,
        ipc_channels=MagicMock(),
        enable_hicache_storage=lambda: False,
    )
    kwargs.update(overrides)
    return SchedulerDebugFaultHandler(**kwargs), kwargs


def _notified_rids(ipc_channels) -> set:
    return {
        call.args[0].rid
        for call in ipc_channels.send_to_tokenizer.send_output.call_args_list
    }


class TestDebugFaultHandler(CustomTestCase):
    def setUp(self):
        super().setUp()
        # release_kv_cache needs the real pool objects and resolved config; the
        # contract here is that we delegate to it once per discarded request.
        patcher = patch(
            "sglang.srt.managers.scheduler_components.debug_fault_handler.release_kv_cache"
        )
        self.release_kv_cache = patcher.start()
        self.addCleanup(patcher.stop)

    def test_discard_batch_aborts_and_releases_each_unfinished_req(self):
        handler, kwargs = _make_handler()
        batch = _fake_batch(("r1", "r2"))

        aborted = handler.discard_batch(batch)

        self.assertEqual(sorted(aborted), ["r1", "r2"])
        # Each request ends up finished, which is what lets the next filter_batch
        # drop it from running_batch, and its KV is released -- the forward pass that
        # would normally do both is what failed.
        for req in batch.reqs:
            self.assertTrue(req.finished())
            self.assertIsNone(req.to_finish)
        self.assertEqual(
            {call.args[0].rid for call in self.release_kv_cache.call_args_list},
            {"r1", "r2"},
        )
        # An aborted request's partial KV must never be inserted into the radix tree.
        for call in self.release_kv_cache.call_args_list:
            self.assertFalse(call.kwargs["is_insert"])
        # The client is told once per request, with the debug-mode reason.
        self.assertEqual(_notified_rids(kwargs["ipc_channels"]), {"r1", "r2"})
        for call in kwargs["ipc_channels"].send_to_tokenizer.send_output.call_args_list:
            self.assertEqual(call.args[0].abort_message, ABORT_MESSAGE)

    def test_discard_batch_skips_already_finished_reqs(self):
        # A request that finished normally before the exception already had its final
        # output streamed and its KV released. Touching it again would send an abort
        # for a completed rid and double-free.
        handler, kwargs = _make_handler()
        batch = _fake_batch(("done", "live"), finished_rids=("done",))

        aborted = handler.discard_batch(batch)

        self.assertEqual(aborted, ["live"])
        self.assertEqual(_notified_rids(kwargs["ipc_channels"]), {"live"})
        self.assertEqual(
            [call.args[0].rid for call in self.release_kv_cache.call_args_list],
            ["live"],
        )

    def test_discard_batch_retracts_hisparse_before_abort(self):
        # HiSparse retraction is only valid while the request is unfinished, so it
        # must run before prepare_abort sets the finish reason.
        seen_finished = []
        coordinator = MagicMock()
        coordinator.retract_req.side_effect = lambda req: seen_finished.append(
            req.finished()
        )
        handler, _ = _make_handler(hisparse_coordinator=coordinator)

        handler.discard_batch(_fake_batch(("r1",)))

        self.assertEqual(seen_finished, [False])


def _make_scheduler_stub(*, batch, chunked_req=None, discard_raises=False) -> Scheduler:
    """A Scheduler carrying only what _discard_failed_batch reads or writes, built
    without the heavy __init__ (no model, no GPU)."""
    sched = Scheduler.__new__(Scheduler)
    sched.running_batch = MagicMock(reqs=[], batch_is_full=True)
    sched.last_batch = batch
    sched.cur_batch_for_debug = batch
    sched.chunked_req = chunked_req
    sched._pending_chunked_abort_req = chunked_req
    sched.waiting_queue = [_FakeReq("queued")]
    sched.debug_fault_handler = MagicMock()
    if discard_raises:
        sched.debug_fault_handler.discard_batch.side_effect = RuntimeError("teardown")
    else:
        sched.debug_fault_handler.discard_batch.return_value = ["r1"]
    return sched


class TestDiscardFailedBatchOrchestration(CustomTestCase):
    def test_discards_batch_and_clears_only_pointers_to_it(self):
        batch = _fake_batch(("r1",))
        sched = _make_scheduler_stub(batch=batch)

        sched._discard_failed_batch(batch, RuntimeError("boom"))  # must not raise

        sched.debug_fault_handler.discard_batch.assert_called_once_with(batch)
        # The pointers to the discarded batch are dropped...
        self.assertIsNone(sched.last_batch)
        self.assertIsNone(sched.cur_batch_for_debug)
        self.assertFalse(sched.running_batch.batch_is_full)
        # ...and nothing else is: the fault is scoped to the batch, so the waiting
        # queue keeps its requests and the memory pools are never wiped. Widening the
        # teardown to more scheduler state must fail this assertion.
        self.assertEqual([r.rid for r in sched.waiting_queue], ["queued"])

    def test_clears_chunked_req_only_when_it_was_in_the_failed_batch(self):
        chunked = _FakeReq("c1")
        batch = _fake_batch(("r1",))
        batch.reqs.append(chunked)
        sched = _make_scheduler_stub(batch=batch, chunked_req=chunked)

        sched._discard_failed_batch(batch, RuntimeError("boom"))

        self.assertIsNone(sched.chunked_req)
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_keeps_chunked_req_that_survived_the_failed_batch(self):
        chunked = _FakeReq("c1")
        batch = _fake_batch(("r1",))
        sched = _make_scheduler_stub(batch=batch, chunked_req=chunked)

        sched._discard_failed_batch(batch, RuntimeError("boom"))

        self.assertIs(sched.chunked_req, chunked)

    def test_reraises_original_exception_when_teardown_fails(self):
        # If the teardown fails the state is unknown, so the original exception must
        # propagate to the normal crash path rather than being swallowed.
        batch = _fake_batch(("r1",))
        sched = _make_scheduler_stub(batch=batch, discard_raises=True)
        original = RuntimeError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            sched._discard_failed_batch(batch, original)
        self.assertIs(ctx.exception, original)
        # State must not be half-updated when bailing out to the crash path.
        self.assertIs(sched.last_batch, batch)


if __name__ == "__main__":
    unittest.main(verbosity=3)
