"""Unit tests for the dev-only fault-tolerant scheduler event loop recovery.

Covers Scheduler._all_inflight_reqs / _abort_all_and_reset /
_recover_from_iteration_error (see --debug-mode). These run the
recovery logic directly on a hand-built Scheduler stub, so no model, GPU, or
event loop is needed.
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid: str):
        self.rid = rid


def _fake_batch(rids):
    return MagicMock(reqs=[_FakeReq(r) for r in rids])


def _make_scheduler_stub(
    *, running_rids=(), last_rids=(), cur_rids=(), chunked_rid=None
) -> Scheduler:
    """A Scheduler carrying only the attributes the recovery path reads/writes,
    built without the heavy __init__ (no model / GPU)."""
    sched = Scheduler.__new__(Scheduler)

    sched.running_batch = _fake_batch(running_rids)
    sched.last_batch = _fake_batch(last_rids) if last_rids else None
    sched.cur_batch_for_debug = _fake_batch(cur_rids) if cur_rids else None
    sched.chunked_req = _FakeReq(chunked_rid) if chunked_rid else None
    sched.waiting_queue = [_FakeReq("w1")]
    sched._pending_chunked_abort_req = _FakeReq("stale")

    sched.abort_request = MagicMock()
    sched.ipc_channels = MagicMock()
    sched.tree_cache = MagicMock()
    sched.req_to_token_pool = MagicMock()
    sched.token_to_kv_pool_allocator = MagicMock()
    sched.grammar_manager = MagicMock()
    sched.draft_worker = None
    # Post-reset verification hook; individual tests override the return value.
    sched.is_fully_idle = MagicMock(return_value=True)
    return sched


class TestDevFaultToleranceRecovery(CustomTestCase):
    def test_all_inflight_reqs_dedups_by_rid(self):
        # r1 is in both running and last batches; the chunked req adds c1. A
        # req must be reported once, or _abort_all_and_reset double-notifies it.
        sched = _make_scheduler_stub(
            running_rids=("r1", "r2"), last_rids=("r1",), chunked_rid="c1"
        )
        rids = sorted(req.rid for req in sched._all_inflight_reqs())
        self.assertEqual(rids, ["c1", "r1", "r2"])

    def test_abort_all_and_reset_notifies_inflight_and_resets_state(self):
        sched = _make_scheduler_stub(running_rids=("r1", "r2"), chunked_rid="c1")

        sched._abort_all_and_reset()

        # Queued requests are handled via abort_request(abort_all=True).
        sched.abort_request.assert_called_once()
        (abort_arg,) = sched.abort_request.call_args.args
        self.assertIsInstance(abort_arg, AbortReq)
        self.assertTrue(abort_arg.abort_all)

        # Running/chunked requests are notified EXPLICITLY: abort_request only
        # marks them (relying on a later forward), so without this loop their
        # clients would hang forever. Guard that every in-flight rid is sent.
        notified = {
            call.args[0].rid
            for call in sched.ipc_channels.send_to_tokenizer.send_output.call_args_list
        }
        self.assertEqual(notified, {"r1", "r2", "c1"})

        # All memory pools are wiped so no KV is leaked across the reset.
        sched.tree_cache.reset.assert_called_once()
        sched.req_to_token_pool.clear.assert_called_once()
        sched.token_to_kv_pool_allocator.clear.assert_called_once()
        sched.grammar_manager.clear.assert_called_once()

        # Loop-carried state returns to the empty __init__ values.
        self.assertEqual(sched.waiting_queue, [])
        self.assertIsInstance(sched.running_batch, ScheduleBatch)
        self.assertEqual(len(sched.running_batch.reqs), 0)
        self.assertIsNone(sched.last_batch)
        self.assertIsNone(sched.cur_batch_for_debug)
        self.assertIsNone(sched.chunked_req)
        # Dropping this marker prevents next iteration's
        # process_pending_chunked_abort from touching a discarded req.
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_recover_drives_reset_and_verification_on_success(self):
        # Guards the success-path orchestration of _recover: it must drive the
        # reset (abort_request) AND the post-reset verification (is_fully_idle),
        # then return without raising. This is the only case exercising the happy
        # path, so it catches a _recover that wrongly raises even on success.
        sched = _make_scheduler_stub(running_rids=("r1",))

        sched._recover_from_iteration_error(RuntimeError("boom"))  # must not raise

        sched.abort_request.assert_called_once()
        sched.is_fully_idle.assert_called_once()

    def test_recover_reraises_when_reset_leaves_non_idle_state(self):
        # Post-reset verification: if the scheduler is not fully idle after the
        # reset, the recovery is unrecoverable and must re-raise the ORIGINAL
        # exception (crash path) rather than resuming on stale state.
        sched = _make_scheduler_stub(running_rids=("r1",))
        sched.is_fully_idle = MagicMock(return_value=False)
        original = RuntimeError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            sched._recover_from_iteration_error(original)
        self.assertIs(ctx.exception, original)

    def test_recover_reraises_original_when_reset_fails(self):
        # Best-effort contract: if the reset itself fails the state is
        # unrecoverable, so the ORIGINAL exception must propagate (falling back
        # to the normal crash path) rather than being swallowed.
        sched = _make_scheduler_stub(running_rids=("r1",))
        sched.abort_request.side_effect = RuntimeError("cleanup failed")
        original = RuntimeError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            sched._recover_from_iteration_error(original)
        self.assertIs(ctx.exception, original)


if __name__ == "__main__":
    unittest.main(verbosity=3)
