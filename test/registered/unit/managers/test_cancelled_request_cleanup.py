"""Cancellation must reach the scheduler even after tokenizer state is gone."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCancelledRequestCleanup(CustomTestCase):
    def test_abort_without_tokenizer_state_still_dispatches(self):
        manager = SimpleNamespace(
            rid_to_state={},
            enable_metrics=False,
            server_args=SimpleNamespace(tokenizer_worker_num=1),
            _dispatch_to_scheduler=Mock(),
        )
        TokenizerManager.abort_request(manager, "cancelled")
        self.assertEqual(
            manager._dispatch_to_scheduler.call_args.args[0].rid, "cancelled"
        )
        manager._dispatch_to_scheduler.reset_mock()
        TokenizerManager.abort_request(manager)
        manager._dispatch_to_scheduler.assert_not_called()
        TokenizerManager.abort_request(manager, abort_all=True)
        self.assertTrue(manager._dispatch_to_scheduler.call_args.args[0].abort_all)

    def test_discard_aborts_tracked_requests_before_removing_state(self):
        for single in (True, False):
            with self.subTest(single=single):
                states = {"a": object(), "b": object()}
                dispatched = []

                def dispatch(req):
                    self.assertIn(req.rid, states)
                    dispatched.append(req.rid)

                manager = SimpleNamespace(
                    rid_to_state=states, _dispatch_to_scheduler=dispatch
                )
                obj = SimpleNamespace(
                    is_single=single, rid="a" if single else ["a", "missing", "b"]
                )
                TokenizerManager._discard_pending_req_states(manager, obj)
                self.assertEqual(dispatched, ["a"] if single else ["a", "b"])
                self.assertEqual(set(states), {"b"} if single else set())

    def test_socket_failure_does_not_prevent_local_cleanup(self):
        manager = SimpleNamespace(
            rid_to_state={"a": object()},
            _dispatch_to_scheduler=Mock(side_effect=RuntimeError("socket closed")),
        )
        TokenizerManager._discard_pending_req_states(manager, SimpleNamespace(rid="a"))
        self.assertEqual(manager.rid_to_state, {})

    def test_deferred_abort_retries_after_request_leaves_chunked_slot(self):
        req = SimpleNamespace(rid="a", req_pool_idx=3, finished=lambda: False)
        scheduler = SimpleNamespace(
            _pending_chunked_abort_req=req, chunked_req=None, abort_request=Mock()
        )
        Scheduler.process_pending_chunked_abort(scheduler)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertEqual(scheduler.abort_request.call_args.args[0].rid, "a")

    def test_completed_deferred_abort_is_cleared_without_dispatch(self):
        for finished, pool_idx in ((True, 3), (False, None)):
            req = SimpleNamespace(
                rid="a", req_pool_idx=pool_idx, finished=lambda: finished
            )
            scheduler = SimpleNamespace(
                _pending_chunked_abort_req=req, chunked_req=None, abort_request=Mock()
            )
            Scheduler.process_pending_chunked_abort(scheduler)
            self.assertIsNone(scheduler._pending_chunked_abort_req)
            scheduler.abort_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
