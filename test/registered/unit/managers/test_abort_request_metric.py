"""Unit tests for abort accounting in ``TokenizerManager.abort_request``.

``sglang:num_aborted_requests_total`` must only count aborts that target a
request still in flight. The ``create_abort_task`` client-disconnect safety net
fires ~2s after a streaming response finishes and calls ``abort_request`` for an
already-finished rid; in multi-tokenizer mode the request-queue early return is
skipped (aborts are forwarded unconditionally for cross-worker correctness), so
those late safety-net aborts must not be miscounted.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.managers.io_struct import AbortReq
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The ipc name a multi-tokenizer worker stamps onto outgoing requests so the
# scheduler can route the response back to the originating worker.
WORKER_IPC = "ipc:///tmp/fake-tokenizer-worker"


def _make_tm(tokenizer_worker_num: int, rid_to_state: dict) -> TokenizerManager:
    """A TokenizerManager with only the fields abort_request touches, built
    via __new__ to bypass __init__ (mirrors test_tokenizer_manager_rid_cleanup).

    `tokenizer_ipc_name` follows the real __init__ contract: None in
    single-tokenizer mode, the worker's ipc name otherwise. _dispatch_to_scheduler
    is left real so the tests cover the http_worker_ipc stamping an abort needs to
    stay routable in multi-tokenizer mode; only sock_send is patched (see
    _capture_dispatch).
    """
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.tokenizer_worker_num = tokenizer_worker_num
    tm.rid_to_state = rid_to_state
    tm.enable_metrics = True
    tm.tokenizer_ipc_name = None if tokenizer_worker_num == 1 else WORKER_IPC
    tm.send_to_scheduler = MagicMock()
    tm.metrics_collector = MagicMock()
    return tm


def _capture_dispatch():
    """Patch the module-level sock_send that _dispatch_to_scheduler calls.

    Asserting on sock_send rather than on send_to_scheduler.send_pyobj keeps the
    assertions independent of SGLANG_USE_PICKLE_IPC, which decides whether the
    socket is driven via send_pyobj or a msgpack send.
    """
    return patch("sglang.srt.managers.tokenizer_manager.sock_send")


class TestAbortRequestMetric(CustomTestCase):
    def _abort(self, tm, **kwargs):
        """Run abort_request and return the AbortReq handed to the socket, or None."""
        with _capture_dispatch() as send:
            tm.abort_request(**kwargs)
        if not send.called:
            return None
        self.assertEqual(send.call_count, 1)
        _socket, req = send.call_args.args
        self.assertIsInstance(req, AbortReq)
        return req

    def test_inflight_abort_is_counted_multi_tokenizer(self):
        tm = _make_tm(tokenizer_worker_num=2, rid_to_state={"r1": object()})
        req = self._abort(tm, rid="r1")
        self.assertIsNotNone(req)
        self.assertEqual(req.rid, "r1")
        self.assertFalse(req.abort_all)
        # Multi-tokenizer aborts must carry the originating worker's ipc name or
        # the scheduler cannot route the response back and the worker hangs.
        self.assertEqual(req.http_worker_ipc, WORKER_IPC)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once_with(
            tm.metrics_collector.labels
        )

    def test_inflight_abort_is_counted_single_tokenizer(self):
        # The default configuration and the dominant production path: a live rid
        # must still be forwarded and counted.
        tm = _make_tm(tokenizer_worker_num=1, rid_to_state={"r1": object()})
        req = self._abort(tm, rid="r1")
        self.assertIsNotNone(req)
        self.assertEqual(req.rid, "r1")
        # Single-tokenizer has no worker ipc to stamp.
        self.assertIsNone(req.http_worker_ipc)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_finished_request_not_counted_in_multi_tokenizer(self):
        # Multi-tokenizer: the abort is still forwarded to the scheduler, but a
        # finished rid (already gone from rid_to_state) must not be counted.
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={})
        self.assertIsNotNone(self._abort(tm, rid="already_finished"))
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_finished_request_short_circuits_single_tokenizer(self):
        # Single-tokenizer: a finished rid early-returns before forward or count.
        tm = _make_tm(tokenizer_worker_num=1, rid_to_state={})
        self.assertIsNone(self._abort(tm, rid="already_finished"))
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_empty_rid_is_ignored(self):
        # An empty rid would startswith-match every request on the scheduler, so
        # it must never be forwarded unless abort_all was asked for explicitly.
        tm = _make_tm(tokenizer_worker_num=2, rid_to_state={"r1": object()})
        self.assertIsNone(self._abort(tm, rid=""))
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_abort_all_is_counted(self):
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={})
        req = self._abort(tm, abort_all=True)
        self.assertIsNotNone(req)
        self.assertTrue(req.abort_all)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_abort_all_is_counted_single_tokenizer(self):
        # abort_all must survive the single-tokenizer early return even with an
        # empty rid_to_state -- pause_generation relies on it.
        tm = _make_tm(tokenizer_worker_num=1, rid_to_state={})
        req = self._abort(tm, abort_all=True)
        self.assertIsNotNone(req)
        self.assertTrue(req.abort_all)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_not_counted_when_metrics_disabled(self):
        # Forwarding still happens, but nothing is observed with metrics off.
        tm = _make_tm(tokenizer_worker_num=2, rid_to_state={"r1": object()})
        tm.enable_metrics = False
        self.assertIsNotNone(self._abort(tm, rid="r1"))
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
