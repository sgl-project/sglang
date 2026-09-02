"""Unit tests for abort accounting in ``TokenizerManager.abort_request``.

``sglang:num_aborted_requests_total`` must not count the delayed cleanup
abort issued after a streaming response has already finished. Explicit aborts
in multi-tokenizer mode are still forwarded and counted when the RID is absent
locally, because that worker may not own the request.
"""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The ipc name a multi-tokenizer worker stamps onto outgoing requests so the
# scheduler can route the response back to the originating worker.
WORKER_IPC = "ipc:///tmp/fake-tokenizer-worker"


def _require_abort(req: Optional[AbortReq]) -> AbortReq:
    if req is None:
        raise AssertionError("expected abort request to be dispatched")
    return req


def _make_tm(
    tokenizer_worker_num: int, rid_to_state: dict[str, object]
) -> TokenizerManager:
    """A TokenizerManager with only the fields abort_request touches, built
    via __new__ to bypass __init__ (mirrors test_tokenizer_manager_rid_cleanup).

    `tokenizer_ipc_name` follows the real __init__ contract: None in
    single-tokenizer mode, the worker's ipc name otherwise. _dispatch_to_scheduler
    is left real so the tests cover the http_worker_ipc stamping an abort needs to
    stay routable in multi-tokenizer mode; only sock_send is patched (see
    _capture_dispatch).
    """
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = SimpleNamespace(tokenizer_worker_num=tokenizer_worker_num)
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
        with (
            _capture_dispatch() as send,
            patch("sglang.srt.managers.tokenizer_manager.get_serving") as serving,
        ):
            serving.return_value.tokenizer_worker_num = (
                tm.server_args.tokenizer_worker_num
            )
            tm.abort_request(**kwargs)
        if not send.called:
            return None
        self.assertEqual(send.call_count, 1)
        _socket, req = send.call_args.args
        self.assertIsInstance(req, AbortReq)
        return req

    def test_inflight_abort_is_counted_multi_tokenizer(self):
        tm = _make_tm(tokenizer_worker_num=2, rid_to_state={"r1": object()})
        req = _require_abort(self._abort(tm, rid="r1"))
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
        req = _require_abort(self._abort(tm, rid="r1"))
        self.assertEqual(req.rid, "r1")
        # Single-tokenizer has no worker ipc to stamp.
        self.assertIsNone(req.http_worker_ipc)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_delayed_cleanup_for_finished_request_not_counted(self):
        # The delayed client-disconnect safety net is a known late no-op after
        # the response path removes the local state. It must still be forwarded
        # in multi-tokenizer mode, but must not increment the metric.
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={})
        obj = SimpleNamespace(is_single=True, rid="already_finished")
        background = tm.create_abort_task(obj)
        with (
            _capture_dispatch() as send,
            patch("sglang.srt.managers.tokenizer_manager.get_serving") as serving,
            patch(
                "sglang.srt.managers.tokenizer_manager.asyncio.sleep", new=AsyncMock()
            ),
        ):
            serving.return_value.tokenizer_worker_num = 8
            asyncio.run(background())
        self.assertEqual(send.call_count, 1)
        _socket, req = send.call_args.args
        self.assertEqual(req.rid, "already_finished")
        self.assertEqual(req.http_worker_ipc, WORKER_IPC)
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_delayed_cleanup_for_finished_batch_is_not_counted(self):
        # The batch cleanup path must apply the same late-no-op rule to every
        # request ID while preserving multi-tokenizer forwarding.
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={})
        obj = SimpleNamespace(is_single=False, rid=["finished-1", "finished-2"])
        background = tm.create_abort_task(obj)
        with (
            _capture_dispatch() as send,
            patch("sglang.srt.managers.tokenizer_manager.get_serving") as serving,
            patch(
                "sglang.srt.managers.tokenizer_manager.asyncio.sleep", new=AsyncMock()
            ),
        ):
            serving.return_value.tokenizer_worker_num = 8
            asyncio.run(background())
        self.assertEqual(send.call_count, 2)
        self.assertEqual(
            [call.args[1].rid for call in send.call_args_list],
            ["finished-1", "finished-2"],
        )
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_delayed_cleanup_for_inflight_request_is_counted(self):
        # The source marker only suppresses a delayed no-op. If the request is
        # still live when cleanup runs, it remains a real abort.
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={"r1": object()})
        self.assertIsNotNone(self._abort(tm, rid="r1", is_from_delayed_cleanup=True))
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_explicit_unknown_abort_is_counted_multi_tokenizer(self):
        # An explicit abort can arrive at a worker that does not own the RID.
        # Keep counting that dispatch because the local map is not authoritative
        # in multi-tokenizer mode.
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={})
        self.assertIsNotNone(self._abort(tm, rid="owned_by_another_worker"))
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_explicit_prefix_abort_is_counted(self):
        # The scheduler matches abort RIDs with startswith(), so a prefix can
        # target an in-flight local request without being an exact dictionary key.
        tm = _make_tm(tokenizer_worker_num=1, rid_to_state={"batch-0": object()})
        self.assertIsNotNone(self._abort(tm, rid="batch-"))
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

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
        req = _require_abort(self._abort(tm, abort_all=True))
        self.assertTrue(req.abort_all)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_abort_all_is_counted_single_tokenizer(self):
        # abort_all must survive the single-tokenizer early return even with an
        # empty rid_to_state -- pause_generation relies on it.
        tm = _make_tm(tokenizer_worker_num=1, rid_to_state={})
        req = _require_abort(self._abort(tm, abort_all=True))
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
