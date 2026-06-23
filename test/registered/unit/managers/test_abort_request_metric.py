"""Unit tests for abort accounting in ``TokenizerManager.abort_request``.

``sglang:num_aborted_requests_total`` must only count aborts that target a
request still in flight. The ``create_abort_task`` client-disconnect safety net
fires ~2s after a streaming response finishes and calls ``abort_request`` for an
already-finished rid; in multi-tokenizer mode the request-queue early return is
skipped (aborts are forwarded unconditionally for cross-worker correctness), so
those late safety-net aborts must not be miscounted.
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tm(tokenizer_worker_num: int, rid_to_state: dict) -> TokenizerManager:
    """A TokenizerManager with only the fields abort_request touches, built
    via __new__ to bypass __init__ (mirrors test_tokenizer_manager_rid_cleanup)."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.tokenizer_worker_num = tokenizer_worker_num
    tm.rid_to_state = rid_to_state
    tm.enable_metrics = True
    tm.send_to_scheduler = MagicMock()
    tm.metrics_collector = MagicMock()
    return tm


class TestAbortRequestMetric(CustomTestCase):
    def test_inflight_abort_is_counted(self):
        tm = _make_tm(tokenizer_worker_num=2, rid_to_state={"r1": object()})
        tm.abort_request(rid="r1")
        tm.send_to_scheduler.send_pyobj.assert_called_once()
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_finished_request_not_counted_in_multi_tokenizer(self):
        # Multi-tokenizer: the abort is still forwarded to the scheduler, but a
        # finished rid (already gone from rid_to_state) must not be counted.
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={})
        tm.abort_request(rid="already_finished")
        tm.send_to_scheduler.send_pyobj.assert_called_once()
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_finished_request_short_circuits_single_tokenizer(self):
        # Single-tokenizer: a finished rid early-returns before forward or count.
        tm = _make_tm(tokenizer_worker_num=1, rid_to_state={})
        tm.abort_request(rid="already_finished")
        tm.send_to_scheduler.send_pyobj.assert_not_called()
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()

    def test_abort_all_is_counted(self):
        tm = _make_tm(tokenizer_worker_num=8, rid_to_state={})
        tm.abort_request(abort_all=True)
        tm.send_to_scheduler.send_pyobj.assert_called_once()
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_not_counted_when_metrics_disabled(self):
        # Forwarding still happens, but nothing is observed with metrics off.
        tm = _make_tm(tokenizer_worker_num=2, rid_to_state={"r1": object()})
        tm.enable_metrics = False
        tm.abort_request(rid="r1")
        tm.send_to_scheduler.send_pyobj.assert_called_once()
        tm.metrics_collector.observe_one_aborted_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
