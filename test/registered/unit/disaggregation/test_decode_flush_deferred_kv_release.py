"""flush_cache / PD role switch must wait for held deferred decode KV releases.

A decode request aborted (or failed) mid-transfer keeps its KV pages out of the
allocator until prefill acks the drain or the hold times out. is_fully_idle()
counts those holds as idle, so a flush in that window reset the allocator under
them and the held release then freed the same pages a second time (available >
total, `usage=-0.0006 is not valid` in pool_stats_observer).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_scheduler(held: int, mode=DisaggregationMode.DECODE):
    """Just the state flush_cache touches; the transfer queue is real so the
    hold count comes from its own _deferred_releases."""
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q._deferred_releases = [(MagicMock(), 0.0, 30.0, i, 1) for i in range(held)]
    s = SimpleNamespace(
        disaggregation_mode=mode,
        disagg_decode_transfer_queue=q,
        _engine_paused=False,
        is_fully_idle=lambda ignore_waiting=False: True,
        waiting_queue=[],
        running_batch=SimpleNamespace(reqs=[]),
        tree_cache=MagicMock(),
        req_to_token_pool=MagicMock(),
        token_to_kv_pool_allocator=MagicMock(),
        grammar_manager=MagicMock(),
        metrics_reporter=MagicMock(),
        draft_worker=None,
    )
    s.has_pending_deferred_kv_releases = lambda: (
        Scheduler.has_pending_deferred_kv_releases(s)
    )
    s.is_idle_for_pool_reset = lambda ignore_waiting=False: (
        Scheduler.is_idle_for_pool_reset(s, ignore_waiting=ignore_waiting)
    )
    return s


class TestPoolResetWaitsForDeferredReleases(CustomTestCase):
    def test_flush_refused_while_release_held(self):
        s = _make_scheduler(held=2)
        self.assertFalse(Scheduler.flush_cache(s, empty_cache=False))
        s.token_to_kv_pool_allocator.clear.assert_not_called()
        s.req_to_token_pool.clear.assert_not_called()

    def test_flush_runs_once_holds_resolve(self):
        s = _make_scheduler(held=0)
        self.assertTrue(Scheduler.flush_cache(s, empty_cache=False))
        s.token_to_kv_pool_allocator.clear.assert_called_once()

    def test_non_decode_role_ignores_transfer_queue(self):
        s = _make_scheduler(held=0, mode=DisaggregationMode.PREFILL)
        s.disagg_decode_transfer_queue = None
        self.assertTrue(Scheduler.flush_cache(s, empty_cache=False))

    def test_timed_flush_waits_for_held_release(self):
        s = _make_scheduler(held=1)
        sent = []
        wrapper = SchedulerFlushWrapper(
            flush_cache=lambda: Scheduler.flush_cache(s, empty_cache=False),
            is_fully_idle=s.is_idle_for_pool_reset,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(
                    send_output=lambda out, req: sent.append(out)
                )
            ),
        )
        self.assertIsNone(wrapper.handle(FlushCacheReqInput(timeout_s=60.0)))
        wrapper.check_pending()
        self.assertEqual(sent, [])
        s.token_to_kv_pool_allocator.clear.assert_not_called()

        s.disagg_decode_transfer_queue._deferred_releases.clear()
        wrapper.check_pending()
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].success)
        s.token_to_kv_pool_allocator.clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
