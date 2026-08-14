"""Unit tests for srt/disaggregation/common/conn -- non-blocking prefill
parallel-info fetch, and the decode-side cache check that consumes it.

The fetch used to run inline on the decode scheduler's event loop, so an
unresponsive bootstrap server stalled the whole loop for the request timeout.
These tests pin the two properties that fix relies on: the scheduler-thread call
never waits for the network, and a config mismatch found on the fetch thread is
still fatal for the engine.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import concurrent.futures
import threading
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.test.test_utils import CustomTestCase

ADDR = "10.0.0.1:8998"

ROUTE_PAYLOAD = {
    "attn_tp_size": 1,
    "attn_cp_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "page_size": 16,
    "kv_cache_dtype": "auto",
    "follow_bootstrap_room": False,
}


def _response(status_code=200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = dict(ROUTE_PAYLOAD if payload is None else payload)
    resp.text = ""
    return resp


class TestNonBlockingParallelInfo(CustomTestCase):
    def setUp(self):
        self.mgr = self._make_manager()
        self.addCleanup(self.mgr._parallel_info_executor.shutdown, wait=True)

    def _make_manager(self):
        """Lightweight manager exposing only what the fetch path touches, in the
        style of test_register_to_bootstrap.py (CommonKVManager.__init__ needs
        zmq and a resolved model config)."""
        mgr = MagicMock(spec=CommonKVManager)
        for name in (
            "try_ensure_parallel_info",
            "raise_parallel_info_error",
            "has_parallel_info",
            "_publish_parallel_info",
            "_fetch_parallel_info",
            "_check_parallel_info",
        ):
            setattr(
                mgr, name, getattr(CommonKVManager, name).__get__(mgr, CommonKVManager)
            )

        mgr.prefill_info_table = {}
        mgr.connection_lock = threading.Lock()
        mgr._parallel_info_inflight = set()
        mgr._parallel_info_fatal = None
        # Single worker so that submitting a no-op is a FIFO barrier: see _drain.
        mgr._parallel_info_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="test-pd-parallel-info"
        )
        mgr.kv_args = MagicMock()
        mgr.kv_args.page_size = 16
        mgr.kv_cache_dtype_str = "auto"
        mgr.dcp_size = 1
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        return mgr

    def _drain(self):
        """Wait for queued fetches to finish without asserting on wall time. The
        executor has a single worker, so a no-op that completes proves every
        earlier task did too."""
        self.mgr._parallel_info_executor.submit(lambda: None).result(timeout=10)

    def test_cache_hit_returns_true_without_any_fetch(self):
        self.mgr.prefill_info_table[ADDR] = PrefillServerInfo(**ROUTE_PAYLOAD)
        with patch("sglang.srt.disaggregation.common.conn.requests.get") as mock_get:
            self.assertTrue(self.mgr.try_ensure_parallel_info(ADDR))
        mock_get.assert_not_called()

    def test_scheduler_thread_does_not_wait_for_the_fetch(self):
        """The core regression guard: the call returns while the HTTP request is
        still outstanding, so a hung bootstrap server cannot stall the loop."""
        gate = threading.Event()
        entered = threading.Event()

        def blocking_get(*args, **kwargs):
            entered.set()
            self.assertTrue(gate.wait(timeout=10), "fetch was never released")
            return _response()

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            side_effect=blocking_get,
        ):
            self.assertFalse(self.mgr.try_ensure_parallel_info(ADDR))
            # Returned with the fetch still in flight, nothing published yet.
            self.assertTrue(entered.wait(timeout=10))
            self.assertNotIn(ADDR, self.mgr.prefill_info_table)
            self.assertFalse(self.mgr.has_parallel_info(ADDR))

            gate.set()
            self._drain()

        self.assertTrue(self.mgr.has_parallel_info(ADDR))
        self.assertEqual(self.mgr.prefill_info_table[ADDR].page_size, 16)
        self.mgr._resolve_rank_mapping.assert_called_once()
        self.assertEqual(self.mgr._parallel_info_inflight, set())

    def test_repeated_calls_do_not_pile_up_fetches(self):
        gate = threading.Event()
        entered = threading.Event()

        def blocking_get(*args, **kwargs):
            entered.set()
            self.assertTrue(gate.wait(timeout=10), "fetch was never released")
            return _response()

        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            side_effect=blocking_get,
        ) as mock_get:
            self.assertFalse(self.mgr.try_ensure_parallel_info(ADDR))
            self.assertTrue(entered.wait(timeout=10))
            for _ in range(5):
                self.assertFalse(self.mgr.try_ensure_parallel_info(ADDR))
            gate.set()
            self._drain()

        self.assertEqual(mock_get.call_count, 1)

    def test_failed_fetch_is_retried_on_a_later_call(self):
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=_response(status_code=503),
        ) as mock_get:
            self.assertFalse(self.mgr.try_ensure_parallel_info(ADDR))
            self._drain()
            self.assertNotIn(ADDR, self.mgr.prefill_info_table)
            # In-flight marker cleared, so the next cycle starts a fresh fetch.
            self.assertEqual(self.mgr._parallel_info_inflight, set())

            self.assertFalse(self.mgr.try_ensure_parallel_info(ADDR))
            self._drain()

        self.assertEqual(mock_get.call_count, 2)
        self.assertIsNone(self.mgr._parallel_info_fatal)

    def test_config_mismatch_is_reraised_on_the_caller_thread(self):
        """A page-size mismatch used to raise directly on the scheduler thread
        and take the engine down; it must still do so from the fetch thread."""
        mismatched = dict(ROUTE_PAYLOAD, page_size=64)
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=_response(payload=mismatched),
        ):
            self.assertFalse(self.mgr.try_ensure_parallel_info(ADDR))
            self._drain()

            self.assertNotIn(ADDR, self.mgr.prefill_info_table)
            with self.assertRaises(RuntimeError) as ctx:
                self.mgr.try_ensure_parallel_info(ADDR)
        self.assertIn("Page size mismatch", str(ctx.exception))

    def test_result_landing_before_the_lock_is_not_refetched(self):
        """A fetch thread can publish between the unlocked cache check and the
        locked section below it; the addr must not be fetched a second time."""
        real_lock = self.mgr.connection_lock
        mgr = self.mgr

        class PublishOnEnter:
            """Stands in for a worker that publishes exactly at the moment the
            caller reaches the locked section."""

            def __enter__(self):
                mgr.prefill_info_table.setdefault(
                    ADDR, PrefillServerInfo(**ROUTE_PAYLOAD)
                )
                return real_lock.__enter__()

            def __exit__(self, *exc_info):
                return real_lock.__exit__(*exc_info)

        self.mgr.connection_lock = PublishOnEnter()
        try:
            with patch(
                "sglang.srt.disaggregation.common.conn.requests.get"
            ) as mock_get:
                self.assertTrue(self.mgr.try_ensure_parallel_info(ADDR))
        finally:
            self.mgr.connection_lock = real_lock

        mock_get.assert_not_called()
        self.assertEqual(self.mgr._parallel_info_inflight, set())


class TestEnsurePrefillInfoConsumesAsyncResult(CustomTestCase):
    """The retry budget must keep counting fetch attempts, not scheduling
    cycles, and a result that lands between two cycles has to be picked up on
    the very next cycle rather than after the retry interval expires."""

    def _make_queue(self):
        queue = MagicMock(spec=DecodePreallocQueue)
        for name in ("_ensure_prefill_info", "_clear_ensure_state"):
            setattr(
                queue,
                name,
                getattr(DecodePreallocQueue, name).__get__(queue, DecodePreallocQueue),
            )
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._ensure_retry_interval = 1.0
        queue._max_ensure_retries = 15
        queue.kv_manager = MagicMock()
        queue.kv_manager.has_parallel_info.return_value = False
        queue.kv_manager.parallel_info_fetch_in_flight.return_value = False
        queue.kv_manager.try_ensure_parallel_info.return_value = False
        return queue

    def test_landed_result_is_used_on_the_next_cycle(self):
        queue = self._make_queue()
        reqs = [MagicMock()]

        # Cycle 1: nothing cached yet, a fetch is started.
        ready, remaining = queue._ensure_prefill_info({ADDR: reqs})
        self.assertEqual(ready, {})
        self.assertEqual(remaining, reqs)
        self.assertEqual(queue._ensure_retry_count[ADDR], 1)

        # Cycle 2, well within the retry interval: the fetch has landed, so the
        # request must be admitted now instead of waiting out the interval.
        queue.kv_manager.has_parallel_info.return_value = True
        ready, remaining = queue._ensure_prefill_info({ADDR: reqs})
        self.assertEqual(ready, {ADDR: reqs})
        self.assertEqual(remaining, [])
        self.assertNotIn(ADDR, queue._ensure_retry_count)
        self.assertNotIn(ADDR, queue._ensure_last_attempt_time)

    def test_retry_interval_still_paces_new_attempts(self):
        queue = self._make_queue()
        reqs = [MagicMock()]

        queue._ensure_prefill_info({ADDR: reqs})
        queue._ensure_prefill_info({ADDR: reqs})

        # Second cycle is inside the interval: no second fetch attempt, and the
        # abort counter did not advance.
        self.assertEqual(queue.kv_manager.try_ensure_parallel_info.call_count, 1)
        self.assertEqual(queue._ensure_retry_count[ADDR], 1)

    def test_outstanding_fetch_does_not_spend_retry_attempts(self):
        """The budget is _max_ensure_retries fetch attempts, and a fetch can now
        outlive many scheduling cycles, so cycles spent waiting on one must not
        consume it -- otherwise a slow server is abandoned after
        _max_ensure_retries cycles instead of that many attempts."""
        queue = self._make_queue()
        queue._max_ensure_retries = 2
        reqs = [MagicMock()]

        # Cycle 1 submits the first attempt.
        queue._ensure_prefill_info({ADDR: reqs})
        self.assertEqual(queue._ensure_retry_count[ADDR], 1)

        # The fetch is still running: many cycles pass, none of them an attempt.
        # Each cycle is placed outside the retry interval, so only the in-flight
        # check can keep them from spending the budget.
        queue.kv_manager.parallel_info_fetch_in_flight.return_value = True
        for _ in range(5):
            queue._ensure_last_attempt_time[ADDR] -= queue._ensure_retry_interval
            ready, remaining = queue._ensure_prefill_info({ADDR: reqs})
            self.assertEqual(ready, {})
            self.assertEqual(remaining, reqs)
        self.assertEqual(queue._ensure_retry_count[ADDR], 1)
        self.assertEqual(queue.kv_manager.try_ensure_parallel_info.call_count, 1)
        reqs[0].kv_receiver.abort.assert_not_called()

        # The fetch failed and cleared its marker; the next cycle outside the
        # interval spends the second and final attempt -- without aborting yet,
        # because that attempt's result is not in.
        queue.kv_manager.parallel_info_fetch_in_flight.return_value = False
        queue._ensure_last_attempt_time[ADDR] -= queue._ensure_retry_interval
        queue._ensure_prefill_info({ADDR: reqs})
        self.assertEqual(queue.kv_manager.try_ensure_parallel_info.call_count, 2)
        self.assertEqual(queue._ensure_retry_count[ADDR], 2)
        reqs[0].kv_receiver.abort.assert_not_called()

        # That attempt is done and published nothing, so the budget is spent.
        queue._ensure_last_attempt_time[ADDR] -= queue._ensure_retry_interval
        queue._ensure_prefill_info({ADDR: reqs})
        self.assertEqual(queue.kv_manager.try_ensure_parallel_info.call_count, 2)
        reqs[0].kv_receiver.abort.assert_called_once()
        self.assertNotIn(ADDR, queue._ensure_retry_count)

    def test_result_landing_just_before_the_abort_is_honoured(self):
        """A fetch can publish between the cache check at the top of a cycle and
        the exhausted-budget branch at the bottom; those requests are fine and
        must not be aborted."""
        queue = self._make_queue()
        queue._max_ensure_retries = 1
        reqs = [MagicMock()]

        queue._ensure_prefill_info({ADDR: reqs})
        queue._ensure_last_attempt_time[ADDR] -= queue._ensure_retry_interval

        # False at the top of the cycle, True by the time the abort is decided.
        queue.kv_manager.has_parallel_info.side_effect = [False, True]
        ready, remaining = queue._ensure_prefill_info({ADDR: reqs})

        self.assertEqual(ready, {ADDR: reqs})
        self.assertEqual(remaining, [])
        reqs[0].kv_receiver.abort.assert_not_called()
        self.assertNotIn(ADDR, queue._ensure_retry_count)

    def test_fatal_fetch_error_surfaces_even_when_the_budget_is_spent(self):
        """A config mismatch is fatal for the engine, so it must not be masked
        by the abort path taking over on the same cycle."""
        queue = self._make_queue()
        queue._max_ensure_retries = 1
        reqs = [MagicMock()]

        queue._ensure_prefill_info({ADDR: reqs})
        queue._ensure_last_attempt_time[ADDR] -= queue._ensure_retry_interval

        queue.kv_manager.raise_parallel_info_error.side_effect = RuntimeError(
            "Page size mismatch"
        )
        with self.assertRaises(RuntimeError):
            queue._ensure_prefill_info({ADDR: reqs})
        reqs[0].kv_receiver.abort.assert_not_called()

    def test_last_attempt_is_not_abandoned_before_its_result(self):
        """The abort is decided before spending an attempt, not after, so a
        fetch that succeeds on the last allowed attempt still admits its
        requests instead of being discarded unseen."""
        queue = self._make_queue()
        queue._max_ensure_retries = 1
        reqs = [MagicMock()]

        # Cycle 1 spends the only allowed attempt; nothing is aborted yet.
        queue._ensure_prefill_info({ADDR: reqs})
        self.assertEqual(queue._ensure_retry_count[ADDR], 1)
        reqs[0].kv_receiver.abort.assert_not_called()

        # That fetch succeeds just before the next cycle.
        queue.kv_manager.has_parallel_info.return_value = True
        queue._ensure_last_attempt_time[ADDR] -= queue._ensure_retry_interval
        ready, remaining = queue._ensure_prefill_info({ADDR: reqs})

        self.assertEqual(ready, {ADDR: reqs})
        self.assertEqual(remaining, [])
        reqs[0].kv_receiver.abort.assert_not_called()


if __name__ == "__main__":
    unittest.main()
