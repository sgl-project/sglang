"""Boundary tests for the scheduler's waiting / running request timeouts.

Both paths are pure bookkeeping over timestamps -- no model, no GPU, no draft
worker -- so they are driven here directly instead of through a server. The
e2e side (503 reaching the client, server stays up) is covered by
scheduler/test_scheduler_control.py.

PD semantics: on the prefill side "waiting" spans the bootstrap queue and the
waiting queue, so the clock starts at arrival (bootstrap-queue entry) and the
per-rank verdicts are MIN-reduced across attn ranks (both queues feed
rank-aligned poll lists, so ranks must drop the same requests in the same
iteration). The decode side has no waiting abort at all: by the time a request
reaches its waiting queue the prefill work and KV transfer are already paid
for; transfer stalls are owned by SGLANG_DISAGGREGATION_WAITING_TIMEOUT and
scheduled requests by the running timeout.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeReq:
    """Must stay hashable: the waiting-timeout paths collect drops in a set."""

    def __init__(
        self,
        rid,
        wait_entry=0.0,
        forward_entry=0.0,
        bootstrap_entry=0.0,
        is_finished=False,
    ):
        self.rid = rid
        self.to_finish = None
        self._finished = is_finished
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=wait_entry,
            forward_entry_time=forward_entry,
            prefill_bootstrap_queue_entry_time=bootstrap_entry,
            trace_ctx=MagicMock(),
        )
        self.metadata_buffer_index = -1
        self.mamba_pool_idx = None
        self.disagg_kv_sender = MagicMock()

    def finished(self):
        return self._finished


def _req(
    rid: str,
    *,
    wait_entry: float = 0.0,
    forward_entry: float = 0.0,
    bootstrap_entry: float = 0.0,
    finished=False,
):
    return _FakeReq(rid, wait_entry, forward_entry, bootstrap_entry, finished)


def _scheduler(waiting_queue, mode=DisaggregationMode.NULL, bootstrap_queue=None):
    s = Scheduler.__new__(Scheduler)
    s.waiting_queue = waiting_queue
    s.enable_hicache_storage = False
    s.enable_hisparse = False
    s.disaggregation_mode = mode
    s.tree_cache = MagicMock()
    s.req_to_metadata_buffer_idx_allocator = MagicMock()
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    if mode == DisaggregationMode.PREFILL:
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=bootstrap_queue or [])
        s.attn_cp_cpu_group = MagicMock()
        s.attn_tp_cpu_group = MagicMock()
    return s


class TestWaitingTimeout(CustomTestCase):
    def test_drops_only_reqs_past_the_deadline(self):
        now = time.perf_counter()
        stale = _req("stale", wait_entry=now - 10)
        fresh = _req("fresh", wait_entry=now)
        s = _scheduler([stale, fresh])

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual([r.rid for r in s.waiting_queue], ["fresh"])
        self.assertEqual(s.ipc_channels.send_to_tokenizer.send_output.call_count, 1)

    def test_unset_entry_time_is_never_dropped(self):
        # 0 is the "not yet stamped" sentinel; the guard is `0 < entry_time`.
        s = _scheduler([_req("unstamped", wait_entry=0.0)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1e-9):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_disabled_timeout_is_a_no_op(self):
        s = _scheduler([_req("stale", wait_entry=time.perf_counter() - 100)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(-1):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


def _identity_reduce(values, **_kwargs):
    return values


class TestPrefillWaitingTimeout(CustomTestCase):
    def test_clock_starts_at_arrival_and_spans_both_queues(self):
        now = time.perf_counter()
        # A recent waiting-queue stamp must not shield a request that arrived
        # long ago: the prefill clock is bootstrap-queue entry, not
        # wait-queue entry.
        stale_waiting = _req("stale_w", wait_entry=now, bootstrap_entry=now - 10)
        stale_waiting.metadata_buffer_index = 3
        stale_boot = _req("stale_b", bootstrap_entry=now - 10)
        fresh_waiting = _req("fresh_w", bootstrap_entry=now)
        fresh_boot = _req("fresh_b", bootstrap_entry=now)
        unstamped = _req("unstamped", bootstrap_entry=0.0)
        s = _scheduler(
            [stale_waiting, fresh_waiting],
            DisaggregationMode.PREFILL,
            bootstrap_queue=[stale_boot, fresh_boot, unstamped],
        )

        with (
            envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0),
            patch(
                "sglang.srt.disaggregation.prefill.all_reduce_min_attn_cp_tp_group",
                side_effect=_identity_reduce,
            ),
        ):
            s._abort_on_prefill_waiting_timeout()

        self.assertEqual([r.rid for r in s.waiting_queue], ["fresh_w"])
        self.assertEqual(
            [r.rid for r in s.disagg_prefill_bootstrap_queue.queue],
            ["fresh_b", "unstamped"],
        )
        self.assertEqual(s.ipc_channels.send_to_tokenizer.send_output.call_count, 2)
        s.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        stale_waiting.disagg_kv_sender.abort.assert_called_once_with()
        stale_boot.disagg_kv_sender.abort.assert_called_once_with()

    def test_rank_consensus_gates_the_local_verdict(self):
        # Only the MIN-reduced verdict may drop a request; a rank-local stale
        # flag alone would misalign the rank-synchronized poll lists.
        stale = _req("stale", bootstrap_entry=time.perf_counter() - 10)
        s = _scheduler([stale], DisaggregationMode.PREFILL)

        with (
            envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0),
            patch(
                "sglang.srt.disaggregation.prefill.all_reduce_min_attn_cp_tp_group",
                return_value=[0],
            ) as reduce_flags,
        ):
            s._abort_on_prefill_waiting_timeout()

        self.assertEqual(reduce_flags.call_args.args[0], [1])
        self.assertEqual(s.waiting_queue, [stale])
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_disabled_timeout_is_a_no_op(self):
        stale = _req("stale", bootstrap_entry=time.perf_counter() - 100)
        s = _scheduler([], DisaggregationMode.PREFILL, bootstrap_queue=[stale])
        with (
            envs.SGLANG_REQ_WAITING_TIMEOUT.override(-1),
            patch(
                "sglang.srt.disaggregation.prefill.all_reduce_min_attn_cp_tp_group",
            ) as reduce_flags,
        ):
            s._abort_on_prefill_waiting_timeout()
        reduce_flags.assert_not_called()
        self.assertEqual(s.disagg_prefill_bootstrap_queue.queue, [stale])
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


class TestAbortWaitingRequest(CustomTestCase):
    """Client aborts of queued PD requests must release role-specific
    resources: leaked decode KV / hisparse state, or a prefill sender whose
    decode peer polls forever."""

    def test_decode_abort_releases_kv_and_hisparse(self):
        req = _req("r")
        s = _scheduler([req], DisaggregationMode.DECODE)
        s.enable_hisparse = True
        s.hisparse_coordinator = MagicMock()

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            s._abort_waiting_request(req)

        s.hisparse_coordinator.request_finished.assert_called_once_with(req)
        release.assert_called_once_with(req, s.tree_cache)

    def test_prefill_abort_releases_metadata_and_notifies_peer(self):
        req = _req("r")
        req.metadata_buffer_index = 5
        s = _scheduler([req], DisaggregationMode.PREFILL)

        s._abort_waiting_request(req)

        s.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(5)
        self.assertEqual(req.metadata_buffer_index, -1)
        req.disagg_kv_sender.abort.assert_called_once_with()


class TestPDTimeoutHooks(CustomTestCase):
    def test_prefill_planner_checks_waiting_timeout(self):
        s = Scheduler.__new__(Scheduler)
        s.process_pending_chunked_abort = MagicMock()
        s._abort_on_prefill_waiting_timeout = MagicMock(side_effect=RuntimeError)

        with self.assertRaises(RuntimeError):
            s.get_next_disagg_prefill_batch_to_run(MagicMock(), None)

        s._abort_on_prefill_waiting_timeout.assert_called_once_with()

    def test_decode_planner_checks_only_running_timeout(self):
        s = Scheduler.__new__(Scheduler)
        s._abort_on_waiting_timeout = MagicMock()
        s._abort_on_prefill_waiting_timeout = MagicMock()
        s._abort_on_running_timeout = MagicMock(side_effect=RuntimeError)
        running_batch = MagicMock()

        with self.assertRaises(RuntimeError):
            s.get_next_disagg_decode_batch_to_run(running_batch)

        s._abort_on_running_timeout.assert_called_once_with(running_batch)
        s._abort_on_waiting_timeout.assert_not_called()
        s._abort_on_prefill_waiting_timeout.assert_not_called()


class TestRunningTimeout(CustomTestCase):
    @staticmethod
    def _batch(reqs):
        return SimpleNamespace(reqs=reqs, is_empty=lambda: not reqs)

    def test_marks_only_stale_unfinished_reqs(self):
        now = time.perf_counter()
        stale = _req("stale", forward_entry=now - 10)
        fresh = _req("fresh", forward_entry=now)
        done = _req("done", forward_entry=now - 10, finished=True)
        s = _scheduler([])

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([stale, fresh, done]))

        self.assertIsNotNone(stale.to_finish)
        self.assertIsNone(fresh.to_finish)
        self.assertIsNone(done.to_finish, "a finished req must not be aborted")

    def test_unset_forward_entry_time_is_never_marked(self):
        s = _scheduler([])
        req = _req("unstamped", forward_entry=0.0)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1e-9):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)

    def test_empty_batch_and_disabled_timeout_are_no_ops(self):
        s = _scheduler([])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([]))
        req = _req("stale", forward_entry=time.perf_counter() - 100)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(-1):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)


if __name__ == "__main__":
    unittest.main()
