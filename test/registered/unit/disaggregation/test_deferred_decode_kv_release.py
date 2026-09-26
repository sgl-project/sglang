"""Unit tests for the deferred decode-side KV release mechanism.

When a decode request is aborted while its prefill->decode KV transfer may still
be in flight, the decode side holds its KV pages / req-slot instead of freeing
them immediately (which could let the still-in-flight write land on pages already
reused by another request). The pages are released once every prefill rank acks
that its transfer drained (CommonKVManager.is_abort_release_safe). Device
destinations may also release on timeout; host destinations require the ack.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.base.conn import BaseKVManager
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_manager():
    """A bare CommonKVManager carrying only the deferred-ack state the helpers
    touch (avoids the heavy real __init__)."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    return mgr


class TestAbortAckAggregation(CustomTestCase):
    def test_release_safe_only_after_all_required_ranks_ack(self):
        mgr = _make_manager()
        room = 100
        mgr.register_deferred_abort_room(room)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_duplicate_rank_ack_does_not_over_count(self):
        mgr = _make_manager()
        room = 101
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.note_abort_ack(room, 0)  # same rank twice
        # Two acks arrived but from one rank: not safe for a 2-rank prefill.
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

    def test_single_rank_fast_path(self):
        mgr = _make_manager()
        room = 102
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=1))

    def test_clear_deferred_abort_state(self):
        mgr = _make_manager()
        room = 103
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.clear_deferred_abort_state(room)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_ack_before_register_is_dropped(self):
        # An ack for a room that isn't actively held must not be recorded (it
        # would otherwise pollute a later request reusing the same room).
        mgr = _make_manager()
        room = 104
        mgr.note_abort_ack(room, 0)  # no register yet
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_late_ack_after_release_does_not_pollute_reused_room(self):
        # Regression for bootstrap_room reuse: req A (room R) releases, then a
        # late ack from A arrives, then req B reuses room R. B must start from a
        # clean slate and not inherit A's ack (which would release B early while
        # its transfer is still in flight -> KV corruption).
        mgr = _make_manager()
        room = 105

        # Req A: held, one of two ranks acks, then released (e.g. timed out).
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.clear_deferred_abort_state(room)

        # Late ack from A's other rank arrives after release -> dropped.
        mgr.note_abort_ack(room, 1)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)

        # Req B reuses room R.
        mgr.register_deferred_abort_room(room)
        # Only B's rank-0 has acked so far; a 2-rank prefill is NOT safe yet.
        mgr.note_abort_ack(room, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))
        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_register_resets_stale_acks(self):
        mgr = _make_manager()
        room = 106
        mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0)
        mgr.note_abort_ack(room, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))
        # Re-registering (a later reuse) wipes the prior acks.
        mgr.register_deferred_abort_room(room)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_queue(timeout=30.0, enable_metrics=False):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q._deferred_releases = []
    q.deferred_kv_release_timeout = timeout
    q.enable_host_receive = False
    q.enable_staging = False
    q.staging_handler = None
    q.tree_cache = object()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    q.scheduler = SimpleNamespace(
        metrics_reporter=SimpleNamespace(enable_metrics=enable_metrics),
        metrics_collector=SimpleNamespace(
            observe_decode_deferred_kv_release=MagicMock()
        ),
    )
    return q


def _make_decode_req(room, idx, mgr, n_prefill_ranks=1):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        # One entry per prefill rank the decode notified of the abort; its length
        # is the required drain-ack count (see DecodeTransferQueue._defer_release).
        bootstrap_infos=[{"rank": r} for r in range(n_prefill_ranks)],
        clear=lambda: None,
    )
    return SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room),
        kv_receiver=receiver,
        metadata_buffer_index=idx,
    )


class TestResolveDeferredReleases(CustomTestCase):
    def test_host_release_requires_drain_on_every_rank_even_after_timeout(self):
        mgr = _make_manager()
        q = _make_queue(timeout=-1)
        q.enable_host_receive = True
        q.gloo_group = object()
        entries = [_make_decode_req(room, room, mgr) for room in (1, 2)]
        for entry in entries:
            entry.host_staged = True
            mgr.register_deferred_abort_room(entry.req.bootstrap_room)
            q._defer_release(entry)
        with (
            patch.object(decode_mod, "discard_kv_cache_backup") as discard,
            patch.object(decode_mod, "release_kv_cache") as device_release,
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as reduce,
        ):
            q.resolve_deferred_releases()
            discard.assert_not_called()
            mgr.note_abort_ack(2, 0)
            reduce.side_effect = lambda ready, **_: ready.zero_()
            q.resolve_deferred_releases()
            discard.assert_not_called()
            reduce.side_effect = None
            q.resolve_deferred_releases()
            discard.assert_called_once_with(entries[1].req, q.tree_cache, "host_pool")
            self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [2])
            mgr.note_abort_ack(1, 0)
            q.resolve_deferred_releases()
            self.assertEqual(discard.call_count, 2)
            self.assertEqual(q._deferred_releases, [])
            device_release.assert_not_called()

    def test_noop_when_nothing_deferred(self):
        q = _make_queue()
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()

    def test_holds_until_drained_then_releases(self):
        mgr = _make_manager()
        room, idx = 200, 7
        q = _make_queue(enable_metrics=True)
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=2)
        # In production the room is armed in abort_request when the ABORT is
        # sent, before the scheduler defers here.
        mgr.register_deferred_abort_room(room)

        with patch.object(
            decode_mod.time, "monotonic", side_effect=[10.0, 11.0, 12.0, 13.0]
        ):
            q._defer_release(dreq)
            with patch.object(decode_mod, "release_kv_cache") as rel:
                # Not yet acked -> held, not released.
                q.resolve_deferred_releases()
                rel.assert_not_called()
                self.assertEqual(len(q._deferred_releases), 1)
                q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_not_called()

                # One of two ranks acked -> still held.
                mgr.note_abort_ack(room, 0)
                q.resolve_deferred_releases()
                rel.assert_not_called()
                self.assertEqual(len(q._deferred_releases), 1)

                # Both ranks acked -> released exactly once.
                mgr.note_abort_ack(room, 1)
                q.resolve_deferred_releases()
                rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        # Held state fully cleaned up.
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.num_pending_deferred_releases(), 0)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertEqual(q.metadata_buffers.bootstrap_room[idx], 0)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertIsNone(dreq.kv_receiver)
        q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_called_once_with(
            duration_seconds=3.0,
            outcome="drained",
        )

    def test_releases_on_timeout_without_ack(self):
        mgr = _make_manager()
        room, idx = 300, 3
        q = _make_queue(timeout=30.0, enable_metrics=True)
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=1)
        # Force an already-expired deadline (no ack will ever arrive).
        q._deferred_releases.append((dreq, 10.0, float("-inf"), idx, 1))

        with (
            patch.object(decode_mod.time, "monotonic", return_value=42.0),
            patch.object(decode_mod, "release_kv_cache") as rel,
        ):
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertIsNone(dreq.kv_receiver)
        q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_called_once_with(
            duration_seconds=32.0,
            outcome="timeout",
        )

    def test_metrics_disabled_does_not_observe_release(self):
        mgr = _make_manager()
        room, idx = 301, 4
        q = _make_queue(enable_metrics=False)
        dreq = _make_decode_req(room, idx, mgr)
        q._deferred_releases.append((dreq, 10.0, float("-inf"), idx, 1))

        with (
            patch.object(decode_mod.time, "monotonic", return_value=20.0),
            patch.object(decode_mod, "release_kv_cache") as rel,
        ):
            q.resolve_deferred_releases()

        rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)
        self.assertEqual(q.num_pending_deferred_releases(), 0)
        q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_not_called()

    def test_failed_release_is_isolated_and_not_retried(self):
        # A raising _do_release must drop the entry (no double-free on retry) and
        # not brick resolve for the remaining entries or subsequent calls.
        mgr = _make_manager()
        q = _make_queue(enable_metrics=True)
        good = _make_decode_req(700, 1, mgr)
        bad = _make_decode_req(701, 2, mgr)
        # Both already past deadline -> both selected for release.
        q._deferred_releases.append((bad, 10.0, float("-inf"), 2, 1))
        q._deferred_releases.append((good, 10.0, float("-inf"), 1, 1))

        calls = []

        def fake_release(req, tree_cache, is_insert):
            calls.append(req)
            if req is bad.req:
                raise RuntimeError("boom")

        with (
            patch.object(decode_mod.time, "monotonic", return_value=40.0),
            patch.object(decode_mod, "release_kv_cache", side_effect=fake_release),
        ):
            q.resolve_deferred_releases()  # must not raise
            # The good one still released despite the bad one throwing.
            self.assertIn(good.req, calls)
            # Nothing left held, and a second call is a clean no-op (no retry).
            self.assertEqual(q._deferred_releases, [])
            q.resolve_deferred_releases()
        q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_has_calls(
            [
                call(duration_seconds=30.0, outcome="error"),
                call(duration_seconds=30.0, outcome="timeout"),
            ]
        )

    def test_defer_release_records_deadline_and_idx(self):
        mgr = _make_manager()
        q = _make_queue(timeout=12.5)
        dreq = _make_decode_req(room=400, idx=9, mgr=mgr)
        with patch.object(decode_mod.time, "monotonic", return_value=5.0):
            q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)
        held_req, start_time, deadline, held_idx, required = q._deferred_releases[0]
        self.assertIs(held_req, dreq)
        self.assertEqual(start_time, 5.0)
        self.assertEqual(deadline, 17.5)
        self.assertEqual(held_idx, 9)
        self.assertEqual(required, 1)


class TestBackendOptIn(CustomTestCase):
    """Without a prefill ack, every hold waits out the full release timeout."""

    def test_enabled_by_default(self):
        self.assertTrue(envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE.get())

    def test_backends_that_ack_opt_in(self):
        # Ascend inherits Mooncake's threads, so it opts in too.
        for cls in (MooncakeKVManager, NixlKVManager):
            with self.subTest(backend=cls.__name__):
                self.assertTrue(cls.supports_deferred_decode_kv_release)

    def test_backends_without_a_drain_ack_stay_opted_out(self):
        # Inheriting CommonKVManager is not enough: mori marks the room Failed
        # without acking.
        self.assertFalse(BaseKVManager.supports_deferred_decode_kv_release)
        self.assertFalse(CommonKVManager.supports_deferred_decode_kv_release)


if __name__ == "__main__":
    unittest.main()
