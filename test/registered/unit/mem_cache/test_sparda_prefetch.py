"""Registered CPU tests for the one-layer SparDA prefetch contract."""

from __future__ import annotations

import unittest

import torch

from sglang.srt.mem_cache.l2_transfer import L2Transfer, TransferCompletion
from sglang.srt.mem_cache.sparda_prefetch import (
    CallbackPageLease,
    CallbackPrefetchResolver,
    PrefetchTicketState,
    ResolvedPrefetch,
    SparDAKVPrefetcher,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _FlakyEvent(_Event):
    def __init__(self):
        super().__init__()
        self.failures = 1

    def synchronize(self):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("event not ready")
        super().synchronize()


class _TransferEngine:
    def __init__(self):
        self.calls = []
        self.finish_event = _Event()

    def submit_host_to_device(self, transfers, *, layer_num, start_event=None):
        self.calls.append((transfers, layer_num, start_event))
        return TransferCompletion(None, self.finish_event, False)


class _SynchronousTransferEngine(_TransferEngine):
    def submit_host_to_device(self, transfers, *, layer_num, start_event=None):
        self.calls.append((transfers, layer_num, start_event))
        return TransferCompletion(None, None, False)


class _Lease:
    def __init__(self, event):
        self.event = event
        self.releases = 0

    def release(self):
        assert self.event.synchronized, "lease released before H2D completion"
        self.releases += 1


def _resolved(engine, lease):
    return ResolvedPrefetch(
        transfers=(object(),),
        layer_num=1,
        lease=lease,
    )


class TestSparDAPrefetcher(CustomTestCase):
    def test_consume_waits_before_releasing_lease(self):
        engine = _TransferEngine()
        lease = _Lease(engine.finish_event)
        resolver = CallbackPrefetchResolver(lambda *_args: _resolved(engine, lease))
        prefetcher = SparDAKVPrefetcher(engine, resolver)

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [3, 7])
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket.state, PrefetchTicketState.SUBMITTED)
        self.assertTrue(prefetcher.wait_for_layer("request-a", 0, 1))
        self.assertTrue(engine.finish_event.synchronized)
        self.assertTrue(prefetcher.consume_for_layer("request-a", 0, 1))
        self.assertEqual(lease.releases, 1)
        self.assertEqual(prefetcher.active_tickets(), ())

    def test_cancel_synchronizes_before_page_reuse(self):
        engine = _TransferEngine()
        lease = _Lease(engine.finish_event)
        prefetcher = SparDAKVPrefetcher(
            engine,
            CallbackPrefetchResolver(lambda *_args: _resolved(engine, lease)),
        )

        ticket = prefetcher.prefetch_forecast("request-a", 1, 2, [4])
        prefetcher.cancel(ticket)
        self.assertEqual(ticket.state, PrefetchTicketState.CANCELLED)
        self.assertTrue(engine.finish_event.synchronized)
        self.assertEqual(lease.releases, 1)
        prefetcher.cancel(ticket)
        prefetcher.release(ticket)
        self.assertEqual(lease.releases, 1)

    def test_failed_event_sync_keeps_ticket_reachable_for_retry(self):
        engine = _TransferEngine()
        engine.finish_event = _FlakyEvent()
        lease = _Lease(engine.finish_event)
        prefetcher = SparDAKVPrefetcher(
            engine,
            CallbackPrefetchResolver(lambda *_args: _resolved(engine, lease)),
        )

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])
        prefetcher.cancel(ticket)
        self.assertEqual(lease.releases, 0)
        self.assertFalse(ticket.done)
        self.assertEqual(prefetcher.active_tickets(), (ticket,))

        prefetcher.cancel(ticket)
        self.assertTrue(engine.finish_event.synchronized)
        self.assertEqual(lease.releases, 1)
        self.assertTrue(ticket.done)
        self.assertEqual(prefetcher.active_tickets(), ())

    def test_unknown_completion_event_fails_closed(self):
        engine = _TransferEngine()
        engine.finish_event = object()
        lease = CallbackPageLease(lambda: None)
        prefetcher = SparDAKVPrefetcher(
            engine,
            CallbackPrefetchResolver(lambda *_args: _resolved(engine, lease)),
        )

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])
        with self.assertRaisesRegex(RuntimeError, "completion event"):
            prefetcher.wait(ticket)
        self.assertFalse(ticket.completion_synchronized)
        self.assertIs(ticket.lease, lease)

    def test_synchronous_completion_without_event_releases_lease(self):
        engine = _SynchronousTransferEngine()
        releases = []
        lease = CallbackPageLease(lambda: releases.append(True))
        prefetcher = SparDAKVPrefetcher(
            engine,
            CallbackPrefetchResolver(lambda *_args: _resolved(engine, lease)),
        )

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])
        self.assertTrue(prefetcher.consume(ticket))
        self.assertEqual(releases, [True])

    def test_generation_invalidation_is_request_local(self):
        engine = _TransferEngine()
        leases = []

        def resolve(*_args):
            lease = _Lease(engine.finish_event)
            leases.append(lease)
            return _resolved(engine, lease)

        prefetcher = SparDAKVPrefetcher(engine, CallbackPrefetchResolver(resolve))
        old = prefetcher.prefetch_forecast("request-a", 0, 1, [1])
        prefetcher.prefetch_forecast("request-b", 0, 1, [1])
        prefetcher.invalidate_generation("request-a", 1)

        self.assertEqual(old.state, PrefetchTicketState.CANCELLED)
        self.assertEqual(leases[0].releases, 1)
        self.assertEqual(len(prefetcher.active_tickets()), 1)
        self.assertEqual(prefetcher.active_tickets()[0].request_id, "request-b")

    def test_cleanup_rejects_a_late_resolver_result(self):
        engine = _TransferEngine()
        releases = []
        prefetcher = None

        def resolve(*_args):
            prefetcher.cleanup_request("request-a")
            return ResolvedPrefetch(
                transfers=(object(),),
                layer_num=1,
                lease=CallbackPageLease(lambda: releases.append(True)),
            )

        prefetcher = SparDAKVPrefetcher(engine, CallbackPrefetchResolver(resolve))
        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])

        self.assertEqual(ticket.state, PrefetchTicketState.CANCELLED)
        self.assertEqual(releases, [True])
        self.assertEqual(prefetcher.active_tickets(), ())
        self.assertEqual(prefetcher.begin_request("request-a"), 1)

    def test_forecast_query_and_missing_prediction_fallback(self):
        engine = _TransferEngine()
        observed = []

        class Resolver:
            def predict(self, request_id, generation, layer_id, query, context):
                observed.append((request_id, generation, layer_id, query, context))
                return [2]

            def resolve(self, *_args):
                return None

        prefetcher = SparDAKVPrefetcher(engine, Resolver())
        self.assertIsNone(
            prefetcher.prefetch_forecast_query(
                "request-a", 0, 3, "forecast", context="batch"
            )
        )
        self.assertEqual(observed, [("request-a", 0, 3, "forecast", "batch")])
        self.assertGreaterEqual(prefetcher.metrics()["fallback"], 1)
        self.assertFalse(prefetcher.wait_for_layer("request-a", 0, 3))

    def test_empty_transfer_plan_releases_lease(self):
        engine = _TransferEngine()
        releases = []
        lease = CallbackPageLease(lambda: releases.append(True))
        resolver = CallbackPrefetchResolver(
            lambda *_args: ResolvedPrefetch((), layer_num=1, lease=lease)
        )
        prefetcher = SparDAKVPrefetcher(engine, resolver)

        self.assertIsNone(prefetcher.prefetch_forecast("request-a", 0, 1, [4]))
        self.assertEqual(releases, [True])
        self.assertEqual(prefetcher.active_tickets(), ())

    def test_callback_page_lease_is_idempotent(self):
        releases = []
        lease = CallbackPageLease(lambda: releases.append(True))
        lease.release()
        lease.release()
        self.assertEqual(releases, [True])

    def test_submit_restricts_plan_to_the_predicted_layer(self):
        engine = _TransferEngine()
        transfer = L2Transfer(
            host_pool=None,
            device_pool=None,
            host_indices=torch.empty(0, dtype=torch.int64),
            device_indices=torch.empty(0, dtype=torch.int64),
        )
        resolver = CallbackPrefetchResolver(
            lambda *_args: ResolvedPrefetch((transfer,), layer_num=1)
        )
        prefetcher = SparDAKVPrefetcher(engine, resolver)

        ticket = prefetcher.prefetch_forecast("request-a", 0, 2, [5])
        self.assertIsNotNone(ticket)
        planned_transfers, layer_num, _ = engine.calls[0]
        self.assertEqual(layer_num, 3)
        self.assertIsNone(planned_transfers[0].layer_mapper(1))
        self.assertEqual(planned_transfers[0].layer_mapper(2), 2)
        prefetcher.cancel(ticket)


if __name__ == "__main__":
    unittest.main()
