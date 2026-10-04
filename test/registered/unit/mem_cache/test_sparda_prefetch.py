"""Registered CPU tests for the one-layer SparDA prefetch contract."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_cleanup_reports_event_sync_failure(self):
        engine = _TransferEngine()
        engine.finish_event = _FlakyEvent()
        lease = _Lease(engine.finish_event)
        prefetcher = SparDAKVPrefetcher(
            engine,
            CallbackPrefetchResolver(lambda *_args: _resolved(engine, lease)),
        )

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])
        self.assertFalse(prefetcher.cleanup_request("request-a"))
        self.assertEqual(prefetcher.active_tickets(), (ticket,))
        self.assertEqual(lease.releases, 0)

        self.assertTrue(prefetcher.cleanup_request("request-a"))
        self.assertEqual(prefetcher.active_tickets(), ())
        self.assertEqual(lease.releases, 1)

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

    def test_demand_mode_submits_only_when_target_layer_starts(self):
        engine = _SynchronousTransferEngine()
        events = []
        lease = CallbackPageLease(lambda: events.append("release"))

        class _Resolver:
            def predict(self, *_args):
                events.append("predict")
                return [3]

            def resolve(self, *_args):
                events.append("resolve")
                return ResolvedPrefetch(
                    transfers=(object(),),
                    layer_num=1,
                    lease=lease,
                )

        prefetcher = SparDAKVPrefetcher(engine, _Resolver(), submit_on_wait=True)
        ticket = prefetcher.prefetch_forecast_query("request-a", 0, 1, "forecast")

        self.assertIsNotNone(ticket)
        self.assertEqual(events, ["predict", "resolve"])
        self.assertEqual(engine.calls, [])
        self.assertTrue(prefetcher.wait_for_layer("request-a", 0, 1))
        self.assertEqual(len(engine.calls), 1)
        self.assertTrue(prefetcher.consume_for_layer("request-a", 0, 1))
        self.assertEqual(events[-1], "release")

    def test_failed_consume_release_keeps_ticket_for_cleanup_retry(self):
        engine = _SynchronousTransferEngine()

        class _RetryLease:
            def __init__(self):
                self.releases = 0

            def release(self):
                self.releases += 1
                if self.releases == 1:
                    raise RuntimeError("release not ready")

        lease = _RetryLease()
        prefetcher = SparDAKVPrefetcher(
            engine,
            CallbackPrefetchResolver(lambda *_args: _resolved(engine, lease)),
        )

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])
        self.assertFalse(prefetcher.consume(ticket))
        self.assertEqual(ticket.state, PrefetchTicketState.CONSUMED)
        self.assertFalse(ticket.done)
        self.assertEqual(prefetcher.active_tickets(), (ticket,))

        self.assertTrue(prefetcher.cleanup_request("request-a"))
        self.assertEqual(lease.releases, 2)
        self.assertEqual(prefetcher.active_tickets(), ())

    def test_submit_failure_keeps_remote_lease_ticket_reachable(self):
        releases = []

        def release():
            releases.append(True)
            if len(releases) == 1:
                raise RuntimeError("remote lease status unknown")

        def submit_callback():
            raise RuntimeError("remote retrieve failed")

        lease = CallbackPageLease(release)
        resolver = CallbackPrefetchResolver(
            lambda *_args: ResolvedPrefetch(
                transfers=(),
                layer_num=1,
                lease=lease,
                submit_callback=submit_callback,
            )
        )
        prefetcher = SparDAKVPrefetcher(None, resolver)

        with self.assertRaisesRegex(RuntimeError, "remote retrieve failed"):
            prefetcher.prefetch_forecast("request-a", 0, 1, [4])

        (ticket,) = prefetcher.active_tickets()
        self.assertEqual(ticket.state, PrefetchTicketState.CANCELLED)
        self.assertEqual(releases, [True])
        self.assertTrue(prefetcher.cleanup_request("request-a"))
        self.assertEqual(releases, [True, True])
        self.assertEqual(prefetcher.active_tickets(), ())

    def test_stale_submit_keeps_failed_release_for_retry(self):
        releases = []

        def release():
            releases.append(True)
            if len(releases) == 1:
                raise RuntimeError("remote lease status unknown")

        lease = CallbackPageLease(release)
        prefetcher = SparDAKVPrefetcher(None)
        prefetcher._latest_generation["request-a"] = 2

        ticket = prefetcher.submit(
            "request-a",
            1,
            0,
            [4],
            ResolvedPrefetch(
                transfers=(),
                layer_num=1,
                lease=lease,
                safe_without_transfer=True,
            ),
        )

        self.assertEqual(ticket.state, PrefetchTicketState.CANCELLED)
        self.assertEqual(releases, [True])
        self.assertEqual(prefetcher.active_tickets(), (ticket,))
        self.assertTrue(prefetcher.cleanup_request("request-a", 1))
        self.assertEqual(releases, [True, True])
        self.assertEqual(prefetcher.active_tickets(), ())

    def test_duplicate_losing_lease_is_retained_when_both_releases_fail(self):
        release_calls = []

        def flaky_release(name):
            def release():
                release_calls.append(name)
                if release_calls.count(name) == 1:
                    raise RuntimeError(f"{name} lease status unknown")

            return release

        prefetcher = SparDAKVPrefetcher(None)
        first = prefetcher.submit(
            "request-a",
            0,
            0,
            [4],
            ResolvedPrefetch(
                transfers=(),
                layer_num=1,
                lease=CallbackPageLease(flaky_release("first")),
                safe_without_transfer=True,
            ),
        )
        loser = prefetcher.submit(
            "request-a",
            0,
            0,
            [4],
            ResolvedPrefetch(
                transfers=(),
                layer_num=1,
                lease=CallbackPageLease(flaky_release("loser")),
                safe_without_transfer=True,
            ),
        )

        self.assertEqual(first.state, PrefetchTicketState.CANCELLED)
        self.assertEqual(loser.state, PrefetchTicketState.CANCELLED)
        self.assertEqual(len(prefetcher.active_tickets()), 2)
        self.assertTrue(prefetcher.cleanup_request("request-a", 0))
        self.assertEqual(
            release_calls,
            ["first", "loser", "first", "loser"],
        )
        self.assertEqual(prefetcher.active_tickets(), ())

    def test_lmcache_overlay_restores_authoritative_pages_across_layers(self):
        try:
            from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
                LMCacheUnifiedRadixCache,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        page_table = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.int64)
        authoritative = SimpleNamespace(
            req_to_token=torch.tensor(
                [[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.int64
            )
        )
        staging_pages = iter(
            [
                torch.tensor([100, 101], dtype=torch.int64),
                torch.tensor([200, 201], dtype=torch.int64),
                # Reuse the first staging pages on the next decode step.
                torch.tensor([100, 101], dtype=torch.int64),
                torch.tensor([300, 301], dtype=torch.int64),
                torch.tensor([400, 401], dtype=torch.int64),
            ]
        )
        freed = []

        class _Future:
            def __init__(self, value):
                self.value = value

            def result(self, timeout=None):
                del timeout
                return self.value

        class _Connector:
            num_layers = 4
            chunk_size = 2

            def sparse_prefetch_available(self):
                return True

            _mq_timeout = 1.0

            def create_sparse_object_keys(
                self, token_ids, chunk_indices, cache_salt, **kwargs
            ):
                del token_ids, cache_salt, kwargs
                return [f"chunk-{index}" for index in chunk_indices]

            def sparse_prefetch(self, *args):
                del args
                return _Future(True)

            def sparse_retrieve(self, *args):
                del args
                return _Future((True, [0]))

            def sparse_release_prefetch(self, *args):
                del args
                return _Future(True)

            def sparse_cancel_prefetch(self, *args):
                del args
                return _Future(True)

        class _Allocator:
            def alloc(self, count):
                pages = next(staging_pages)
                assert pages.numel() == count
                return pages.clone()

            def free(self, pages):
                freed.append(pages.clone())

        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.tree_core = SimpleNamespace(device=torch.device("cpu"))
        cache.page_size = 1
        cache.req_to_token_pool = authoritative
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.lmcache_connector = _Connector()

        request = SimpleNamespace(
            rid="request",
            cache_salt="salt",
            kv=SimpleNamespace(req_pool_idx=0),
            get_fill_ids=lambda: list(range(5)),
        )
        forward_batch = SimpleNamespace(
            seq_lens_cpu=[5],
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
        )
        backend = SimpleNamespace(
            block_size=1,
            forward_metadata=SimpleNamespace(
                base=SimpleNamespace(page_table=page_table)
            ),
        )
        context = SimpleNamespace(
            request=request,
            forward_batch=forward_batch,
            selector_backend=backend,
            request_index=0,
        )

        def run_adjacent_layers(expected_active, expected_restored):
            tickets = []
            for layer_id in (0, 1):
                resolved = cache.resolve_sparda_prefetch(
                    "request", 0, layer_id, [0], context
                )
                completion = resolved.submit_callback()
                completion.synchronize()
                self.assertTrue(resolved.on_ready())
                tickets.append(resolved.lease)
            self.assertEqual(page_table[0, :2].tolist(), expected_active)
            # Simulate the authoritative request mapping changing while two
            # layer overlays are live. The last owner must restore this value,
            # not the mapping captured by an earlier layer.
            authoritative.req_to_token[0, :2] = torch.tensor([40, 41])
            tickets[0].release()
            # Layer 1 owns the visible overlay, so layer 0 must not restore
            # over it while its staging pages are still in use.
            self.assertEqual(page_table[0, :2].tolist(), expected_active)
            tickets[1].release()
            self.assertEqual(page_table[0, :2].tolist(), expected_restored)
            authoritative.req_to_token[0, :2] = torch.tensor([10, 11])

        run_adjacent_layers([200, 201], [40, 41])
        # A later decode reuses an old staging page; it must still restore the
        # authoritative request mapping rather than a previous overlay.
        run_adjacent_layers([300, 301], [40, 41])
        self.assertEqual(
            [page.tolist() for page in freed],
            [
                [100, 101],
                [200, 201],
                [100, 101],
                [300, 301],
            ],
        )

        authoritative.req_generation = torch.tensor([7], dtype=torch.int64)
        resolved = cache.resolve_sparda_prefetch("request", 0, 0, [0], context)
        completion = resolved.submit_callback()
        completion.synchronize()
        self.assertTrue(resolved.on_ready())
        authoritative.req_generation[0] = 8
        with self.assertRaisesRegex(RuntimeError, "overlay owner changed"):
            resolved.lease.release()
        authoritative.req_generation[0] = 7
        resolved.lease.release()
        self.assertEqual(page_table[0, :2].tolist(), [10, 11])
        self.assertEqual(
            [page.tolist() for page in freed],
            [
                [100, 101],
                [200, 201],
                [100, 101],
                [300, 301],
                [400, 401],
            ],
        )

    def test_lmcache_store_uses_radix_mapping_after_request_row_reuse(self):
        from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
            LMCacheUnifiedRadixCache,
        )
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        request_row = torch.tensor([[7, 8, 9, 10]], dtype=torch.int64)
        authoritative = torch.tensor([41, 42, 43, 44], dtype=torch.int64)
        node = object()
        lock_params = object()
        operation = SimpleNamespace(request_id="request")
        connector = SimpleNamespace(
            chunk_size=4,
            parallel_all_reduce=Mock(),
            get_store_start=Mock(return_value=0),
            device_indices_by_group=lambda slots, **kwargs: [slots],
            build_cache_salt=Mock(return_value=""),
            submit_store=Mock(return_value=operation),
        )
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.tree_core = SimpleNamespace(device=torch.device("cpu"))
        cache.page_size = 1
        cache.tree_core = SimpleNamespace(is_eagle=False, page_size=1)
        cache._mamba_component = None
        cache._pending_stores = []
        cache._pending_store_counts = {}
        cache.lmcache_connector = connector
        cache.inc_lock_ref = Mock(
            return_value=SimpleNamespace(to_dec_params=lambda: lock_params)
        )
        cache.dec_lock_ref = Mock()
        matched = SimpleNamespace(device_indices=authoritative, last_device_node=node)
        request = SimpleNamespace(rid="request", extra_key=None, cache_salt=None)
        # The request row has already been recycled. Store must use the tree.
        request_row.fill_(-1)
        with patch.object(UnifiedRadixCache, "match_prefix", return_value=matched):
            cache._submit_store(request, [1, 2, 3, 4])

        args = connector.submit_store.call_args.args
        self.assertEqual(args[:2], ("request", [1, 2, 3, 4]))
        self.assertEqual(args[2][0].tolist(), authoritative.tolist())
        self.assertEqual(request_row.tolist(), [[-1, -1, -1, -1]])
        self.assertIs(cache._pending_stores[0].operation, operation)
        self.assertIs(cache._pending_stores[0].lock_params, lock_params)
        cache.dec_lock_ref.assert_not_called()

    def test_lmcache_host_resident_prefetch_targets_authoritative_pages(self):
        try:
            from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
                LMCacheUnifiedRadixCache,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        retrieved_block_ids = []

        class _Future:
            def __init__(self, value):
                self.value = value

            def result(self, timeout=None):
                del timeout
                return self.value

        class _Connector:
            num_layers = 2
            chunk_size = 2

            def sparse_prefetch_available(self):
                return True

            _mq_timeout = 1.0

            def create_sparse_object_keys(self, token_ids, chunk_indices, **kwargs):
                del token_ids, kwargs
                return [f"chunk-{index}" for index in chunk_indices]

            def sparse_prefetch(self, *args):
                del args
                return _Future(True)

            def sparse_retrieve(self, request_id, generation, layer_id, keys, blocks):
                del request_id, generation, layer_id, keys
                retrieved_block_ids.append(blocks)
                return _Future((True, [0]))

            def sparse_release_prefetch(self, *args):
                del args
                return _Future(True)

            def sparse_cancel_prefetch(self, *args):
                del args
                return _Future(True)

        class _Allocator:
            def __init__(self):
                self.freed = []

            def alloc(self, count):
                raise AssertionError(f"unexpected staging allocation: {count}")

            def free(self, pages):
                self.freed.append(pages)

        page_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.tree_core = SimpleNamespace(device=torch.device("cpu"))
        cache.page_size = 1
        cache.req_to_token_pool = SimpleNamespace(req_to_token=page_table.clone())
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.lmcache_connector = _Connector()

        request = SimpleNamespace(
            rid="request",
            cache_salt=None,
            _sparda_host_resident=True,
            kv=SimpleNamespace(req_pool_idx=0),
            get_fill_ids=lambda: list(range(4)),
        )
        forward_batch = SimpleNamespace(
            seq_lens_cpu=[4],
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
        )
        backend = SimpleNamespace(
            block_size=1,
            forward_metadata=SimpleNamespace(
                base=SimpleNamespace(page_table=page_table)
            ),
        )
        context = SimpleNamespace(
            request=request,
            forward_batch=forward_batch,
            selector_backend=backend,
            request_index=0,
        )

        resolved = cache.resolve_sparda_prefetch("request", 0, 0, [0], context)
        completion = resolved.submit_callback()
        completion.synchronize()
        self.assertTrue(resolved.on_ready())
        resolved.lease.release()

        self.assertEqual(retrieved_block_ids, [[[10, 11]]])
        self.assertEqual(page_table.tolist(), [[10, 11, 12, 13]])
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [])

    def test_lmcache_sparse_prefetch_requires_cleanup_hooks(self):
        try:
            from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
                LMCacheUnifiedRadixCache,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.tree_core = SimpleNamespace(device=torch.device("cpu"))
        cache.page_size = 1
        cache.lmcache_connector = SimpleNamespace(
            sparse_prefetch_available=lambda: True,
            sparse_prefetch=lambda: None,
            sparse_retrieve=lambda: None,
            create_sparse_object_keys=lambda: None,
        )

        self.assertFalse(cache.sparda_prefetch_available())

    def test_lmcache_host_resident_admission_requires_resolver(self):
        try:
            from sglang.srt.mem_cache.radix_cache import RadixKey
            from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
                LMCacheUnifiedRadixCache,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        request = SimpleNamespace(
            full_untruncated_fill_ids=list(range(8)),
        )
        marker = SimpleNamespace(
            key=RadixKey(list(range(8)), None, cache_salt=None),
            local_hit_tokens=0,
            total_hit=8,
            load=None,
            lookup=SimpleNamespace(lock_start=0),
        )
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.tree_core = SimpleNamespace(device=torch.device("cpu"))
        cache.page_size = 1
        cache._sparda_host_resident_enabled = True
        cache.sparda_prefetcher = None
        cache._sparda_index_available = Mock(return_value=True)
        cache.lmcache_connector = SimpleNamespace(
            sparse_prefetch_available=lambda: True,
            sparse_prefetch=lambda: None,
            sparse_retrieve=lambda: None,
            create_sparse_object_keys=lambda: None,
            sparse_cancel_prefetch=lambda: None,
            sparse_release_prefetch=lambda: None,
        )

        self.assertFalse(cache._sparda_can_admit_host_resident(marker, request))
        cache._sparda_index_available.assert_not_called()

    def test_lmcache_host_resident_admission_skips_full_retrieve(self):
        try:
            from sglang.srt.mem_cache.radix_cache import RadixKey
            from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
                LMCacheUnifiedRadixCache,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        class _Allocator:
            def __init__(self):
                self.free = Mock()

            def available_size(self):
                return 64

            def alloc(self, num_tokens):
                return torch.arange(100, 100 + num_tokens, dtype=torch.int64)

        request = SimpleNamespace(
            rid="request",
            kv=SimpleNamespace(req_pool_idx=0),
            full_untruncated_fill_ids=list(range(9)),
        )
        marker = SimpleNamespace(
            key=RadixKey(list(range(8)), None, cache_salt=None),
            local_hit_tokens=0,
            total_hit=8,
            load=None,
            lookup=SimpleNamespace(lock_start=0),
        )
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.tree_core = SimpleNamespace(device=torch.device("cpu"))
        cache.page_size = 1
        cache.tree_core.device = torch.device("cpu")
        cache._external_flows = {request.rid: marker}
        cache._sparda_host_resident_enabled = True
        cache.sparda_prefetcher = object()
        cache._sparda_index_available = Mock(return_value=True)
        cache._start_external_load = Mock(
            side_effect=AssertionError("host-resident admission did a full load")
        )
        cache.token_to_kv_pool_allocator = _Allocator()
        cache.lmcache_connector = SimpleNamespace(
            chunk_size=4,
            sparse_prefetch_available=lambda: True,
            sparse_prefetch=lambda: None,
            sparse_retrieve=lambda: None,
            create_sparse_object_keys=lambda: None,
            sparse_cancel_prefetch=lambda: None,
            sparse_release_prefetch=lambda: None,
            retrieve_kv=Mock(
                side_effect=AssertionError("host-resident admission retrieved KV")
            ),
            free_lookup_locks=Mock(),
            parallel_all_reduce=Mock(),
        )

        result = cache.init_load_back(
            SimpleNamespace(
                req=request,
                best_match_node=SimpleNamespace(),
                host_hit_length=8,
            )
        )

        self.assertEqual(result[0].tolist(), list(range(100, 108)))
        self.assertIs(result[1], result[1])
        self.assertTrue(request._sparda_host_resident)
        self.assertEqual(request._sparda_host_prefix_len, 8)
        cache._start_external_load.assert_not_called()
        cache.lmcache_connector.retrieve_kv.assert_not_called()

    def test_lmcache_host_resident_index_miss_uses_full_load_fallback(self):
        try:
            from sglang.srt.mem_cache.radix_cache import RadixKey
            from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
                LMCacheUnifiedRadixCache,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        request = SimpleNamespace(rid="request")
        marker = SimpleNamespace(
            key=RadixKey(list(range(8)), None, cache_salt=None),
            local_hit_tokens=0,
            total_hit=8,
            load=None,
            lookup=SimpleNamespace(lock_start=0),
        )
        expected_slots = torch.tensor([7, 8])
        node = SimpleNamespace()
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache._external_flows = {request.rid: marker}
        cache._sparda_host_resident_enabled = True
        cache._sparda_index_available = Mock(return_value=False)
        cache._start_external_load = Mock(return_value=expected_slots)
        cache._sparda_can_admit_host_resident = Mock(return_value=False)

        result = cache.init_load_back(
            SimpleNamespace(
                req=request,
                best_match_node=SimpleNamespace(),
                host_hit_length=8,
            )
        )

        self.assertIs(result[0], expected_slots)
        cache._start_external_load.assert_called_once()
        self.assertFalse(getattr(request, "_sparda_host_resident", False))

    def test_host_resident_finish_discards_partial_pages_without_full_restore(self):
        from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
            LMCacheUnifiedRadixCache,
        )

        request = SimpleNamespace(
            rid="request",
            finished=lambda: True,
            _sparda_host_resident=True,
            _sparda_host_marker=object(),
            _sparda_host_prefix_len=4,
            _sparda_host_prefix_start=0,
            _sparda_host_lookup_released=True,
        )
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.sparda_prefetcher = SimpleNamespace(
            cleanup_request=Mock(return_value=True)
        )
        cache._retire_loaded_flow = Mock()
        cache._request_session_finish = Mock()
        cache._materialize_sparda_host_request = Mock(
            side_effect=AssertionError("completion restored full KV")
        )
        self.assertFalse(cache.claim_kv_row(request))
        self.assertTrue(request.skip_radix_cache_insert)
        self.assertFalse(request._sparda_host_resident)
        cache.sparda_prefetcher.cleanup_request.assert_called_once_with("request")
        cache._materialize_sparda_host_request.assert_not_called()
        cache._retire_loaded_flow.assert_called_once_with("request")
        cache._request_session_finish.assert_called_once_with("request")

    def test_host_restore_timeout_retains_writer_for_retry(self):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
            LMCacheUnifiedRadixCache,
        )

        lookup = SimpleNamespace(total_hit_tokens=4, lock_start=0)
        operation = SimpleNamespace(
            future=SimpleNamespace(result=Mock(side_effect=[TimeoutError(), True]))
        )
        connector = SimpleNamespace(
            operation_timeout=1.0,
            submit_lookup=Mock(return_value=lookup),
            poll_lookup=Mock(return_value=4),
            device_indices_by_group=lambda slots: [slots],
            submit_load=Mock(return_value=operation),
            prepare_load_on_stream=Mock(return_value=True),
            complete_load=Mock(return_value=True),
        )
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.tree_core = SimpleNamespace(device=torch.device("cpu"))
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 13]])
        )
        cache._forward_stream = object()
        cache.lmcache_connector = connector
        request = SimpleNamespace(
            rid="request",
            kv=SimpleNamespace(req_pool_idx=0),
            _sparda_host_resident=True,
            _sparda_host_marker=SimpleNamespace(key=RadixKey([1, 2, 3, 4])),
            _sparda_host_prefix_start=0,
            _sparda_host_prefix_len=4,
        )

        self.assertFalse(cache._materialize_sparda_host_request(request))
        self.assertTrue(request._sparda_host_resident)
        self.assertIs(request._sparda_restore_load, operation)
        connector.complete_load.assert_not_called()
        self.assertTrue(cache._materialize_sparda_host_request(request))
        connector.submit_lookup.assert_called_once()
        connector.submit_load.assert_called_once()
        self.assertEqual(
            connector.submit_load.call_args.kwargs["owned_device_indices"].tolist(),
            [10, 11, 12, 13],
        )
        self.assertFalse(request._sparda_host_resident)
        self.assertIsNone(request._sparda_restore_load)

    def test_host_discard_keeps_pending_writer_until_it_can_be_drained(self):
        from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
            LMCacheUnifiedRadixCache,
        )

        operation = SimpleNamespace(
            future=SimpleNamespace(result=Mock(side_effect=[TimeoutError(), False]))
        )
        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache.sparda_prefetcher = None
        cache.lmcache_connector = SimpleNamespace(
            operation_timeout=1.0, complete_load=Mock(return_value=False)
        )
        request = SimpleNamespace(
            rid="request",
            _sparda_host_resident=True,
            _sparda_restore_load=operation,
        )
        self.assertFalse(cache._discard_sparda_host_request(request))
        self.assertTrue(request._sparda_host_resident)
        self.assertIs(request._sparda_restore_load, operation)
        cache.lmcache_connector.complete_load.assert_not_called()
        self.assertTrue(cache._discard_sparda_host_request(request))
        cache.lmcache_connector.complete_load.assert_called_once_with(operation)
        self.assertFalse(request._sparda_host_resident)
        self.assertIsNone(request._sparda_restore_load)

    def test_gpu_reset_keeps_host_compressed_index(self):
        try:
            from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
                LMCacheUnifiedRadixCache,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        cache = LMCacheUnifiedRadixCache.__new__(LMCacheUnifiedRadixCache)
        cache._sparda_compressed_indices = {("request", None): {0: (1, 2)}}
        cache._sparda_compressed_index_order = [("request", None)]
        cache._sparda_metrics = {"index_hit": 1}
        cache.lmcache_connector = None

        with patch.object(LMCacheUnifiedRadixCache.__mro__[1], "reset"):
            cache.reset()

        self.assertEqual(
            cache._sparda_compressed_indices,
            {("request", None): {0: (1, 2)}},
        )
        self.assertEqual(cache._sparda_compressed_index_order, [("request", None)])
        self.assertEqual(cache._sparda_metrics, {})

    def test_ready_and_consume_callbacks_run_before_release(self):
        engine = _SynchronousTransferEngine()
        callbacks = []
        lease = CallbackPageLease(
            lambda: callbacks.append("release"),
            consumed_callback=lambda: callbacks.append("consumed"),
        )
        resolver = CallbackPrefetchResolver(
            lambda *_args: ResolvedPrefetch(
                transfers=(object(),),
                layer_num=1,
                lease=lease,
                on_ready=lambda: callbacks.append("ready"),
            )
        )
        prefetcher = SparDAKVPrefetcher(engine, resolver)

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])
        self.assertTrue(prefetcher.consume(ticket))
        self.assertEqual(callbacks, ["ready", "consumed", "release"])

    def test_safe_without_transfer_is_a_ready_ticket(self):
        engine = _SynchronousTransferEngine()
        callbacks = []
        resolver = CallbackPrefetchResolver(
            lambda *_args: ResolvedPrefetch(
                transfers=(),
                layer_num=1,
                on_ready=lambda: callbacks.append("ready"),
                safe_without_transfer=True,
            )
        )
        prefetcher = SparDAKVPrefetcher(engine, resolver)

        ticket = prefetcher.prefetch_forecast("request-a", 0, 1, [4])
        self.assertTrue(prefetcher.wait(ticket))
        self.assertEqual(callbacks, ["ready"])
        self.assertTrue(prefetcher.consume(ticket))
        self.assertEqual(engine.calls, [])

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

    def test_callback_page_lease_serializes_consume_and_release(self):
        consumed_started = threading.Event()
        allow_consume = threading.Event()
        callbacks = []
        consume_allowed = []

        def consumed_callback():
            consumed_started.set()
            consume_allowed.append(allow_consume.wait(timeout=1))
            callbacks.append("consumed")

        lease = CallbackPageLease(
            release_callback=lambda: callbacks.append("release"),
            consumed_callback=consumed_callback,
        )
        consume_thread = threading.Thread(target=lease.mark_consumed)
        release_thread = threading.Thread(target=lease.release)
        consume_thread.start()
        self.assertTrue(consumed_started.wait(timeout=1))
        release_thread.start()
        allow_consume.set()
        consume_thread.join(timeout=1)
        release_thread.join(timeout=1)

        self.assertFalse(consume_thread.is_alive())
        self.assertFalse(release_thread.is_alive())
        self.assertEqual(consume_allowed, [True])
        self.assertEqual(callbacks, ["consumed", "release"])

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
