"""Buffer-only mode cuts a storage hit too large for one load down to the
prefix that fits; a sidecar staging shortfall frees the KV bounce at once."""

import unittest
from array import array
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.cache_controller import HICACHE_LOAD_POOL_USAGE_FRACTION
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE = 4
_REQ = CacheRequestHandle("oversize", 0)


def _host_pool(device_tokens: int, ratio: float) -> MLATokenToKVPoolHost:
    device_pool = MLATokenToKVPool(
        size=device_tokens,
        page_size=PAGE,
        dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return MLATokenToKVPoolHost(
        device_pool.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=ratio,
        host_size=0,
        page_size=PAGE,
        layout="layer_first",
        pin_memory=False,
        device="cpu",
        device_capacity_tokens=device_pool.size,
        start_layer=device_pool.start_layer,
        end_layer=device_pool.end_layer,
        target_layer_num=device_pool.layer_num,
    )


class _Fixture:
    """A buffer-mode pipeline over a real (small, CPU) host pool group; the
    hit-time allocation, accounting and staging paths run unmocked. The KV
    arena holds 9 pages; the load budget (as the controller sizes it in
    buffer mode) is 8 of them; the SWA pool, when present, holds 5."""

    def __init__(self, test, *, swa_window_pages: int = 0):
        self.kv_pool = _host_pool(device_tokens=64, ratio=0.5)
        test.addCleanup(self.kv_pool.destroy)
        entries = [PoolEntry(PoolName.KV, self.kv_pool, None, None)]
        self.swa_pool = None
        if swa_window_pages:
            self.swa_pool = _host_pool(device_tokens=32, ratio=0.5)
            test.addCleanup(self.swa_pool.destroy)
            entries.append(PoolEntry(PoolName.SWA, self.swa_pool, None, None))

        cc = HybridCacheController.__new__(HybridCacheController)
        cc.page_size = PAGE
        cc.host_memory_mode = "buffer_only"
        cc.mem_pool_host = HostPoolGroup(entries)
        cc.prefetch_capacity_limit = int(
            HICACHE_LOAD_POOL_USAGE_FRACTION * cc.mem_pool_host.size
        )
        cc.prefetch_tokens_occupied = 0
        cc.prefetch_buffer = Queue()
        cc.host_mem_release_queue = Queue()
        cc.extra_host_mem_release_queues = {}
        cc.prefetch_hits_sync_groups = []
        self.cc = cc
        self.arena_pages = self.kv_pool.logical_size // PAGE
        self.budget_pages = cc.prefetch_capacity_limit // PAGE
        assert self.arena_pages == 9 and self.budget_pages == 8

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = cc
        cache.tree_core = SimpleNamespace(page_size=PAGE, is_eagle=False)
        cache.prefetch_threshold = PAGE
        cache.host_memory_mode = "buffer_only"
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = Mock()
        cache._prefetch_outcome_stats = defaultdict(int)
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache.revoke_pending_prefetch = Mock()
        cache.evict_host = Mock(return_value=0)
        cache.ongoing_prefetch = {}
        cache.storage_existence_cache = StorageExistenceCache()
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache.components = {}
        self.swa_component = None
        if swa_window_pages:
            self.swa_component = SimpleNamespace(
                pool_name=PoolName.SWA,
                component_type=ComponentType.SWA,
                full_window_pages=swa_window_pages,
                alloc_prefetch_staging=self.swa_pool.alloc,
            )
            cache.components[ComponentType.SWA] = self.swa_component
        self.cache = cache

        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline._swa_component = self.swa_component
        pipeline._auxiliary_swa_component = None
        pipeline.write_backlog_cap = 8
        pipeline.reset()
        cache.buffer_pipeline = pipeline
        self.pipeline = pipeline

    def hit(
        self,
        hit_pages: int,
        *,
        swa_window_pages: int = 0,
        assume_stored=False,
        request: CacheRequestHandle = _REQ,
    ):
        """An agreed hit of ``hit_pages`` over a span of as many pages, with
        the SWA window transfer the enqueue builds (placeholder keys, one per
        window page) when the fixture has an SWA pool."""
        tokens = list(range(1, hit_pages * PAGE + 1))
        chain = [f"page-{i}" for i in range(hit_pages)]
        comp_xfers = {}
        if swa_window_pages:
            swa = PoolTransfer(
                name=PoolName.SWA,
                keys=["__placeholder__"] * swa_window_pages,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
            comp_xfers[ComponentType.SWA] = [swa]
        aux = [x for xfers in comp_xfers.values() for x in xfers]
        operation = PrefetchOperation(
            request, tokens, pool_transfers=aux or None, assume_stored=assume_stored
        )
        operation.hash_value = list(chain)
        operation.storage_hit_count = hit_pages * PAGE
        operation.stats_requested_tokens = hit_pages * PAGE
        info = _OngoingPrefetch(
            0, RadixKey(array("q", tokens)), None, operation, None, comp_xfers
        )
        self.cache.ongoing_prefetch[request] = info
        self.cache._storage_prefetch_hit_remaining_by_reqid[request] = hit_pages * PAGE
        self.pipeline.set_prefix_ctx(request, [])
        return operation, info, chain

    def sync_hit_query(self, operation, local_restorable, *, peer_missing=()):
        """The hit-query agreement with this rank's restorable prefixes and a
        simulated peer that cannot restore ``peer_missing``."""
        operation.pool_storage_result.restorable_prefix_pages = local_restorable

        def reduce_with_peer(values, reduce_op, groups):
            assert reduce_op == torch.distributed.ReduceOp.MIN
            peer = values.clone()
            for page in peer_missing:
                peer[page] = 0
            values.copy_(torch.minimum(values, peer))

        self.cc._all_reduce = reduce_with_peer
        agreed = self.cc._sync_prefetch_hit_query(
            operation, operation.storage_hit_count
        )
        operation.storage_hit_count = agreed
        return agreed

    def allocate(self, operation, info, hit_tokens=None, aux_hit_tokens=None):
        if hit_tokens is None:
            hit_tokens = operation.storage_hit_count
        if aux_hit_tokens is None:
            aux_hit_tokens = hit_tokens
        return self.pipeline.allocate_and_queue_prefetch(
            operation, info, hit_tokens, aux_hit_tokens
        )

    def unfulfilled(self):
        return (
            self.cache.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens
        )


class TestBufferModeOversizeTrim(unittest.TestCase):
    def test_hit_larger_than_one_load_becomes_a_prefix_hit(self):
        fx = _Fixture(self)
        hit_pages = fx.arena_pages + 3
        operation, info, chain = fx.hit(hit_pages)
        span = fx.budget_pages * PAGE

        self.assertTrue(fx.allocate(operation, info))

        # Queued as a hit of the fitting prefix: chain, count, bounce and
        # occupancy all describe the prefix; nothing was declined.
        queued = fx.cc.prefetch_buffer.get_nowait()
        self.assertIs(queued, operation)
        self.assertEqual(operation.hash_value, chain[: fx.budget_pages])
        self.assertEqual(operation.storage_hit_count, span)
        self.assertEqual(operation.host_indices.numel(), span)
        self.assertEqual(fx.cc.prefetch_tokens_occupied, span)
        self.assertEqual(fx.kv_pool.available_size(), fx.kv_pool.logical_size - span)
        self.assertEqual(fx.cache._prefetch_outcome_stats["declined_host_oversize"], 0)
        self.assertEqual(fx.cache._prefetch_outcome_stats["cut_host_oversize"], 1)
        fx.cache.revoke_pending_prefetch.assert_not_called()
        # The cut tokens are unfulfilled for want of host capacity, once.
        fx.unfulfilled().assert_called_once_with(
            (hit_pages - fx.budget_pages) * PAGE, "host_capacity"
        )
        self.assertEqual(fx.cache._storage_prefetch_hit_remaining_by_reqid[_REQ], span)

        # The completed fetch stages exactly the prefix.
        self.assertTrue(
            fx.pipeline.stage_completed_prefetch(
                _REQ, operation.storage_hit_count, operation.hash_value
            )
        )
        staged = fx.pipeline.staged_prefetches[_REQ]
        self.assertEqual(staged.num_tokens, span)
        self.assertEqual(staged.hash_values, chain[: fx.budget_pages])
        self.assertEqual(staged.host_indices.numel(), span)
        self.assertEqual(len(staged.key_tokens), span)
        self.assertEqual(fx.cache.prefetch_loaded_tokens_by_reqid[_REQ], span)
        beliefs = fx.cache.storage_existence_cache.pool(PoolName.KV)
        self.assertTrue(beliefs.contains_all(chain[: fx.budget_pages]))
        self.assertFalse(beliefs.contains(chain[fx.budget_pages]))

    def test_cut_is_bounded_by_the_load_budget(self):
        """The cut hit holds at most the load budget, so it stages beside the
        budget's margin (write staging, a staged load) instead of waiting
        for an empty arena. A hit that fits the arena but not one load is
        cut the same way."""
        fx = _Fixture(self)
        margin = (fx.arena_pages - fx.budget_pages) * PAGE
        for hit_pages in (fx.arena_pages + 3, fx.arena_pages):
            with self.subTest(hit_pages=hit_pages):
                busy = fx.kv_pool.alloc(margin)
                operation, info, chain = fx.hit(hit_pages)

                self.assertTrue(fx.allocate(operation, info))

                self.assertIs(fx.cc.prefetch_buffer.get_nowait(), operation)
                self.assertEqual(operation.hash_value, chain[: fx.budget_pages])
                self.assertEqual(fx.kv_pool.available_size(), 0)
                fx.kv_pool.free(busy)
                fx.kv_pool.free(operation.host_indices)
                fx.cc.prefetch_tokens_occupied = 0

    def test_cut_hit_the_arena_cannot_hold_now_is_declined_not_parked(self):
        """Under a staged write backlog the arena cannot hold a cut hit's
        load. The hit is cut, then declined as a host-capacity shortfall
        (revoked holding nothing, a paced re-issue armed up to the cut end)
        instead of parked, so it never heads the parked FIFO: the next,
        smaller hit allocates at once beside the backlog. An un-cut hit keeps
        parking (the sidecar-shortfall test below)."""
        fx = _Fixture(self)
        # The write gate stages writes up to the load budget while no load is
        # granted; three staged pages already deny an eight-page load.
        backlog_pages = 3
        fx.kv_pool.alloc(backlog_pages * PAGE)
        hit_pages = fx.arena_pages + 3
        operation, info, chain = fx.hit(hit_pages)
        operation.storage_start = 2 * PAGE

        self.assertTrue(fx.allocate(operation, info))

        stats = fx.cache._prefetch_outcome_stats
        self.assertEqual(stats["cut_host_oversize"], 1)
        self.assertEqual(stats["declined_host_capacity"], 1)
        self.assertEqual(stats["declined_host_oversize"], 0)
        fx.cache.revoke_pending_prefetch.assert_called_once_with(_REQ)
        self.assertTrue(fx.cc.prefetch_buffer.empty())
        self.assertEqual(fx.cc.prefetch_tokens_occupied, 0)
        self.assertIsNone(operation.host_indices)
        self.assertEqual(
            fx.kv_pool.available_size(), (fx.arena_pages - backlog_pages) * PAGE
        )
        # The whole hit resolves as a host-capacity shortfall: the tail at the
        # cut, the cut span at the decline; nothing is left to account.
        self.assertEqual(
            [call.args for call in fx.unfulfilled().call_args_list],
            [
                ((hit_pages - fx.budget_pages) * PAGE, "host_capacity"),
                (fx.budget_pages * PAGE, "host_capacity"),
            ],
        )
        self.assertNotIn(_REQ, fx.cache._storage_prefetch_hit_remaining_by_reqid)
        # No paced re-issue: it would return as an un-cut hit and park.
        self.assertNotIn(_REQ.rid, fx.cache.storage_prefetch_retries._pending)

        # The next hit is not blocked: it stages beside the backlog at once.
        other = CacheRequestHandle("next", 0)
        operation2, info2, _ = fx.hit(4, request=other)
        self.assertTrue(fx.allocate(operation2, info2))
        self.assertIs(fx.cc.prefetch_buffer.get_nowait(), operation2)
        self.assertEqual(fx.cc.prefetch_tokens_occupied, 4 * PAGE)
        self.assertEqual(stats["declined_host_capacity"], 1)

    def test_cut_hit_with_a_sidecar_staging_shortfall_is_declined(self):
        """The same for the sidecar leg: a cut hit whose window cannot be
        staged now is declined with the KV bounce already back in the pool,
        not parked behind the shortfall."""
        window = 2
        fx = _Fixture(self, swa_window_pages=window)
        fx.swa_component.alloc_prefetch_staging = lambda num_tokens: None
        hit_pages = fx.arena_pages + 3
        operation, info, chain = fx.hit(hit_pages, swa_window_pages=window)
        fx.sync_hit_query(operation, [fx.budget_pages, hit_pages])

        self.assertTrue(fx.allocate(operation, info))

        stats = fx.cache._prefetch_outcome_stats
        self.assertEqual(stats["cut_host_oversize"], 1)
        self.assertEqual(stats["declined_host_capacity"], 1)
        fx.cache.revoke_pending_prefetch.assert_called_once_with(_REQ)
        self.assertEqual(fx.kv_pool.available_size(), fx.kv_pool.logical_size)
        self.assertEqual(fx.swa_pool.available_size(), fx.swa_pool.logical_size)
        self.assertTrue(fx.cc.host_mem_release_queue.empty())
        self.assertTrue(fx.cc.prefetch_buffer.empty())
        self.assertEqual(fx.cc.prefetch_tokens_occupied, 0)
        self.assertIsNone(info.comp_xfers[ComponentType.SWA][0].host_indices)
        self.assertEqual(fx.unfulfilled().call_count, 2)
        self.assertNotIn(_REQ.rid, fx.cache.storage_prefetch_retries._pending)

    def test_swa_hit_cuts_only_to_an_agreed_restorable_prefix(self):
        """With a trailing SWA window the cut may only end where every rank
        agreed the window is restorable (the hit-query MIN); without such a
        prefix inside the fit the hit is declined as oversize, holding
        nothing."""
        window = 2
        fx = _Fixture(self, swa_window_pages=window)
        hit_pages = fx.arena_pages + 3

        operation, info, chain = fx.hit(hit_pages, swa_window_pages=window)
        self.assertEqual(fx.sync_hit_query(operation, [hit_pages]), hit_pages * PAGE)
        self.assertTrue(fx.allocate(operation, info))
        self.assertEqual(fx.cache._prefetch_outcome_stats["declined_host_oversize"], 1)
        fx.cache.revoke_pending_prefetch.assert_called_once_with(_REQ)
        self.assertTrue(fx.cc.prefetch_buffer.empty())
        self.assertEqual(fx.kv_pool.available_size(), fx.kv_pool.logical_size)
        self.assertEqual(fx.swa_pool.available_size(), fx.swa_pool.logical_size)
        fx.unfulfilled().assert_not_called()

        # This rank can restore the fit itself (7 pages); the peer cannot, so
        # the agreed set decides: the longest agreed prefix inside the fit
        # wins and the SWA window follows the new endpoint.
        fx.cache.revoke_pending_prefetch.reset_mock()
        agreed = fx.budget_pages - 3
        local = [agreed, fx.budget_pages, fx.budget_pages + 1, hit_pages]
        operation, info, chain = fx.hit(hit_pages, swa_window_pages=window)
        self.assertEqual(
            fx.sync_hit_query(operation, local, peer_missing=[fx.budget_pages]),
            hit_pages * PAGE,
        )
        self.assertEqual(
            operation.restorable_hit_pages, [agreed, fx.budget_pages + 1, hit_pages]
        )
        # The backend's own verdict stays what this rank saw.
        self.assertEqual(operation.pool_storage_result.restorable_prefix_pages, local)
        self.assertTrue(fx.allocate(operation, info))
        queued = fx.cc.prefetch_buffer.get_nowait()
        self.assertEqual(queued.hash_value, chain[:agreed])
        self.assertEqual(queued.storage_hit_count, agreed * PAGE)
        self.assertIsNone(queued.sidecar_hash_values)
        swa = info.comp_xfers[ComponentType.SWA][0]
        self.assertEqual(swa.host_indices.numel(), window * PAGE)
        self.assertEqual(
            fx.swa_pool.available_size(), fx.swa_pool.logical_size - window * PAGE
        )
        fx.unfulfilled().assert_called_once_with(
            (hit_pages - agreed) * PAGE, "host_capacity"
        )
        fx.cache.revoke_pending_prefetch.assert_not_called()

    def test_head_trimmed_swa_hit_keeps_sidecar_chain_consistent(self):
        """After the hit-time device rematch dropped a FULL head, the cut keeps
        ``sidecar_hash_values`` as the chain ending where ``hash_value`` ends,
        the invariant the completion's belief healing relies on."""
        window = 2
        fx = _Fixture(self, swa_window_pages=window)
        head = 2
        hit_pages = fx.arena_pages + head + 2
        restorable = fx.arena_pages
        operation, info, chain = fx.hit(hit_pages, swa_window_pages=window)
        fx.sync_hit_query(operation, [restorable, hit_pages])
        fx.cc.trim_prefetch_full_head(operation, head * PAGE)
        tail_tokens = (hit_pages - head) * PAGE

        self.assertTrue(
            fx.allocate(operation, info, tail_tokens, aux_hit_tokens=hit_pages * PAGE)
        )

        queued = fx.cc.prefetch_buffer.get_nowait()
        self.assertEqual(queued.hash_value, chain[head:restorable])
        self.assertEqual(queued.storage_hit_count, (restorable - head) * PAGE)
        self.assertEqual(queued.sidecar_hit_pages, restorable)
        self.assertEqual(queued.sidecar_hash_values, chain[:restorable])
        self.assertEqual(len(queued.sidecar_hash_values) - len(queued.hash_value), head)
        fx.unfulfilled().assert_called_once_with(
            (hit_pages - restorable) * PAGE, "host_capacity"
        )

    def test_assumed_stored_hit_keeps_no_restorable_set(self):
        """An assumed-stored hit synthesizes every prefix flag (nothing was
        queried): only its endpoint is evidence, so no set is kept and an
        oversized such hit declines instead of cutting."""
        window = 2
        fx = _Fixture(self, swa_window_pages=window)
        hit_pages = fx.arena_pages + 3
        operation, info, chain = fx.hit(
            hit_pages, swa_window_pages=window, assume_stored=True
        )
        self.assertEqual(fx.sync_hit_query(operation, None), hit_pages * PAGE)
        self.assertIsNone(operation.restorable_hit_pages)

        self.assertTrue(fx.allocate(operation, info))

        self.assertEqual(fx.cache._prefetch_outcome_stats["declined_host_oversize"], 1)
        self.assertEqual(operation.hash_value, chain)
        self.assertTrue(fx.cc.prefetch_buffer.empty())

    def test_hit_without_a_usable_prefix_declines_as_oversize(self):
        """Not one page fits, or the only agreed-restorable prefix inside the
        fit is below the prefetch threshold: declined, holding nothing."""
        with self.subTest(case="no page fits"):
            fx = _Fixture(self)
            fx.cc.can_fit_prefetch_host_buffers = lambda operation, tokens, **_: False
            operation, info, chain = fx.hit(3)

            self.assertTrue(fx.allocate(operation, info))

            self.assertEqual(
                fx.cache._prefetch_outcome_stats["declined_host_oversize"], 1
            )
            fx.cache.revoke_pending_prefetch.assert_called_once_with(_REQ)
            self.assertEqual(operation.hash_value, chain)
            self.assertEqual(fx.kv_pool.available_size(), fx.kv_pool.logical_size)
            self.assertTrue(fx.cc.prefetch_buffer.empty())
            self.assertEqual(fx.cc.prefetch_tokens_occupied, 0)
            fx.unfulfilled().assert_not_called()

        with self.subTest(case="restorable prefix below the threshold"):
            window = 2
            fx = _Fixture(self, swa_window_pages=window)
            fx.cache.prefetch_threshold = 3 * PAGE
            hit_pages = fx.arena_pages + 3
            operation, info, chain = fx.hit(hit_pages, swa_window_pages=window)
            fx.sync_hit_query(operation, [2, hit_pages])

            self.assertTrue(fx.allocate(operation, info))

            self.assertEqual(
                fx.cache._prefetch_outcome_stats["declined_host_oversize"], 1
            )
            self.assertEqual(fx.cache._prefetch_outcome_stats["cut_host_oversize"], 0)
            fx.cache.revoke_pending_prefetch.assert_called_once_with(_REQ)
            self.assertEqual(operation.hash_value, chain)
            self.assertEqual(fx.kv_pool.available_size(), fx.kv_pool.logical_size)
            self.assertEqual(fx.swa_pool.available_size(), fx.swa_pool.logical_size)
            self.assertTrue(fx.cc.prefetch_buffer.empty())
            fx.unfulfilled().assert_not_called()

    def test_fit_sees_a_sidecar_pool_of_its_own(self):
        """With separate host pools the fit also bounds the SWA window by the
        SWA pool: a hit whose window outgrows that pool is cut to the longest
        agreed prefix whose window fits, rather than parked on a staging
        shortfall no drain can clear."""
        window = 6  # the SWA pool holds 5 pages
        fx = _Fixture(self, swa_window_pages=window)
        swa_pages = fx.swa_pool.logical_size // PAGE
        self.assertLess(swa_pages, window)
        hit_pages = window
        operation, info, chain = fx.hit(hit_pages, swa_window_pages=window)
        fx.sync_hit_query(operation, [swa_pages, hit_pages])

        self.assertTrue(fx.allocate(operation, info))

        queued = fx.cc.prefetch_buffer.get_nowait()
        self.assertEqual(queued.hash_value, chain[:swa_pages])
        self.assertEqual(
            info.comp_xfers[ComponentType.SWA][0].host_indices.numel(),
            swa_pages * PAGE,
        )
        self.assertEqual(fx.swa_pool.available_size(), 0)
        fx.unfulfilled().assert_called_once_with(
            (hit_pages - swa_pages) * PAGE, "host_capacity"
        )

    def test_sidecar_staging_shortfall_frees_the_kv_bounce_at_once(self):
        """A hit whose SWA window cannot be staged parks whole: the KV bounce
        is back in the pool before the call returns, not queued for the next
        release drain, and no occupancy or accounting moved."""
        window = 2
        fx = _Fixture(self, swa_window_pages=window)
        fx.swa_component.alloc_prefetch_staging = lambda num_tokens: None
        hit_pages = 4
        operation, info, chain = fx.hit(hit_pages, swa_window_pages=window)

        self.assertFalse(fx.allocate(operation, info))

        self.assertEqual(fx.kv_pool.available_size(), fx.kv_pool.logical_size)
        self.assertTrue(fx.cc.host_mem_release_queue.empty())
        self.assertTrue(fx.cc.prefetch_buffer.empty())
        self.assertEqual(fx.cc.prefetch_tokens_occupied, 0)
        self.assertIsNone(operation.host_indices)
        self.assertIsNone(info.comp_xfers[ComponentType.SWA][0].host_indices)
        self.assertEqual(operation.hash_value, chain)
        self.assertEqual(
            fx.cache._storage_prefetch_hit_remaining_by_reqid[_REQ], hit_pages * PAGE
        )
        fx.unfulfilled().assert_not_called()
        fx.cache.revoke_pending_prefetch.assert_not_called()

        # The retry launches once the window can be staged.
        fx.swa_component.alloc_prefetch_staging = fx.swa_pool.alloc
        self.assertTrue(fx.allocate(operation, info))
        self.assertIs(fx.cc.prefetch_buffer.get_nowait(), operation)
        self.assertEqual(fx.cc.prefetch_tokens_occupied, hit_pages * PAGE)


if __name__ == "__main__":
    unittest.main()
