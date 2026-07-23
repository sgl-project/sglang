"""Pure-logic unit tests for the strict bit-exact SWA HiCache feature.

Covers the strict bit-exact SWA HiCache logic:
  * sizing: hybrid_pool_assembler._swa_host_num_pages
  * co-eviction observability: UnifiedRadixCache._note_binding_full_coevict
  * strict atomic leaf eviction: UnifiedRadixCache.drive_host_leaf_eviction
    and SWAComponent.drive_host_eviction routing
  * offload geometry: DeepSeekV4TokenToKVPool.swa_region_buffers page unit

No GPU / model is required; heavy collaborators are faked so we exercise only
the new logic. Run:
  PYTHONPATH=<worktree>/python python -m pytest test/srt/mem_cache/test_swa_bitexact_hicache.py -q
"""

import math
import types
import unittest

from sglang.srt.mem_cache import unified_radix_cache as R
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    EvictLayer,
)

FULL = R.BASE_COMPONENT_TYPE
SWA = ComponentType.SWA


def _sargs(stride=1):
    return types.SimpleNamespace(hicache_swa_offload_page_stride=stride)


class TestSwaHostSizing(unittest.TestCase):
    """Stride model (Task A3): SWA host pool == ceil(full_host_pages / stride)
    + a device-ring-bounded tail allowance."""

    def _pages(
        self,
        *,
        stride=1,
        full_host_pages=100_000,
        device_ring_pages=65,
        page_bytes=1,
        page_size=256,
    ):
        return A._swa_host_num_pages(
            server_args=_sargs(stride),
            full_host_pages=full_host_pages,
            device_ring_pages=device_ring_pages,
            page_bytes=page_bytes,
            page_size=page_size,
        )

    def test_stride1_covers_all_pages(self):
        # stride 1 == per-page: one window per full page + tail allowance.
        self.assertEqual(self._pages(stride=1), 100_000 + 65)

    def test_larger_stride_shrinks_pool(self):
        # coarser stride -> fewer windows -> smaller SWA pool.
        self.assertLess(self._pages(stride=8), self._pages(stride=1))
        self.assertEqual(self._pages(stride=8), math.ceil(100_000 / 8) + 65)

    def test_tail_allowance_bounded_by_device_ring(self):
        # a huge stride collapses the strided part to a single window; the pool
        # is then just that window plus the device-ring tail allowance.
        self.assertEqual(self._pages(stride=10_000_000, device_ring_pages=65), 1 + 65)

    def test_floor_one_page(self):
        self.assertEqual(
            self._pages(stride=10_000_000, device_ring_pages=0, full_host_pages=1),
            1,
        )

    def test_no_84gb_regression(self):
        # A coarse stride keeps the pool a small fraction of the full host pool
        # -- NOT device_ring * ratio which over-allocated to ~84GB.
        pages = self._pages(stride=64, full_host_pages=100_000)
        self.assertLess(pages, 100_000 * 0.02)
        self.assertEqual(pages, math.ceil(100_000 / 64) + 65)

    def test_warn_above_16gb_but_no_clamp(self):
        # page_bytes chosen so the result exceeds the 16GB slow-launch threshold.
        expected = 100_000 + 65  # stride 1
        page_bytes = int(16e9 / expected) + 1_000_000  # push over 16GB
        with self.assertLogs(A.logger, level="WARNING") as cm:
            pages = self._pages(
                stride=1, full_host_pages=100_000, page_bytes=page_bytes
            )
        self.assertEqual(pages, expected)  # warned, not clamped
        self.assertTrue(any("may slow server launch" in m for m in cm.output))

    def test_no_warn_below_16gb(self):
        # Small page_bytes -> comfortably under threshold -> no warning emitted.
        with self.assertRaises(AssertionError):
            with self.assertLogs(A.logger, level="WARNING"):
                self._pages(stride=1, full_host_pages=100_000, page_bytes=1)


class TestCoEvictWarning(unittest.TestCase):
    def _fresh(self):
        return types.SimpleNamespace()

    def _note(self, obj, full_tokens, leaves):
        R.UnifiedRadixCache._note_binding_full_coevict(obj, full_tokens, leaves)

    def test_noop_on_nonpositive(self):
        obj = self._fresh()
        self._note(obj, 0, 5)
        self._note(obj, 5, 0)
        self.assertFalse(hasattr(obj, "_binding_full_coevict_tokens"))

    def test_below_threshold_no_warn(self):
        obj = self._fresh()
        # 15 leaves total (< 16) across two calls -> accumulate, do not warn.
        with self.assertRaises(AssertionError):
            with self.assertLogs(R.logger, level="WARNING"):
                self._note(obj, 100, 8)
                self._note(obj, 100, 7)
        self.assertEqual(obj._binding_full_coevict_leaves, 15)
        self.assertFalse(getattr(obj, "_binding_full_coevict_warned", False))

    def test_warns_once_after_threshold(self):
        obj = self._fresh()
        with self.assertLogs(R.logger, level="WARNING") as cm:
            self._note(obj, 8_000, 8)
            self._note(obj, 8_000, 8)  # now 16 leaves -> warn
        self.assertTrue(getattr(obj, "_binding_full_coevict_warned"))
        self.assertEqual(len(cm.output), 1)
        # Further pressure must not warn again.
        with self.assertRaises(AssertionError):
            with self.assertLogs(R.logger, level="WARNING"):
                self._note(obj, 8_000, 8)

    def test_recommended_avg_matches_observed(self):
        obj = self._fresh()
        # 16 leaves, 16*1000 tokens -> avg 1000 tokens/prefix, recommend "1000".
        with self.assertLogs(R.logger, level="WARNING") as cm:
            self._note(obj, 16_000, 16)
        self.assertTrue(any("1000" in m for m in cm.output))


class _Node:
    def __init__(self, name, prio, full, swa, parent=None):
        self.name = name
        self.prio = prio
        self.full = full
        self.swa = swa
        self.parent = parent

    def __repr__(self):
        return f"_Node({self.name})"


class _FakeCacheForLeafEvict:
    """Minimal stand-in exercising drive_host_leaf_eviction's traversal and
    accounting. _evict_host_leaf models atomic Full+SWA drop and exposes the
    parent as a new host leaf (walk-up)."""

    def __init__(self, leaves):
        self.evictable_host_leaves = set(leaves)
        self.eviction_strategy = types.SimpleNamespace(get_priority=lambda n: n.prio)
        self.evicted = []
        self.coevict_calls = []

    def _evict_host_leaf(self, x, tracker):
        self.evictable_host_leaves.discard(x)
        tracker[FULL] = tracker.get(FULL, 0) + x.full
        tracker[SWA] = tracker.get(SWA, 0) + x.swa
        self.evicted.append(x)
        if x.parent is not None:
            # Parent becomes a host leaf now that its child is gone.
            self.evictable_host_leaves.add(x.parent)

    def _note_binding_full_coevict(self, full_tokens, leaves):
        self.coevict_calls.append((full_tokens, leaves))


class TestDriveHostLeafEviction(unittest.TestCase):
    def _drive(self, cache, num_tokens, key, tracker):
        R.UnifiedRadixCache.drive_host_leaf_eviction(cache, num_tokens, key, tracker)

    def test_priority_order_and_stop(self):
        # Two independent leaves; lower priority (LRU) evicted first, stop once
        # the key component target is met -> the colder leaf is spared.
        a = _Node("a", prio=1, full=10, swa=10)  # colder -> evicted first
        b = _Node("b", prio=5, full=10, swa=10)
        cache = _FakeCacheForLeafEvict([a, b])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=10, key=SWA, tracker=tracker)
        self.assertEqual(cache.evicted, [a])
        self.assertIn(b, cache.evictable_host_leaves)

    def test_walk_up_parents(self):
        # Chain c <- b <- a (a is the only initial leaf); freeing pulls up the
        # whole branch as each parent becomes a leaf.
        c = _Node("c", prio=3, full=5, swa=5)
        b = _Node("b", prio=2, full=5, swa=5, parent=c)
        a = _Node("a", prio=1, full=5, swa=5, parent=b)
        cache = _FakeCacheForLeafEvict([a])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=15, key=SWA, tracker=tracker)
        self.assertEqual(cache.evicted, [a, b, c])
        self.assertEqual(tracker[SWA], 15)

    def test_stale_entries_skipped(self):
        # Evicting a also removes sibling b from the evictable set (collapsed);
        # b is then popped-but-stale and skipped, so it is not counted.
        a = _Node("a", prio=1, full=10, swa=10)
        b = _Node("b", prio=2, full=10, swa=10)
        cache = _FakeCacheForLeafEvict([a, b])
        orig = cache._evict_host_leaf

        def evict_and_collapse(x, tracker):
            orig(x, tracker)
            cache.evictable_host_leaves.discard(b)

        cache._evict_host_leaf = evict_and_collapse
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=100, key=SWA, tracker=tracker)
        self.assertEqual(cache.evicted, [a])

    def test_coevict_recorded_for_aux_component(self):
        a = _Node("a", prio=1, full=7, swa=7)
        cache = _FakeCacheForLeafEvict([a])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=7, key=SWA, tracker=tracker)
        self.assertEqual(cache.coevict_calls, [(7, 1)])  # full freed, 1 leaf

    def test_no_coevict_note_for_full_key(self):
        # When Full itself is the driver there is no auxiliary binding pressure.
        a = _Node("a", prio=1, full=7, swa=7)
        cache = _FakeCacheForLeafEvict([a])
        tracker = {FULL: 0, SWA: 0}
        self._drive(cache, num_tokens=7, key=FULL, tracker=tracker)
        self.assertEqual(cache.coevict_calls, [])


class TestSwaComponentRouting(unittest.TestCase):
    """drive_host_eviction must route to atomic leaf eviction iff strict."""

    def _fake_component(self, strict):
        comp = types.SimpleNamespace()
        comp._strict_bit_exact = strict
        comp.component_type = SWA
        calls = {"leaf": [], "lru_get": 0}

        class _Cache:
            def drive_host_leaf_eviction(self, num_tokens, ct, tracker):
                calls["leaf"].append((num_tokens, ct))

            @property
            def host_lru_lists(self):
                calls["lru_get"] += 1

                class _L:
                    def get_lru_no_host_lock(_s):
                        return None

                return {SWA: _L()}

        comp.cache = _Cache()
        return comp, calls

    def _drive(self, comp, tracker):
        SWAComponent.drive_host_eviction(comp, 100, tracker)

    def test_strict_routes_to_leaf_eviction(self):
        comp, calls = self._fake_component(strict=True)
        self._drive(comp, {SWA: 0})
        self.assertEqual(calls["leaf"], [(100, SWA)])
        self.assertEqual(calls["lru_get"], 0)  # never touches the tombstone path

    def test_non_strict_uses_lru_path(self):
        comp, calls = self._fake_component(strict=False)
        self._drive(comp, {SWA: 0})
        self.assertEqual(calls["leaf"], [])
        self.assertGreaterEqual(calls["lru_get"], 1)


class TestWriteBackGuard(unittest.TestCase):
    """Strict bit-exact must fail fast at build() entry if write policy is not
    write_through (write_back can leave the SWA ring un-offloaded -> silent
    non-bit-exact reuse)."""

    import unittest.mock as _mock

    def _build(self, *, unified, write_policy, flag):
        strategy = A._DeepSeekV4Strategy()
        kv = types.SimpleNamespace(_unified_kv=unified)
        sa = types.SimpleNamespace(hicache_write_policy=write_policy)
        flag_obj = types.SimpleNamespace(get=lambda: flag)
        with self._mock.patch.object(
            A.envs, "SGLANG_UNIFIED_KV_BIT_EXACT_HICACHE", flag_obj
        ):
            strategy.build(
                cache=None,
                kvcache=kv,
                params=None,
                server_args=sa,
                load_cache_event=None,
            )

    def test_write_back_trips_guard(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(unified=True, write_policy="write_back", flag=True)
        self.assertIn("write_through", str(ctx.exception))

    def test_write_through_passes_guard(self):
        # Passes the guard, then fails later on the None collaborators -- we only
        # assert it is NOT the guard ValueError.
        with self.assertRaises(Exception) as ctx:
            self._build(unified=True, write_policy="write_through", flag=True)
        self.assertNotIn("requires --hicache-write-policy", str(ctx.exception))

    def test_flag_off_no_guard(self):
        with self.assertRaises(Exception) as ctx:
            self._build(unified=True, write_policy="write_back", flag=False)
        self.assertNotIn("requires --hicache-write-policy", str(ctx.exception))

    def test_non_unified_no_guard(self):
        with self.assertRaises(Exception) as ctx:
            self._build(unified=False, write_policy="write_back", flag=True)
        self.assertNotIn("requires --hicache-write-policy", str(ctx.exception))


class TestSwaRegionBuffers(unittest.TestCase):
    """The SWA-ring host pool must be page-granular with the sliding window as
    the page unit, so each indexed device row is exactly one host item_bytes.
    Row-granular device buffers (head_dim rows) declared with a page-granular
    item_bytes mismatch in transfer_kv_direct and crash."""

    def _fake_pool(self, *, num_slots, ring_size, head_dim, compress_rows, layers):
        import torch

        swa_pages = num_slots * ring_size
        rows = swa_pages + compress_rows
        kv_buffer = [
            torch.arange(rows * head_dim, dtype=torch.bfloat16).reshape(rows, head_dim)
            for _ in range(layers)
        ]
        unified_kv_pool = types.SimpleNamespace(
            swa_pages=swa_pages, head_dim=head_dim, kv_buffer=kv_buffer
        )
        return types.SimpleNamespace(
            _unified_kv=True,
            unified_swa_ring_size=ring_size,
            unified_kv_pool=unified_kv_pool,
        )

    def test_page_granular_geometry(self):
        import torch

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        ring_size, head_dim, num_slots, layers = 2, 4, 3, 2
        pool = self._fake_pool(
            num_slots=num_slots,
            ring_size=ring_size,
            head_dim=head_dim,
            compress_rows=8,
            layers=layers,
        )
        views, item_bytes = P.swa_region_buffers(pool)
        # one page == one sliding window == ring_size rows (bf16 = 2 bytes).
        self.assertEqual(item_bytes, ring_size * head_dim * 2)
        self.assertEqual(len(views), layers)
        for v in views:
            self.assertEqual(v.dtype, torch.uint8)
            # num_pages == swa_pages // ring_size == num_slots; row width == item_bytes.
            self.assertEqual(v.shape, (num_slots, item_bytes))
            self.assertEqual(v[0].nbytes, item_bytes)

    def test_view_preserves_ring_data(self):
        import torch

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        ring_size, head_dim = 2, 4
        pool = self._fake_pool(
            num_slots=3,
            ring_size=ring_size,
            head_dim=head_dim,
            compress_rows=8,
            layers=1,
        )
        views, _ = P.swa_region_buffers(pool)
        buf = pool.unified_kv_pool.kv_buffer[0]
        # host page 1 must be ring rows [ring_size, 2*ring_size) byte-identical.
        expected = buf.narrow(0, ring_size, ring_size).reshape(-1).view(torch.uint8)
        self.assertTrue(torch.equal(views[0][1], expected))

    def test_rejects_non_unified(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        pool = types.SimpleNamespace(_unified_kv=False)
        with self.assertRaises(AssertionError):
            P.swa_region_buffers(pool)

    def test_rejects_non_divisible_ring(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        # swa_pages must be a whole number of windows; a corrupt pool trips the guard.
        pool = self._fake_pool(
            num_slots=3, ring_size=2, head_dim=4, compress_rows=8, layers=1
        )
        pool.unified_kv_pool.swa_pages += 1  # 7, not a multiple of ring_size=2
        with self.assertRaises(AssertionError):
            P.swa_region_buffers(pool)

    def test_all_pages_map_to_their_window(self):
        import torch

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        ring_size, head_dim, num_slots = 2, 4, 5
        pool = self._fake_pool(
            num_slots=num_slots,
            ring_size=ring_size,
            head_dim=head_dim,
            compress_rows=6,
            layers=2,
        )
        views, _ = P.swa_region_buffers(pool)
        for layer, view in enumerate(views):
            buf = pool.unified_kv_pool.kv_buffer[layer]
            self.assertEqual(view.shape[0], num_slots)  # one page per window
            for page in range(num_slots):
                # page p must be exactly ring rows [p*ring, (p+1)*ring), byte-identical.
                expected = (
                    buf.narrow(0, page * ring_size, ring_size)
                    .reshape(-1)
                    .view(torch.uint8)
                )
                self.assertTrue(torch.equal(view[page], expected))


class TestSwaRingRegionDelegation(unittest.TestCase):
    """The assembler seam must delegate SWA-ring buffer resolution to the pool
    (which owns the unified_kv layout) and never re-derive geometry itself."""

    def test_delegates_to_pool(self):
        sentinel = (["buf0", "buf1"], 131072)
        kvcache = types.SimpleNamespace(
            _unified_kv=True, swa_region_buffers=lambda: sentinel
        )
        self.assertIs(A._dsv4_swa_ring_region_buffers(kvcache), sentinel)

    def test_rejects_non_unified(self):
        kvcache = types.SimpleNamespace(_unified_kv=False)
        with self.assertRaises(AssertionError):
            A._dsv4_swa_ring_region_buffers(kvcache)


class TestStrictL3Coupled(unittest.TestCase):
    """I4' (L3-coupled): strict bit-exact SWA HiCache persists each captured
    window to L3 under its carrier node's Full page hash (keys=[hash_value[-1]]),
    so the SWA window and its Full page share one L3 lifetime (SWA-L3 =>
    Full-L3). A window page == slot_page_size tokens."""

    PAGE_SIZE = 256
    RING = 256  # slot_page_size == swa_ring_size (one window page)

    def _comp(self, strict):
        comp = types.SimpleNamespace()
        comp._strict_bit_exact = strict
        comp.component_type = SWA
        comp._swa_kv_pool_host = types.SimpleNamespace(slot_page_size=self.RING)
        comp.cache = types.SimpleNamespace(
            cache_controller=object(), page_size=self.PAGE_SIZE
        )
        # bind the real helper so the test exercises _swa_l3_key (not a mock).
        comp._swa_l3_key = types.MethodType(SWAComponent._swa_l3_key, comp)
        return comp

    def _carrier_node(self):
        # exactly one captured window page on the carrier, with a Full page hash.
        cd = types.SimpleNamespace(host_value=list(range(self.RING)), value=None)
        return types.SimpleNamespace(component_data={SWA: cd}, hash_value=["h0", "h1"])

    def _build_storage(self, comp, node):
        return SWAComponent.build_hicache_transfers(
            comp, node, CacheTransferPhase.BACKUP_STORAGE
        )

    def test_strict_persists_window_coupled_to_full_hash(self):
        transfers = self._build_storage(self._comp(strict=True), self._carrier_node())
        self.assertIsNotNone(transfers)
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].name, PoolName.SWA)
        self.assertEqual(transfers[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        # Full-coupled key: the carrier's own Full page hash, not a suffix.
        self.assertEqual(transfers[0].keys, ["h1"])

    def test_strict_no_window_emits_nothing(self):
        comp = self._comp(strict=True)
        node = self._carrier_node()
        node.component_data[SWA].host_value = None
        self.assertIsNone(self._build_storage(comp, node))

    def _prefetch_comp(self, ring, sliding_window):
        import torch as _t

        comp = self._comp(strict=True)
        comp.sliding_window_size = sliding_window
        comp._swa_kv_pool_host = types.SimpleNamespace(
            slot_page_size=ring,
            alloc=lambda n: _t.arange(n, dtype=_t.int64),
        )
        comp.cache = types.SimpleNamespace(
            cache_controller=object(),
            page_size=self.PAGE_SIZE,
            evict_host=lambda *a, **k: None,
        )
        return comp

    def _build_prefetch(self, comp, prefetch_tokens):
        # `node` is the matched-prefix anchor (empty after a flush); the strict
        # branch must NOT key off it -- it emits placeholders the controller
        # later rewrites to real hit-range hashes.
        node = types.SimpleNamespace(
            component_data={SWA: types.SimpleNamespace(host_value=None, value=None)},
            hash_value=[],
        )
        return SWAComponent.build_hicache_transfers(
            comp,
            node,
            CacheTransferPhase.PREFETCH,
            token_ids=list(range(prefetch_tokens)),
            prefetch_tokens=prefetch_tokens,
        )

    def test_strict_prefetch_emits_placeholder_trailing_window(self):
        # ring == page_size: a window spans ceil(sliding_window / ring)
        # *contiguous* carrier pages; each is a placeholder for the controller.
        comp = self._prefetch_comp(
            ring=self.PAGE_SIZE, sliding_window=2 * self.PAGE_SIZE
        )
        transfers = self._build_prefetch(comp, prefetch_tokens=8 * self.PAGE_SIZE)
        self.assertIsNotNone(transfers)
        self.assertEqual(transfers[0].name, PoolName.SWA)
        self.assertEqual(transfers[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        # placeholders only -- never the anchor's matched-prefix hashes.
        self.assertEqual(transfers[0].keys, ["__placeholder__", "__placeholder__"])
        self.assertEqual(transfers[0].host_indices.numel(), 2 * self.PAGE_SIZE)

    def test_strict_prefetch_unified_single_window(self):
        # unified_kv: ring == sliding_window -> exactly one window page, because
        # _sync_trailing_keys can only map one contiguous key onto a ring-spaced
        # carrier. More than one placeholder would fetch garbage.
        ring = 4 * self.PAGE_SIZE
        comp = self._prefetch_comp(ring=ring, sliding_window=ring)
        transfers = self._build_prefetch(comp, prefetch_tokens=8 * self.PAGE_SIZE)
        self.assertIsNotNone(transfers)
        self.assertEqual(transfers[0].keys, ["__placeholder__"])
        self.assertEqual(transfers[0].host_indices.numel(), ring)

    def test_strict_prefetch_skips_when_region_shorter_than_ring(self):
        ring = 4 * self.PAGE_SIZE
        comp = self._prefetch_comp(ring=ring, sliding_window=ring)
        # only 2 pages prefetched, ring needs 4 -> not worth a partial window.
        self.assertIsNone(
            self._build_prefetch(comp, prefetch_tokens=2 * self.PAGE_SIZE)
        )

    def test_prefetch_placeholders_resolve_to_carrier_backup_keys(self):
        # End-to-end key contract: the placeholders a strict PREFETCH emits are
        # rewritten by the controller's _sync_trailing_keys to the SAME Full page
        # hashes a strict BACKUP_STORAGE would have stored the window under
        # (carrier hash_value[-1]). Proven directly against the real controller
        # helper, not a re-implementation.
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        comp = self._prefetch_comp(
            ring=self.PAGE_SIZE, sliding_window=2 * self.PAGE_SIZE
        )
        transfers = self._build_prefetch(comp, prefetch_tokens=8 * self.PAGE_SIZE)

        # Storage hit range: 8 Full pages fully hit; _sync_trailing_keys must
        # rewrite the 2 placeholders to the last 2 hit-page hashes (the window).
        all_hashes = [f"ph{i}" for i in range(8)]
        kv_hit_pages = 8
        HybridCacheController._sync_trailing_keys(
            None, transfers, all_hashes, kv_hit_pages
        )
        self.assertEqual(transfers[0].keys, ["ph6", "ph7"])
        # The tail carrier BACKUP_STORAGE key for the same page is hash_value[-1]
        # == "ph7"; the trailing window keys end on it (co-lifetime holds).
        self.assertEqual(transfers[0].keys[-1], all_hashes[kv_hit_pages - 1])

    def test_prefetch_placeholders_truncate_on_short_hit(self):
        # If storage hit is shorter than the target, keys realign to the tail of
        # the *actual* hit range (never point past it into evicted pages).
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        comp = self._prefetch_comp(
            ring=self.PAGE_SIZE, sliding_window=2 * self.PAGE_SIZE
        )
        transfers = self._build_prefetch(comp, prefetch_tokens=8 * self.PAGE_SIZE)
        all_hashes = [f"ph{i}" for i in range(8)]
        HybridCacheController._sync_trailing_keys(None, transfers, all_hashes, 3)
        self.assertEqual(transfers[0].keys, ["ph1", "ph2"])


if __name__ == "__main__":
    unittest.main()


class TestEvictDeviceOnOwnerRelease(unittest.TestCase):
    """SWAComponent.evict_device_on_owner_release: once the owning request has
    finished (SWA lock_ref==0) and the host copy is committed, the per-request
    device SWA ring value must be tombstoned so cross-request reuse restores
    from host (I1) rather than trusting the recycled device ring. Gated so it
    never fires while another request still holds the SWA lock (sanity_check
    forbids value=None with lock_ref>0) and never destroys the only copy."""

    def _node(self, *, value, host_value, lock_ref):
        cd = types.SimpleNamespace(
            value=value, host_value=host_value, lock_ref=lock_ref
        )
        return types.SimpleNamespace(component_data={SWA: cd}), cd

    def _comp(self, *, strict=True, host_pool=object()):
        comp = types.SimpleNamespace()
        comp._strict_bit_exact = strict
        comp._swa_kv_pool_host = host_pool
        comp.component_type = SWA
        calls = []

        class _Cache:
            def _evict_component_and_detach_lru(_s, node, c, target=None):
                calls.append((node, c, target))

        comp.cache = _Cache()
        return comp, calls

    def _run(self, comp, node):
        SWAComponent.evict_device_on_owner_release(comp, node)

    def test_released_and_backed_up_tombstones_device(self):
        comp, calls = self._comp()
        node, _ = self._node(value=[1, 2], host_value=[9], lock_ref=0)
        self._run(comp, node)
        self.assertEqual(len(calls), 1)
        got_node, got_comp, target = calls[0]
        self.assertIs(got_node, node)
        self.assertIs(got_comp, comp)
        self.assertEqual(target, EvictLayer.DEVICE)

    def test_still_locked_is_left_intact(self):
        # Another active request holds the SWA lock -> must not tombstone
        # (device value=None with lock_ref>0 would trip sanity_check).
        comp, calls = self._comp()
        node, _ = self._node(value=[1, 2], host_value=[9], lock_ref=1)
        self._run(comp, node)
        self.assertEqual(calls, [])

    def test_no_host_copy_is_left_intact(self):
        # host_value not committed yet: dropping device now would force I6
        # recompute AND (if only pending) risk co-lifetime races -> keep device.
        comp, calls = self._comp()
        node, _ = self._node(value=[1, 2], host_value=None, lock_ref=0)
        self._run(comp, node)
        self.assertEqual(calls, [])

    def test_already_device_absent_is_noop(self):
        comp, calls = self._comp()
        node, _ = self._node(value=None, host_value=[9], lock_ref=0)
        self._run(comp, node)
        self.assertEqual(calls, [])

    def test_non_strict_is_noop(self):
        comp, calls = self._comp(strict=False)
        node, _ = self._node(value=[1, 2], host_value=[9], lock_ref=0)
        self._run(comp, node)
        self.assertEqual(calls, [])

    def test_feature_off_when_host_pool_unwired(self):
        comp, calls = self._comp(host_pool=None)
        node, _ = self._node(value=[1, 2], host_value=[9], lock_ref=0)
        self._run(comp, node)
        self.assertEqual(calls, [])


class TestSwaL3RoundTrip(unittest.TestCase):
    """B2 component-level round trip: a strict window persisted via
    BACKUP_STORAGE is re-attached to its carrier tombstone by _commit_prefetch,
    then device landing is delegated to the existing positional restore.

    RING != PAGE_SIZE on purpose: a window is ring-paged (one item == one
    window == ring tokens), so _commit_prefetch must count window_require_pages
    with stride=ring, not page_size. With page_size math the single-window hit
    (loaded_pages=1) would look like 2 page-pages required and drop the window.
    """

    PAGE_SIZE = 256
    RING = 512  # one window == 2 kv pages worth of tokens

    def _comp(self):
        comp = types.SimpleNamespace()
        comp._strict_bit_exact = True
        comp.component_type = SWA
        comp._swa_kv_pool_host = types.SimpleNamespace(slot_page_size=self.RING)
        attached = []
        comp._attach_swa_host_value = lambda n, s: attached.append(n)
        comp._release_swa_host = lambda s: None
        comp.cache = types.SimpleNamespace(
            page_size=self.PAGE_SIZE,
            _split_node=lambda *a, **k: None,
        )
        comp._attached = attached
        return comp

    def test_commit_prefetch_reattaches_window_to_carrier(self):
        import torch as _t

        comp = self._comp()
        carrier = types.SimpleNamespace(
            component_data={SWA: types.SimpleNamespace(host_value=None, value=None)},
            parent=None,
        )
        carrier.key = list(range(self.RING))  # len(key) == one window in tokens
        transfers = [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=_t.arange(self.RING, dtype=_t.int64),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        insert_result = types.SimpleNamespace(
            inserted_host_node=carrier, total_len=self.RING
        )
        # one window (RING=512) covers stride//page = 2 Full pages -> the
        # coupling guard needs kv_hit_pages >= 2 for the window to attach.
        pool_storage_result = types.SimpleNamespace(
            extra_pool_hit_pages={PoolName.SWA: 1}, kv_hit_pages=2
        )
        SWAComponent._commit_prefetch(
            comp,
            anchor=None,
            transfers=transfers,
            insert_result=insert_result,
            pool_storage_result=pool_storage_result,
        )
        # SimpleNamespace is unhashable (defines __eq__); check by identity.
        self.assertTrue(any(n is carrier for n in comp._attached))

    def test_commit_prefetch_drops_window_when_full_evicted(self):
        # I4' coupling guard: SWA window hit but its Full pages were evicted
        # (kv_hit_pages < window's Full-page span) -> the window must be dropped
        # (never attached), falling back to recompute.
        import torch as _t

        comp = self._comp()
        carrier = types.SimpleNamespace(
            component_data={SWA: types.SimpleNamespace(host_value=None, value=None)},
            parent=None,
        )
        carrier.key = list(range(self.RING))
        transfers = [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=_t.arange(self.RING, dtype=_t.int64),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        insert_result = types.SimpleNamespace(
            inserted_host_node=carrier, total_len=self.RING
        )
        # window needs 2 Full pages but only 1 hit -> guard drops it.
        pool_storage_result = types.SimpleNamespace(
            extra_pool_hit_pages={PoolName.SWA: 1}, kv_hit_pages=1
        )
        SWAComponent._commit_prefetch(
            comp,
            anchor=None,
            transfers=transfers,
            insert_result=insert_result,
            pool_storage_result=pool_storage_result,
        )
        self.assertFalse(any(n is carrier for n in comp._attached))
