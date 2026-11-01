# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tri-pool chain (`ChainAllocator` over full, swa and mamba) -- full KV +
SWA KV + mamba/conv state in ONE unified byte buffer, chain
``[mamba (up END) | swa (FLOAT) | full (down END)]``.

Pure CPU; fakes stand in for the KV pools (data markers verify moves).
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    kv_mamba_slots,
)
from sglang.srt.mem_cache.base_prefix_cache import AllocDemand
from sglang.srt.mem_cache.kv.alloc.band import (
    Band,
    FloatBand,
)
from sglang.srt.mem_cache.kv.alloc.chain import (
    ChainAllocator,
    ChainRoles,
    take_new_pages,
)
from sglang.srt.mem_cache.kv.alloc.pricing import FLUSH_ENDS, plan
from sglang.srt.mem_cache.kv.alloc.signals import wait_signals
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import (
    UnifiedMambaSlotAllocator,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.token_demand import demand_windows, token_demand
from sglang.test.tree_slots import index_slots
from sglang.test.unified_allocator_fixtures import (
    FakeKVCache,
    FakeUnifiedSWAKVPool,
    arena_of,
    build_test_unified_pools,
    open_float_side,
    setup_allocator_context,
    stand_in_views,
    tri_sub_pool_specs,
)

# Hermetic convention of this directory's pool tests: plain unittest.TestCase,
# only ci_register imported (no heavy sglang.test.test_utils chain).
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"

# The owner page take behind the paged extend; stubbed so these CPU tests
# never launch the GPU paging kernel.
_TAKE_NEW_PAGES = "sglang.srt.mem_cache.kv.alloc.chain.take_new_pages"


def _take_new_pages_on_host(owner, n, page_tokens, **kwargs):
    """The owner page take with the extend kernel replaced by the token ids of
    the new pages (an empty prefix), so the planned physical pages still bind."""
    return take_new_pages(
        owner, n, lambda free: owner._expand_pages_to_tokens(free[:n]), **kwargs
    )


class _PoolTestCase(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import reset_context

        setup_allocator_context()
        self.addCleanup(reset_context)


class TestUnifiedTriPool(_PoolTestCase):
    def _build(
        self,
        n_full=32,
        n_swa=16,
        n_state=8,
        page_size=1,
        lazy_compaction=False,
    ):
        full, swa, mamba = tri_sub_pool_specs()
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = arena_of(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        mamba_kv = FakeKVCache(pool, "mamba")
        allocator = ChainAllocator(
            roles=ChainRoles(owner="full", window="swa", state="mamba"),
            arena=pool,
            kvcache=kvcache,
            views=stand_in_views(
                ChainRoles(owner="full", window="swa", state="mamba"), kvcache, mamba_kv
            ),
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            page_size=page_size,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy_compaction,
        )
        return pool, allocator, kvcache, mamba_kv

    def _stamp(self, allocator, kvcache, v):
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        kvcache.full_kv_pool.buf[fa.ids.virtual_to_physical[v]] = v
        kvcache.swa_kv_pool.buf[sa.ids.virtual_to_physical[v]] = v

    # -- construction --

    def test_chain_wiring_and_float_swa(self):
        pool, allocator, kvcache, _ = self._build()
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        ma = allocator.mamba_allocator
        self.assertIsInstance(sa, FloatBand)
        self.assertIs(ma.high_peer, sa)
        self.assertIs(sa.low_peer, ma)
        self.assertIs(sa.high_peer, fa)
        self.assertIs(fa.low_peer, sa)
        # Canonical chain order in the pool.
        self.assertEqual([b.name for b in pool.bands], ["mamba", "swa", "full"])
        # KV pool got the allocators.

    def test_pd_preallocation_binds_only_swa_tail_pages(self):
        _, allocator, _, _ = self._build(page_size=4)
        before = allocator.available_size()
        prefix = torch.tensor([0], dtype=torch.int64)
        seq = torch.tensor([12], dtype=torch.int64)
        # With an empty prefix, ordinary allocation supplies the same virtual
        # pages without launching the GPU extend kernel. All page binding and
        # capacity accounting still run through the real sub-allocators.
        with patch(_TAKE_NEW_PAGES, side_effect=_take_new_pages_on_host):
            virtual = allocator.alloc_extend_swa_tail(
                prefix,
                prefix,
                seq,
                seq,
                torch.tensor([-1]),
                12,
                window_tail_lens=dict.fromkeys(allocator.window_pool_names(), 5),
            )
        self.assertIsNotNone(virtual)
        self.assertEqual(len(virtual), 12)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 12)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 8)
        swa_pages = allocator.swa_v2p_page_table[virtual[::4] // 4]
        self.assertLessEqual(swa_pages[0].item(), 0)
        self.assertTrue(torch.all(swa_pages[1:] > 0).item())
        allocator.free(virtual)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.available_size(), before)

    def test_pd_short_tail_fits_beyond_joint_capacity(self):
        for lazy in (False, True):
            for tail_len in (0, 5):
                with self.subTest(lazy=lazy, tail_len=tail_len):
                    _, allocator, _, _ = self._build(page_size=4, lazy_compaction=lazy)
                    full = allocator.full_attn_allocator
                    length = allocator.available_size() + 4
                    self.assertFalse(
                        allocator.admits(
                            length, demand_windows(allocator, length, length)
                        )
                    )
                    self.assertTrue(
                        allocator.admits(
                            length, demand_windows(allocator, length, tail_len)
                        )
                    )
                    prefix = torch.tensor([0], dtype=torch.int64)
                    seq = torch.tensor([length], dtype=torch.int64)
                    with patch(_TAKE_NEW_PAGES, side_effect=_take_new_pages_on_host):
                        virtual = allocator.alloc_extend_swa_tail(
                            prefix,
                            prefix,
                            seq,
                            seq,
                            torch.tensor([-1]),
                            length,
                            dict.fromkeys(allocator.window_pool_names(), tail_len),
                        )
                    self.assertIsNotNone(virtual)
                    self.assertEqual(full.allocated_count(), length)
                    self.assertEqual(
                        allocator.swa_attn_allocator.allocated_count(),
                        -(-tail_len // 4) * 4,
                    )
                    self.assertEqual(allocator.verify_byte_accounting(), [])
                    allocator.free(virtual)
                    self.assertEqual(full.allocated_count(), 0)
                    self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)

    def test_pd_tail_rejects_full_capacity_shortfall(self):
        _, allocator, _, _ = self._build(page_size=4)
        full = allocator.full_attn_allocator
        length = full.available_size() + 4
        prefix = torch.tensor([0], dtype=torch.int64)
        seq = torch.tensor([length], dtype=torch.int64)
        with patch(_TAKE_NEW_PAGES) as extend:
            self.assertIsNone(
                allocator.alloc_extend_swa_tail(
                    prefix,
                    prefix,
                    seq,
                    seq,
                    torch.tensor([-1]),
                    length,
                    dict.fromkeys(allocator.window_pool_names(), 0),
                )
            )
        extend.assert_not_called()
        self.assertEqual(full.allocated_count(), 0)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)

    def test_empty_float_is_transparent_to_the_ends(self):
        _, allocator, _, _ = self._build()
        fa = allocator.full_attn_allocator
        ma = allocator.mamba_allocator
        self.assertTrue(allocator.swa_attn_allocator._is_frontier_transparent())
        # full's chain gap reaches the mamba end's frontier straight through.
        self.assertEqual(
            fa._current_gap_bytes(),
            fa._byte_low_frontier() - ma._byte_high_frontier(),
        )

    # -- the joint availability contract --

    def test_available_shrinks_as_state_slots_grow(self):
        _, allocator, _, _ = self._build()
        before = allocator.available_size()
        slots = allocator.mamba_allocator.alloc(4)
        self.assertIsNotNone(slots)
        after = allocator.available_size()
        self.assertLess(after, before)
        allocator.mamba_allocator.free(slots)
        self.assertEqual(allocator.available_size(), before)

    # -- steady-state SWA churn: tombstones -> holes -> in-place reuse --

    def _swa_interior_block(self, allocator, blocks):
        """The block whose SWA-physical pages touch neither float boundary:
        `free_swa` on it holes; a boundary block would be absorbed instead."""
        sa = allocator.swa_attn_allocator
        for v in blocks:
            pages = set(int(x) for x in sa.ids.virtual_to_physical[v].tolist())
            if sa.low_wm_page not in pages and (sa.high_wm_page - 1) not in pages:
                return v
        raise AssertionError("no interior block in layout")

    def test_free_swa_holes_recycled_in_place_zero_copy(self):
        _, allocator, kvcache, _ = self._build()
        blocks = [allocator.alloc(4) for _ in range(3)]
        for v in blocks:
            self.assertIsNotNone(v)
            self._stamp(allocator, kvcache, v)
        sa = allocator.swa_attn_allocator
        fa = allocator.full_attn_allocator
        span_before = (sa.low_wm_page, sa.high_wm_page)

        # Window slide: an INTERIOR block ages out of the SWA window.
        v_mid = self._swa_interior_block(allocator, blocks)
        allocator.free_swa(v_mid)
        # The full side keeps the token; the swa side tombstoned it.
        self.assertTrue(bool((fa.ids.virtual_to_physical[v_mid] >= 0).all()))
        self.assertTrue(bool((sa.ids.virtual_to_physical[v_mid] == -1).all()))
        self.assertEqual(sa._hole_pages(), 4)
        self.assertEqual((sa.low_wm_page, sa.high_wm_page), span_before)

        # The next alloc recycles the holes IN PLACE: no span growth, no moves.
        vd = allocator.alloc(4)
        self.assertIsNotNone(vd)
        self._stamp(allocator, kvcache, vd)
        self.assertEqual(sa._hole_pages(), 0)
        self.assertEqual((sa.low_wm_page, sa.high_wm_page), span_before)
        self.assertEqual(len(sa._inverse_history), 0)  # zero copies

    def test_free_swa_boundary_block_absorbed_zero_copy(self):
        # A boundary block's tombstones shrink the span instead of holing it.
        # The shrink needs the hole set on the host, so it is deferred out of
        # the per-step free and lands in the opportunistic flush.
        _, allocator, kvcache, _ = self._build()
        blocks = [allocator.alloc(4) for _ in range(2)]
        for v in blocks:
            self._stamp(allocator, kvcache, v)
        sa = allocator.swa_attn_allocator
        span_pages = sa._span_pages()
        # Pick a block holding a span-boundary page.
        boundary = None
        for v in blocks:
            pages = set(int(x) for x in sa.ids.virtual_to_physical[v].tolist())
            if sa.low_wm_page in pages or (sa.high_wm_page - 1) in pages:
                boundary = v
                break
        self.assertIsNotNone(boundary)
        allocator.free_swa(boundary)
        allocator.flush_opportunistic()  # the deferred reclaim point
        self.assertEqual(sa._hole_pages(), 0)  # absorbed, not holed
        self.assertEqual(sa._span_pages(), span_pages - 4)
        self.assertEqual(len(sa._inverse_history), 0)  # zero copies

    def test_free_releases_both_sides_and_filters_tombstones(self):
        _, allocator, kvcache, _ = self._build()
        va = allocator.alloc(4)
        self._stamp(allocator, kvcache, va)
        allocator.free_swa(va)  # tombstone first (aged out of window)
        allocator.free(va)  # then the request finishes
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        self.assertTrue(bool((fa.ids.virtual_to_physical[va] == -1).all()))
        self.assertTrue(bool((sa.ids.virtual_to_physical[va] == -1).all()))
        # Fully-freed float parks and is transparent again.
        self.assertTrue(sa._is_frontier_transparent())

    # -- per-request state surface --

    def test_mamba_slot_allocator_surface(self):
        pool, allocator, _, mamba_kv = self._build()
        slot_alloc = UnifiedMambaSlotAllocator(
            allocator.mamba_allocator,
            device=_DEV,
            take=allocator.alloc_state_slots,
            capacity=allocator.state_slots_available,
        )
        v = slot_alloc.alloc(3)
        self.assertIsNotNone(v)
        p = slot_alloc.translate(v)
        self.assertTrue(bool((p >= 0).all()))
        mamba_kv.buf[p] = v
        self.assertEqual(
            slot_alloc.available_size(),
            (pool.num_entries("mamba") - 1) - 3,
        )
        slot_alloc.free(v)
        self.assertEqual(slot_alloc.available_size(), pool.num_entries("mamba") - 1)
        # Group prefetch draw-down + surplus return.
        slot_alloc.alloc_group_begin(4)
        s1 = slot_alloc.alloc(1)
        self.assertIsNotNone(s1)
        slot_alloc.alloc_group_end()
        self.assertEqual(
            slot_alloc.available_size(),
            (pool.num_entries("mamba") - 1) - 1,
        )

    # -- flush semantics --

    def test_urgent_flush_preserves_float_holes(self):
        _, allocator, kvcache, _ = self._build(lazy_compaction=True)
        blocks = [allocator.alloc(4) for _ in range(3)]
        for v in blocks:
            self._stamp(allocator, kvcache, v)
        allocator.free_swa(self._swa_interior_block(allocator, blocks))
        sa = allocator.swa_attn_allocator
        holes = sa._hole_pages()
        self.assertGreater(holes, 0)
        # A layout that compacts the ends leaves the float's holes in place.
        demands = {
            n: d
            for n, d in allocator.page_demands(token_demand(allocator, 1, 1)).items()
            if d
        }
        layout = plan(allocator.pricing_snapshot(), demands, FLUSH_ENDS)
        self.assertNotIn(sa.sub_pool_name, layout.flush)
        allocator._lay_out(layout)
        self.assertEqual(sa._hole_pages(), holes)  # holes are assets, not backlog

    def test_state_target_stops_after_float_exposes_state_capacity(self):
        for page_size, lazy_compaction in ((1, False), (4, True)):
            with self.subTest(page_size=page_size, lazy_compaction=lazy_compaction):
                _, allocator, _, _ = self._build(
                    page_size=page_size, lazy_compaction=lazy_compaction
                )
                # Wired as production wires it: the chain takes and prices slots.
                mamba_slots = UnifiedMambaSlotAllocator(
                    allocator.mamba_allocator,
                    device=_DEV,
                    take=allocator.alloc_state_slots,
                    capacity=allocator.state_slots_available,
                )

                full_leaves = []
                while True:
                    indices = allocator.alloc(allocator.page_size)
                    if indices is None:
                        break
                    full_leaves.append(indices)
                self.assertTrue(full_leaves)
                residual_mamba = mamba_slots.schedulable_available_size()
                if residual_mamba:
                    self.assertIsNotNone(mamba_slots.alloc(residual_mamba))
                while mamba_slots.alloc(1) is not None:
                    pass
                self.assertGreater(mamba_slots.available_size(), 0)
                self.assertEqual(mamba_slots.schedulable_available_size(), 0)

                cache = object.__new__(UnifiedRadixCache)
                cache.disable = False
                cache.tree_components = (
                    ComponentType.FULL,
                    ComponentType.SWA,
                    ComponentType.MAMBA,
                )
                index_slots(cache, cache.tree_components)
                cache.components = dict.fromkeys(cache.tree_components)
                cache.is_swa_enabled = True
                cache.is_mamba_enabled = True
                cache.cache_controller = None
                cache.metrics_collector = None
                cache.window_retain_size = MagicMock(return_value=0)
                allocator.state = SimpleNamespace(slots=mamba_slots)
                cache.token_to_kv_pool_allocator = allocator
                cache.req_to_token_pool = MagicMock()

                tree_core = MagicMock()
                tree_core.full_evictable_size.return_value = len(full_leaves)
                tree_core.mamba_evictable_size.return_value = 0
                tree_core.swa_evictable_size.return_value = 0
                tree_core.component_evictable_size.side_effect = lambda ct: (
                    len(full_leaves) if ct == ComponentType.FULL else 0
                )
                walk = {"request_cnt": 0, "freed_leaves": 0}

                def start(component_type, request_cnt):
                    self.assertEqual(component_type, ComponentType.FULL)
                    walk["request_cnt"] = request_cnt

                def next_node(component_type, tracker):
                    self.assertEqual(component_type, ComponentType.FULL)
                    if tracker[ComponentType.FULL] >= walk["request_cnt"] or walk[
                        "freed_leaves"
                    ] >= len(full_leaves):
                        return None, False
                    return walk["freed_leaves"] + 1, True

                def evict_leaf(node_id, tracker):
                    self.assertEqual(node_id, walk["freed_leaves"] + 1)
                    indices = full_leaves[-node_id]
                    tracker[ComponentType.FULL] += int(indices.numel())
                    allocator.full_attn_allocator.free(indices)
                    walk["freed_leaves"] += 1
                    return None

                tree_core.evict_device_start.side_effect = start
                cache.tree_core = tree_core
                cache._evict_device_next_node = MagicMock(side_effect=next_node)
                cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

                swa_live_before = allocator.swa_attn_allocator._live_pages()
                result = cache.evict_for_alloc(AllocDemand(mamba_num=1))

                self.assertEqual(walk["freed_leaves"], 1)
                self.assertEqual(result.num_tokens_evicted, page_size)
                self.assertEqual(
                    result.window_num_tokens_evicted.get(ComponentType.SWA, 0), 0
                )
                self.assertEqual(result.mamba_num_evicted, 0)
                self.assertEqual(
                    allocator.swa_attn_allocator._live_pages(), swa_live_before
                )
                self.assertGreaterEqual(mamba_slots.schedulable_available_size(), 1)
                self.assertIsNotNone(mamba_slots.alloc(1))
                self.assertEqual(allocator.verify_byte_accounting(), [])


class TestTriPagedFreeGroup(_PoolTestCase):
    """The tri composite at PAGE SIZE > 1, driven through the production free
    path: free_group_begin -> free_segment -> free_group_end.

    Regression: every other tri test runs at page_size=1, where the
    page-representative machinery is dead code. At ps>1 the composite releases
    reps via `swa_attn_allocator.free(..., _pages=...)`, which the float must
    accept or the first decode batch dies; honouring `_pages` is also what
    keeps the free path off the data-dependent `torch.unique` host sync.
    """

    def _build_paged(self, page_size=4, n_full=64, n_swa=32, n_state=8):
        full, swa, mamba = tri_sub_pool_specs()
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = arena_of(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        mamba_kv = FakeKVCache(pool, "mamba")
        allocator = ChainAllocator(
            roles=ChainRoles(owner="full", window="swa", state="mamba"),
            arena=pool,
            kvcache=kvcache,
            views=stand_in_views(
                ChainRoles(owner="full", window="swa", state="mamba"), kvcache, mamba_kv
            ),
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            page_size=page_size,
            need_sort=False,
            forward_stream=None,
        )
        return pool, allocator

    def test_free_group_segment_release_reaches_the_float(self):
        """The exact production sequence the scheduler runs per decode batch."""
        pool, allocator = self._build_paged()
        v = allocator.alloc(8)
        self.assertIsNotNone(v)
        before = allocator.available_size()

        allocator.free_group_begin()
        allocator.free_segment(v, start_pos=0)
        allocator.free_group_end()  # -> _release_page_reps -> float.free(_pages=)

        self.assertEqual(allocator.verify_byte_accounting(), [])
        self.assertGreaterEqual(allocator.available_size(), before)
        # Capacity fully recovered: the float parked, both ends rewound.
        self.assertTrue(allocator.swa_attn_allocator._is_frontier_transparent())

    def test_deferred_full_frees_flush_full_only_group_without_closing_it(self):
        _, allocator = self._build_paged(page_size=1)
        full_indices = allocator.alloc(8)
        self.assertIsNotNone(full_indices)
        allocator.free_swa(full_indices)
        allocated_before = allocator.full_attn_allocator.allocated_count()

        allocator.free_group_begin()
        allocator.free_full_segment(full_indices, start_pos=0)
        self.assertTrue(allocator.full_page_reps_group)

        allocator.flush_deferred_full_frees()

        self.assertEqual(allocator.free_group, [])
        self.assertEqual(allocator.free_page_reps_group, [])
        self.assertEqual(allocator.full_page_reps_group, [])
        self.assertEqual(allocator.full_free_group, [])
        self.assertLess(
            allocator.full_attn_allocator.allocated_count(), allocated_before
        )
        self.assertEqual(allocator.verify_byte_accounting(), [])
        allocator.free_group_end()

    def test_ungrouped_segment_free_also_reaches_the_float(self):
        pool, allocator = self._build_paged()
        v = allocator.alloc(8)
        self.assertIsNotNone(v)
        allocator.free_segment(v, start_pos=0)
        self.assertEqual(allocator.verify_byte_accounting(), [])


class TestTriFreeSwaNoHostSync(_PoolTestCase):
    """The tri's swa side is the FLOAT, which can never run the lazy event
    pipeline: unless the per-step frees carry caller-derived page ids, the tri
    reintroduces the host syncs the sync-free free path removed. Fixtures run
    at page_size > 1; ps==1 short-circuits the page machinery and hides this.
    """

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_ratchet_shape_free_swa_never_syncs_on_the_float(self):
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        self.assertIsNotNone(v)
        from unittest import mock

        with (
            mock.patch.object(
                torch, "unique", side_effect=AssertionError("unique = host sync")
            ),
            mock.patch.object(
                torch.Tensor, "item", side_effect=AssertionError("item = host sync")
            ),
        ):
            alloc.free_swa_segment(v[: 4 * self.PS], start_pos=0)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_fallback_free_swa_still_correct_for_radix_shapes(self):
        """Radix eviction hands arbitrary node values (no start_pos): the
        dedup fallback must end in the same state as the stride path."""
        a1, a2 = self._tri(), self._tri()
        v1, v2 = a1.alloc(6 * self.PS), a2.alloc(6 * self.PS)
        self.assertTrue(torch.equal(v1, v2))
        a1.free_swa_segment(v1[: 4 * self.PS], start_pos=0)
        a2.free_swa(v2[: 4 * self.PS])
        self.assertTrue(
            torch.equal(
                a1.swa_attn_allocator.ids.virtual_to_physical,
                a2.swa_attn_allocator.ids.virtual_to_physical,
            )
        )
        self.assertEqual(a1.available_size(), a2.available_size())
        self.assertEqual(a1.verify_byte_accounting(), [])
        self.assertEqual(a2.verify_byte_accounting(), [])


class TestShortEndRelief(_PoolTestCase):
    """A band short on its own take has its chain lay itself out for it: the
    float yields to whichever end is short, and nothing moves when no layout
    could serve the take."""

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_the_float_yields_to_either_short_end(self):
        """Either end, short where it stands, has the float move away from it."""
        for side in ("low", "high"):
            with self.subTest(side=side):
                alloc = self._tri()
                flt = alloc.swa_attn_allocator
                self.assertIsNotNone(alloc.alloc(8 * self.PS))
                end = (
                    alloc.mamba_allocator
                    if side == "low"
                    else alloc.full_attn_allocator
                )
                fit = end.available_size() // end.page_size
                if fit:
                    self.assertIsNotNone(end.alloc(fit * end.page_size))
                before = flt.low_wm_page if side == "low" else flt.high_wm_page
                self.assertIsNotNone(end.alloc(end.page_size))
                if side == "low":
                    self.assertGreater(flt.low_wm_page, before)
                else:
                    self.assertLess(flt.high_wm_page, before)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_a_take_beyond_any_layout_moves_nothing(self):
        """When the band's own index space binds, no layout serves the take:
        it is refused and no page moves."""
        alloc = self._tri()
        self.assertIsNotNone(alloc.alloc(4 * self.PS))
        ma = alloc.mamba_allocator
        sa = alloc.swa_attn_allocator
        huge = (ma.num_pages + 10) * ma.page_size  # beyond index space
        with patch.object(
            sa, "relocate_into", side_effect=AssertionError("useless move")
        ):
            self.assertIsNone(ma.alloc(huge))


class TestFloatPolicyTotalTarget(_PoolTestCase):
    """A short end's relief serves its whole take, not the part beyond the
    gap it has.

    Regression: an ask shaped as `deficit + one page` landed BELOW the current
    gap when the band was only PARTIALLY free, so the float did not move and
    the allocation failed though the float had room to slide.
    """

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_partial_gap_state_alloc_still_succeeds(self):
        alloc = self._tri()
        v = alloc.alloc(6 * self.PS)
        self.assertIsNotNone(v)
        ma, sa = alloc.mamba_allocator, alloc.swa_attn_allocator
        e_m = ma.entry_bytes_per_page
        gap_slots = int((sa._byte_low_frontier() - ma._byte_high_frontier()) // e_m)
        self.assertGreater(gap_slots, 2)
        low_before = sa.low_wm_page
        # partial-gap + 3: a delta-shaped ask lands below the current gap.
        got = ma.alloc((gap_slots + 3) * ma.page_size)
        self.assertIsNotNone(got, "partial-gap shortfall must relocate, not fail")
        self.assertGreater(sa.low_wm_page, low_before)
        self.assertEqual(alloc.verify_byte_accounting(), [])


class TestTriDeferredAbsorption(_PoolTestCase):
    """Boundary absorption is deferred out of the per-step free and paid once
    at a quiescent point. These pin WHERE it is paid, and that skipping it
    stays merely conservative."""

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_per_step_flush_reclaims_the_span(self):
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        sa = alloc.swa_attn_allocator
        span = sa._span_pages()
        alloc.free_swa_segment(v[6 * self.PS :], start_pos=6 * self.PS)  # high edge
        self.assertGreater(sa._hole_pages(), 0)  # deferred
        self.assertEqual(sa._span_pages(), span)
        moved = alloc.flush_opportunistic()
        self.assertGreater(moved, 0)
        self.assertLess(sa._span_pages(), span)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_deferral_is_conservative_never_over_reports(self):
        """Availability with a stale-wide span must never EXCEED the absorbed
        value -- under-reporting is safe, over-reporting would over-admit."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        alloc.free_swa_segment(v[6 * self.PS :], start_pos=6 * self.PS)
        deferred = alloc.available_size()
        alloc.swa_attn_allocator._flush(urgent=False)
        absorbed = alloc.available_size()
        self.assertLessEqual(deferred, absorbed)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_clean_flush_skips_the_d2h_entirely(self):
        """Only `free` can put a hole ON a boundary (alloc DRAINS holes into
        live pages, extension adds live pages), so with nothing freed since
        the last absorb the walk provably finds nothing and must not pay the
        D2H."""
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        alloc.free_swa_segment(v[2 * self.PS : 4 * self.PS], start_pos=2 * self.PS)
        alloc.flush_opportunistic()  # consumes the dirty flag
        sa = alloc.swa_attn_allocator
        self.assertGreater(sa._hole_pages(), 0)  # interior holes remain
        with mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("tolist = D2H")
        ):
            self.assertEqual(alloc.flush_opportunistic(), 0)
            self.assertEqual(sa._flush(urgent=False), 0)

    def test_alloc_between_frees_cannot_hide_a_boundary_hole(self):
        """An alloc drains holes and can restore a previously-seen hole
        COUNT, so the flag must be armed by `free`, not read off `numel()`."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        sa = alloc.swa_attn_allocator
        alloc.free_swa_segment(v[: 2 * self.PS], start_pos=0)  # low-edge holes
        n_after_free = sa._hole_pages()
        alloc.alloc(2 * self.PS)  # drains them back to live
        alloc.free_swa_segment(v[6 * self.PS :], start_pos=6 * self.PS)  # high edge
        self.assertEqual(sa._hole_pages(), n_after_free)  # same COUNT as before
        span = sa._span_pages()
        self.assertGreater(alloc.flush_opportunistic(), 0)  # still absorbed
        self.assertLess(sa._span_pages(), span)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_transparency_still_exact_without_absorption(self):
        """Park-on-empty stays in `free` because it is sync-free -- a float
        that empties must go transparent immediately, with no flush needed."""
        alloc = self._tri()
        v = alloc.alloc(4 * self.PS)
        sa = alloc.swa_attn_allocator
        self.assertFalse(sa._is_frontier_transparent())
        from unittest import mock

        with mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("tolist = D2H")
        ):
            alloc.free_swa_segment(v, start_pos=0)
        self.assertTrue(sa._is_frontier_transparent())
        self.assertEqual(sa._hole_pages(), 0)


class TestTriFactorySizing(_PoolTestCase):
    """Factory-level contracts: byte-budget sizing, the bs=1 feasibility
    floor, and the boot signature the GPU harness greps for (3 sub-pools,
    swa grow=float)."""

    def _factory_kwargs(self, **over):
        import types

        cp = types.SimpleNamespace(
            shape=types.SimpleNamespace(conv=[(3, 8)], temporal=(0, 0, 0)),
            dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0, 1],
        )
        kw = dict(
            device=_DEV,
            kv_cache_dtype=torch.float16,
            head_num=2,
            head_dim=4,
            v_head_dim=4,
            swa_head_num=2,
            swa_head_dim=4,
            swa_v_head_dim=4,
            page_size=1,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            mamba_layer_ids=[0, 1],
            mamba2_cache_params=cp,
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            max_mamba_cache_size=20,
            model_context_len=16,
            extra_max_context_len=4,
            max_num_reqs=4,
            enable_memory_saver=False,
            enable_mamba_extra_buffer=False,
            enable_mamba_extra_buffer_lazy=False,
            disable_overlap_schedule=True,
            need_sort=False,
        )
        kw.update(over)
        return kw

    def test_streaming_session_lazy_checkpoint_cleanup(self):
        from types import SimpleNamespace

        from sglang.srt.managers.schedule_batch import ReqKvInfo
        from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.session.streaming_session import SessionSlot, StreamingSession

        for lazy_compaction in (None, False, True):
            for ids in ((0, -1), (-1, 0), (-1, -1), (0, 1)):
                with self.subTest(compaction=lazy_compaction, ids=ids):
                    kw = self._factory_kwargs(
                        enable_mamba_extra_buffer=True,
                        enable_mamba_extra_buffer_lazy=True,
                        lazy_compaction=lazy_compaction,
                    )
                    with get_parallel().override(attn_dcp_size=1):
                        bundle = build_test_unified_pools(**kw)
                    pool = bundle.req_to_token_pool
                    allocator = bundle.token_to_kv_pool_allocator.state_slots
                    if lazy_compaction is None:
                        pool = SimpleNamespace()
                        allocator = MambaSlotAllocator(16, "cpu")
                    available = allocator.available_size()
                    count = sum(i != -1 for i in ids)
                    live = allocator.alloc(count + 1)
                    track = live.new_full((2,), -1)
                    for position, index in enumerate(ids):
                        if index != -1:
                            track[position] = live[index + 1]
                    slot = SessionSlot(
                        kv=ReqKvInfo(
                            req_pool_idx=1,
                            mamba_pool_idx=live[0],
                            mamba_ping_pong_track_buffer=track,
                            mamba_ping_pong_empty=[index == -1 for index in ids],
                        )
                    )
                    session = StreamingSession(
                        SimpleNamespace(
                            req_to_token_pool=pool,
                            token_to_kv_pool_allocator=SimpleNamespace(
                                state_slots=allocator
                            ),
                        )
                    )
                    session.slots = {"test": slot}
                    self.assertEqual(kv_mamba_slots(slot.kv), count + 1)
                    session._free_slot_mamba(slot)
                    self.assertEqual(kv_mamba_slots(slot.kv), 0)
                    self.assertEqual(allocator.available_size(), available)
                    session._free_slot_mamba(slot)
                    self.assertEqual(allocator.available_size(), available)

    def test_mamba_lazy_checkpoint_cleanup(self):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_parallel

        for tri_pool in (False, True):
            for lazy in (False, True):
                for keep_checkpoint in (False, True):
                    with self.subTest(
                        tri_pool=tri_pool, lazy=lazy, keep=keep_checkpoint
                    ):
                        from sglang.srt.mem_cache import kv_cache_configurator as cfg

                        kw = self._factory_kwargs()
                        model = SimpleNamespace(
                            get_num_kv_heads=lambda tp, dcp: 2,
                            get_swa_num_kv_heads=lambda tp: 2,
                            head_dim=4,
                            v_head_dim=4,
                            swa_head_dim=4,
                            swa_v_head_dim=4,
                            context_len=16,
                            full_attention_layer_ids=[0],
                            swa_attention_layer_ids=[1],
                            sliding_window_size=8,
                        )
                        configurator = SimpleNamespace(
                            mambaish_config=SimpleNamespace(
                                mamba2_cache_params=kw["mamba2_cache_params"],
                                full_attention_layer_ids=[0],
                            ),
                            model_config=model,
                            layer_info=SimpleNamespace(
                                swa_attention_layer_ids=[1],
                                full_attention_layer_ids=[0],
                                start_layer=0,
                                end_layer=2,
                            ),
                            device=_DEV,
                            kv_cache_dtype=torch.float16,
                            page_size=1,
                            is_draft_worker=False,
                            use_mla_backend=False,
                            is_hybrid_swa=tri_pool,
                            is_hybrid_swa_compress=False,
                            sliding_window_size=model.sliding_window_size,
                            forward_stream=None,
                            # Spec off: no draft region to fuse.
                            _fused_draft_decision=lambda: None,
                            _fused_draft_for_pool_factory=lambda decision: None,
                            _unified_request_rows=lambda *, has_state: (
                                cfg.KVCacheConfigurator._unified_request_rows(
                                    configurator, has_state=has_state
                                )
                            ),
                            _unified_swa_head_geometry=lambda: (
                                cfg.KVCacheConfigurator._unified_swa_head_geometry(
                                    configurator
                                )
                            ),
                            _unified_window_and_state=lambda: (tri_pool, True),
                            _kv_layer_decls=lambda: (
                                cfg.KVCacheConfigurator._kv_layer_decls(configurator)
                            ),
                            post_capture_kv_active=False,
                        )
                        # Run the production configurator AND factory. Reverting
                        # either top-level flag forwarding must break cleanup.
                        with (
                            get_parallel().override(attn_dcp_size=1, attn_tp_size=1),
                            patch.object(
                                cfg,
                                "get_exec",
                                return_value=SimpleNamespace(
                                    features=SimpleNamespace(enable_memory_saver=False),
                                    mamba=SimpleNamespace(
                                        enable_mamba_extra_buffer=True,
                                        enable_mamba_extra_buffer_lazy=lazy,
                                    ),
                                ),
                            ),
                            patch.object(
                                cfg,
                                "get_spec",
                                return_value=SimpleNamespace(
                                    speculative_num_draft_tokens=None,
                                ),
                            ),
                            patch.object(
                                cfg,
                                "get_schedule",
                                return_value=SimpleNamespace(
                                    max_mamba_cache_size=20,
                                    disable_overlap_schedule=False,
                                    mamba_full_memory_ratio=None,
                                ),
                            ),
                            patch.object(
                                cfg,
                                "get_disagg",
                                return_value=SimpleNamespace(
                                    disaggregation_mode="null",
                                ),
                            ),
                            patch.object(
                                cfg, "_should_enable_lazy_compaction", return_value=True
                            ),
                            patch.object(
                                cfg, "mamba_cache_chunk_size", return_value=64
                            ),
                            patch.object(
                                cfg,
                                "max_speculative_num_draft_tokens",
                                return_value=None,
                            ),
                        ):
                            # Construction consumes the already resolved plan and size.
                            # Keep request tracking reservations identical to the runner.
                            planned = build_test_unified_pools(
                                **self._factory_kwargs(
                                    swa_attention_layer_ids=[1] if tri_pool else [],
                                    enable_mamba_extra_buffer=True,
                                    enable_mamba_extra_buffer_lazy=lazy,
                                    disable_overlap_schedule=False,
                                )
                            ).token_to_kv_pool_allocator.storage
                            configurator._unified_plan = planned.plan
                            configurator._unified_sized = planned.sized
                            storage = cfg.KVCacheConfigurator._init_unified_pools(
                                configurator,
                                sizes=SimpleNamespace(
                                    max_running_requests=4,
                                    max_total_num_tokens=64,
                                    full_max_total_num_tokens=64,
                                    swa_max_total_num_tokens=32,
                                    unified_total_bytes=None,
                                    unified_memory_pool_bytes=None,
                                ),
                            )
                            pool = storage.requests
                        self.assertEqual(pool.enable_mamba_extra_buffer_lazy, lazy)
                        state = storage.allocator.state
                        allocator = state.slots
                        initial_available = allocator.available_size()
                        req = SimpleNamespace(
                            kv=SimpleNamespace(
                                req_pool_idx=0, mamba_pool_idx=allocator.alloc(1)[0]
                            )
                        )
                        self.assertTrue(state.alloc_ping_pong_buffer(pool, req))
                        buf = req.kv.mamba_ping_pong_track_buffer
                        self.assertEqual(int((buf != -1).sum()), 1 if lazy else 2)
                        if lazy:
                            # A decode boundary replaces the old checkpoint and
                            # leaves its ping-pong entry unallocated (-1).
                            replacement = allocator.alloc(1)
                            allocator.free(buf[:1].clone())
                            pool.set_mamba_ping_pong_slot(req, 0, -1)
                            pool.set_mamba_ping_pong_slot(req, 1, replacement[0])
                        else:
                            pool.set_mamba_ping_pong_slot(req, 1, buf[1])
                        retained = buf[1:2].clone()
                        with patch.object(
                            allocator, "free", wraps=allocator.free
                        ) as free:
                            state.release(pool, req, 1 if keep_checkpoint else None)
                            for call in free.call_args_list:
                                self.assertTrue(bool((call.args[0] >= 0).all()))
                        self.assertEqual(
                            allocator.available_size(),
                            initial_available - int(keep_checkpoint),
                        )
                        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
                        if keep_checkpoint:
                            self.assertTrue(
                                bool((allocator.translate(retained) >= 0).all())
                            )
                            allocator.free(retained)
                        self.assertEqual(allocator.available_size(), initial_available)

    def test_ping_pong_mapping_mirrors_the_request_buffers(self):
        from types import SimpleNamespace

        from sglang.srt.managers.schedule_batch import (
            ReqKvInfo,
            _gather_mamba_track_indices,
        )
        from sglang.srt.runtime_context import get_parallel

        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                kw = self._factory_kwargs(
                    enable_mamba_extra_buffer=True,
                    enable_mamba_extra_buffer_lazy=lazy,
                    disable_overlap_schedule=False,
                    max_mamba_cache_size=16,
                )
                with get_parallel().override(attn_dcp_size=1):
                    bundle = build_test_unified_pools(**kw)
                pool = bundle.req_to_token_pool
                state = bundle.token_to_kv_pool_allocator.state
                mapping = pool.req_index_to_mamba_ping_pong_track_buffer_mapping
                allocator = state.slots
                reqs = [SimpleNamespace(kv=ReqKvInfo()) for _ in range(3)]
                rows = acquire_rows(pool, state, reqs)

                def buffers():
                    return [
                        req.kv.mamba_ping_pong_track_buffer.tolist() for req in reqs
                    ]

                self.assertEqual(mapping[rows].tolist(), buffers())
                # Every writer of a buffer entry rewrites the request's row.
                pool.donate_mamba_ping_pong_slot(reqs[0], allocator.alloc(1))
                pool.set_mamba_ping_pong_slot(reqs[1], 0, -1)
                pool.set_mamba_ping_pong_slot(reqs[1], 1, allocator.alloc(1)[0])
                self.assertEqual(mapping[rows].tolist(), buffers())
                # The device gather returns what the per-request .item() read did.
                positions = [0, 0, 1]
                gathered = _gather_mamba_track_indices(
                    mapping, torch.tensor(rows), torch.tensor(positions)
                )
                item_reads = [
                    req.kv.mamba_ping_pong_track_buffer[pos].item()
                    for req, pos in zip(reqs, positions)
                ]
                self.assertEqual(gathered.tolist(), item_reads)

    def test_budget_sizing_and_boot_signature(self):
        from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator

        budget = 1 << 20
        bundle = build_test_unified_pools(
            **self._factory_kwargs(unified_total_bytes=budget)
        )
        pool = bundle.arena
        allocator = bundle.token_to_kv_pool_allocator
        plan = allocator.storage.plan
        self.assertIsInstance(allocator, ChainAllocator)
        self.assertEqual(
            [m.sub_pool_name for m in allocator.members],
            ["full", plan.primary_window().name, "mamba"],
        )
        sized = allocator.storage.sized
        self.assertEqual(pool.total_bytes + sized.fixed_bytes, budget)
        # Boot signature: 3 sub-pools in chain order, swa is the float.
        self.assertEqual(len(pool.bands), 3)
        self.assertEqual(
            [(b.name, b.grow_direction) for b in pool.bands],
            [("mamba", "up"), (plan.primary_window().name, "float"), ("full", "down")],
        )

    def test_fallback_is_the_token_count_resum(self):
        bundle = build_test_unified_pools(**self._factory_kwargs())
        pool = bundle.arena
        plan = bundle.token_to_kv_pool_allocator.storage.plan
        want = (
            64 * plan.sub_pool("full").spec.entry_bytes()
            + 32 * plan.primary_window().spec.entry_bytes()
            + self._factory_kwargs()["max_mamba_cache_size"]
            * plan.sub_pool("mamba").spec.entry_bytes()
            + plan.sink_bytes()
        )
        self.assertEqual(pool.total_bytes, want)

    def test_full_side_carries_the_fused_draft_region(self):
        """The boot solve prices the fused entry whenever a placement resolves,
        so a factory that dropped the kwarg would allocate UNFUSED under that
        price and under-budget the private draft pool."""
        region = DenseDraftRegion(
            lane_num=1, head_num=2, head_dim=4, store_dtype=torch.float16
        )
        placement = FusedDraftPlacement.from_counts(
            counts={"full": [1]}, regions={"full": region}
        )
        bundle = build_test_unified_pools(**self._factory_kwargs(fused_draft=placement))
        plan = bundle.token_to_kv_pool_allocator.storage.plan
        self.assertIs(plan.fused_draft, placement)
        self.assertIs(plan.sub_pool("full").spec.draft_region, region)
        for other in (plan.primary_window().name, "mamba"):
            self.assertIsNone(plan.sub_pool(other).spec.draft_region, other)

    def _state_placement(self):
        region = DraftStateRegion(
            lane_num=2,
            state=DraftStateGeometry(
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(0, 0, 0),
                temporal_dtype=torch.float32,
            ),
        )
        return FusedDraftPlacement.from_counts(
            counts={"mamba": [1, 1]}, regions={"mamba": region}
        )

    def test_fused_draft_state_shares_the_target_slots(self):
        """A draft runner's state over the fused state block keeps the
        target's slot allocator (the draft's slot ids ARE the target's), and
        views only its own block, keyed by the runner's layer ids."""
        from sglang.srt.mem_cache.unified_draft_pool import fused_draft_state

        bundle = build_test_unified_pools(
            **self._factory_kwargs(fused_draft=self._state_placement())
        )
        target = bundle.token_to_kv_pool_allocator.state
        state_spec = bundle.token_to_kv_pool_allocator.storage.plan.sub_pool(
            "mamba"
        ).spec
        draft = fused_draft_state(
            target=target,
            arena=bundle.arena,
            spec=state_spec,
            layer_lanes={1: 1},
        )
        self.assertEqual(draft.sub_pool, target.sub_pool)
        self.assertIs(draft.slots, target.slots)
        self.assertEqual(draft.layer_map, {1: 0})
        conv = draft.pool.mamba2_layer_cache(draft.layer_map[1]).conv[0]
        self.assertEqual(tuple(conv.shape), (target.pool.size + 1, 3, 8))
        draft_conv, _ = state_spec.draft_views(bundle.arena)
        self.assertEqual(conv.data_ptr(), draft_conv[0][1].data_ptr())

    def test_bs1_floor_fails_loud_before_construction(self):
        """A budget far below one worst-case request must raise BEFORE any
        pool construction -- under-sizing is a retract LIVELOCK at runtime."""
        with self.assertRaisesRegex(RuntimeError, "bs=1 floor"):
            build_test_unified_pools(
                **self._factory_kwargs(
                    unified_total_bytes=1024,  # << ctx * e_f alone
                    model_context_len=100_000,
                    sliding_window_size=64,
                )
            )


class TestTriPoolHardening(_PoolTestCase):
    """Pressure lanes: the planned-rebalance remedy in the alloc path (a
    mis-positioned float must not fail an alloc that fits in total bytes),
    retract-loop convergence through `check_decode_capacity`, and bounded copy
    traffic under alternating end pressure.
    """

    def _build(self, **kw):
        return TestUnifiedTriPool._build(self, **kw)

    def test_alloc_rebalances_a_blocking_float(self):
        # Fill much of the high band so the midpoint-placed float walls the
        # low band's free bytes off from `full`; the next alloc must SLIDE it
        # rather than fail while total bytes suffice.
        _, allocator, kvcache, _ = self._build(n_full=32, n_swa=24, n_state=8)
        sa = allocator.swa_attn_allocator
        v0 = allocator.alloc(4)  # places the float at the region midpoint
        self.assertIsNotNone(v0)
        TestUnifiedTriPool._stamp(self, allocator, kvcache, v0)
        # Exhaust the high band directly on the full end (full-only growth,
        # e.g. long decode of already-admitted requests).
        fa = allocator.full_attn_allocator
        b_high_pages = fa._current_gap_bytes() // fa.entry_bytes_per_page
        grab = fa.alloc(max(0, (b_high_pages - 2)))
        self.assertIsNotNone(grab)
        # The read credits sliding the float, so exactly what it reports must
        # be taken, and no layout that leaves the float in place holds it.
        need = allocator.available_size()
        page = allocator.page_size
        demands = {
            n: d
            for n, d in allocator.page_demands(
                token_demand(allocator, need, need)
            ).items()
            if d
        }
        self.assertIsNone(plan(allocator.pricing_snapshot(), demands, FLUSH_ENDS))
        self.assertFalse(
            allocator.ensure_capacity(token_demand(allocator, need + page, need + page))
        )
        live_before = sa._live_pages()
        moves_before = len(sa._inverse_history)
        v1 = allocator.alloc(need)
        self.assertIsNotNone(
            v1, "alloc must rebalance the blocking float instead of failing"
        )
        self.assertEqual(int(v1.numel()), need)
        moved = sum(int(s.numel()) for s, _, _ in sa._inverse_history[moves_before:])
        self.assertGreater(moved, 0, "the rebalance path must have fired")
        # Cost bound min(L_live, G): never more than the live pages present
        # when the slide ran (the leapfrog cap).
        self.assertLessEqual(moved, live_before)
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_check_decode_capacity_retract_convergence(self):
        # Retract loop: token blocks freed one at a time until the next step
        # fits; must converge before bs=1 and never report capacity early.
        _, allocator, _, _ = self._build(n_full=32, n_swa=24, n_state=8)
        reqs = []
        while True:
            v = allocator.alloc(4)
            if v is None or allocator.available_size() < 4:
                if v is not None:
                    reqs.append(v)
                break
            reqs.append(v)
        self.assertGreater(len(reqs), 2)
        # Pool saturated: a large decode step does not fit.
        need = 16
        while not allocator.check_decode_capacity(num_tokens=need, tree_cache=None):
            self.assertGreater(len(reqs), 1, "retract must converge before bs=1")
            allocator.free(reqs.pop())
        # A yes is the decode take succeeding, relayout included.
        self.assertIsNotNone(allocator.alloc(need))
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_alternating_pressure_copy_traffic_bounded(self):
        # Alternating full-grow / swa-churn: hole recycling and absorption do
        # the steady-state work, so total float moves stay bounded.
        _, allocator, kvcache, _ = self._build(n_full=48, n_swa=32, n_state=8)
        sa = allocator.swa_attn_allocator
        fa = allocator.full_attn_allocator
        total_alloc_pages = 0
        for _ in range(6):
            v = allocator.alloc(8)
            self.assertIsNotNone(v)
            total_alloc_pages += 8
            TestUnifiedTriPool._stamp(self, allocator, kvcache, v)
            allocator.free_swa(v)  # window slide: tombstones -> holes/absorb
            g = fa.alloc(4)  # full-side decode growth
            self.assertIsNotNone(g)
            total_alloc_pages += 4
            fa.free(g)
        moved = sum(int(s.numel()) for s, _, _ in sa._inverse_history)
        self.assertLessEqual(
            moved,
            total_alloc_pages // 2,
            "steady-state churn must be predominantly zero-copy",
        )
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_joint_eviction_loop_stops_without_progress(self):
        # tree_cache=None: the default helper no-ops; the bounded loop must
        # return promptly (no infinite re-check) and the gate reports honestly.
        _, allocator, _, _ = self._build()
        big = allocator.available_size() + 64
        self.assertFalse(
            allocator.check_decode_capacity(num_tokens=big, tree_cache=None)
        )


class TestJointCapacityIsHonoured(_PoolTestCase):
    """`alloc(available_size())` must never fail.

    REGRESSION: the joint predicate priced the swa float's extension in RAW
    BYTES, but `take_physical_pages` yields only whole pages on the float's
    OWN grid, whose low edge is rounded up to a multiple of the NEIGHBOUR's
    entry size -- so the budget credited a page the grid could not yield.
    Swept over geometries so a symmetric mistake on the FULL side surfaces
    here too.
    """

    def _build(self, *, page_size, n_full, n_swa, n_state, lazy, specs):
        full, swa, mamba = specs
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = arena_of(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            # The allocator pages single tokens; ``page_size`` only scales
            # the capacities.
            page_size=1,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        return pool, ChainAllocator(
            roles=ChainRoles(owner="full", window="swa", state="mamba"),
            arena=pool,
            kvcache=kvcache,
            views=stand_in_views(
                ChainRoles(owner="full", window="swa", state="mamba"),
                kvcache,
                FakeKVCache(pool, "mamba"),
            ),
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )

    def test_fresh_boot_alloc_of_available_size_succeeds(self):
        # Geometries chosen so the mamba end's frontier (a multiple of the
        # STATE entry size) lands off the swa float's page grid.
        for page_size in (1, 2, 4):
            for fl, sl, ml in ((4, 3, 1), (4, 2, 2), (6, 3, 1), (3, 5, 2)):
                for n_full, n_swa, n_state in ((24, 16, 4), (32, 16, 8), (20, 12, 6)):
                    for lazy in (False, True):
                        specs = tri_sub_pool_specs(
                            full_layer_num=fl,
                            swa_layer_num=sl,
                            state_layer_num=ml,
                            head_num=1,
                            head_dim=8,
                        )
                        with self.subTest(
                            ps=page_size,
                            layers=(fl, sl, ml),
                            n=(n_full, n_swa, n_state),
                            lazy=lazy,
                        ):
                            _pool, alloc = self._build(
                                page_size=page_size,
                                n_full=n_full * page_size,
                                n_swa=n_swa * page_size,
                                n_state=n_state,
                                lazy=lazy,
                                specs=specs,
                            )
                            n = alloc.available_size()
                            if n <= 0:
                                continue
                            # The whole point: the number the scheduler reads
                            # must be allocatable, with no backstop assert.
                            out = alloc.alloc(n)
                            self.assertIsNotNone(
                                out,
                                f"alloc(available_size()={n}) returned None",
                            )
                            self.assertEqual(out.numel(), n)
                            # Both sides bound for every allocated virtual id.
                            fa = alloc.full_attn_allocator
                            sa = alloc.swa_attn_allocator
                            self.assertTrue(
                                bool((fa.ids.virtual_to_physical[out] >= 0).all())
                            )
                            self.assertTrue(
                                bool((sa.ids.virtual_to_physical[out] >= 0).all())
                            )

    def test_available_size_never_exceeds_the_float_page_grid(self):
        """Direct form: the joint answer, converted to float pages, must fit
        inside what `_region_bounds_pages` actually offers."""
        for page_size in (1, 4):
            specs = tri_sub_pool_specs(
                full_layer_num=4,
                swa_layer_num=3,
                state_layer_num=1,
                head_num=1,
                head_dim=8,
            )
            _pool, alloc = self._build(
                page_size=page_size,
                n_full=24 * page_size,
                n_swa=16 * page_size,
                n_state=4,
                lazy=False,
                specs=specs,
            )
            sa = alloc.swa_attn_allocator
            n_pages = alloc.available_size() // page_size
            lo, hi = sa._region_bounds_pages()
            with self.subTest(ps=page_size):
                self.assertLessEqual(
                    n_pages - sa._hole_pages(),
                    max(0, hi - lo),
                    "joint available_size() promises more float pages than the "
                    "float's own page grid can yield",
                )


class TestFloatRelocationIsOrderedAgainstTheForward(_PoolTestCase):
    """Float relocation must order itself after the in-flight forward's
    writes to the pages it moves BEFORE its first copy.

    REGRESSION: float moves issued `move_kv_cache` and rebound
    `virtual_to_physical` with no ordering against the running forward, so a
    copy could carry pre-write bytes and every later reader saw a destination
    that never received those writes -- silently wrong KV, no crash. The END
    pools guard the same hazard via `_settle_inflight_forward`.
    """

    def _tri(self, lazy=True):
        full, swa, mamba = tri_sub_pool_specs(head_num=1, head_dim=8)
        total = (
            48 * full.entry_bytes() + 32 * swa.entry_bytes() + 8 * mamba.entry_bytes()
        )
        pool = arena_of(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = FakeUnifiedSWAKVPool(pool)
        alloc = ChainAllocator(
            roles=ChainRoles(owner="full", window="swa", state="mamba"),
            arena=pool,
            kvcache=kvcache,
            views=stand_in_views(
                ChainRoles(owner="full", window="swa", state="mamba"),
                kvcache,
                FakeKVCache(pool, "mamba"),
            ),
            device=_DEV,
            full_max_total_num_tokens=48,
            swa_max_total_num_tokens=32,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )
        return pool, alloc, kvcache

    def _trace(self, flt):
        """Record the order of (wait, move) on the float."""
        order = []
        real_wait = flt._wait_writers
        real_move = flt._move_pages_and_rebind

        def wait(*args, **kwargs):
            order.append("wait")
            return real_wait(*args, **kwargs)

        def move(src, dst):
            order.append("move")
            return real_move(src, dst)

        flt._wait_writers = wait
        flt._move_pages_and_rebind = move
        return order

    def test_a_float_move_waits_the_writers_before_the_first_copy(self):
        _pool, alloc, _kv = self._tri()
        flt = alloc.swa_attn_allocator
        # Occupy the float, then free a page so a relocation has something to
        # move and somewhere to move it.
        v = alloc.alloc(12)
        self.assertIsNotNone(v)
        alloc.free(v[:4])
        order = self._trace(flt)
        open_float_side(flt, "low", flt._gap_pages()[0] * flt.entry_bytes_per_page + 1)
        self.assertIn("move", order, "the geometry asked for no relocation")
        self.assertIn("wait", order, "the move never waited the writers")
        self.assertLess(
            order.index("wait"),
            order.index("move"),
            f"a copy was issued before the wait: {order}",
        )

    def test_the_settle_is_a_stream_wait_not_a_host_sync(self):
        """The settle before a float move must be a stream wait: a host sync
        there would land on the alloc-shortfall path every time the float moves."""
        settle = inspect.getsource(Band._settle_inflight_forward)
        self.assertIn("_wait_on_stream", settle)
        # The wait itself is issued by the signal helper the band delegates to.
        src = inspect.getsource(Band._wait_on_stream) + inspect.getsource(wait_signals)
        self.assertIn("wait_signals", src)
        self.assertIn("wait_event", src)
        for src in (settle, src):
            self.assertNotIn(".item()", src)
            self.assertNotIn("synchronize()", src)


class TestPreallocIsPricedOnTheSharedGrid(_PoolTestCase):
    """REGRESSION: PD admission compared FULL and SWA against per-side token
    budgets, but each side's `available_size` credits the peer's drainable
    holes, so a pair that each side can host alone can be jointly infeasible.
    Such a pair was admitted and then refused inside `alloc_extend_swa_tail`."""

    def _build(self, **kw):
        return TestUnifiedTriPool._build(self, **kw)

    @staticmethod
    def _empty_tree():
        """A prefix tree with nothing evictable."""
        return SimpleNamespace(
            full_evictable_size=lambda: 0,
            swa_evictable_size=lambda: 0,
            window_evictable_size=lambda component: 0,
            mamba_evictable_size=lambda: 0,
            supports_mamba=lambda: True,
            window_retain_size=lambda component: 1 << 20,
        )

    def test_a_pair_each_side_can_host_alone_is_still_refused(self):
        # page_size 1 leaves no slack between the per-side and joint views;
        # the double-count only has room to show on a paged grid.
        _, allocator, _, _ = self._build(page_size=4)
        full_demand = allocator.full_available_size()
        swa_demand = allocator.swa_available_size()
        self.assertGreater(min(full_demand, swa_demand), 0)
        # Each side alone reports room for its own half ...
        self.assertLessEqual(full_demand, allocator.full_available_size())
        self.assertLessEqual(swa_demand, allocator.swa_available_size())
        # ... yet the two draw on the same bytes, so the grid refuses the pair.
        self.assertFalse(
            allocator.ensure_capacity(token_demand(allocator, full_demand, swa_demand))
        )
        self.assertFalse(
            allocator.prealloc_fits(
                self._empty_tree(),
                token_demand(allocator, full_demand, swa_demand),
                full_budget_tokens=full_demand,
                swa_budget_tokens=swa_demand,
            )
        )

    def test_the_scheduler_budget_still_binds(self):
        # Decode builds each side's budget as that side's availability minus
        # the tokens it holds back for running requests; what a budget holds
        # back is charged beside the request on both sides at once.
        _, allocator, _, _ = self._build()
        page_size = allocator.page_size
        held = 2 * page_size
        full_budget = allocator.full_available_size() - held
        swa_budget = allocator.swa_available_size() - held
        self.assertTrue(
            allocator.prealloc_fits(
                self._empty_tree(),
                token_demand(allocator, page_size, page_size),
                full_budget_tokens=full_budget,
                swa_budget_tokens=swa_budget,
            )
        )
        # The request may not exceed its side's budget.
        self.assertFalse(
            allocator.prealloc_fits(
                self._empty_tree(),
                token_demand(allocator, page_size, page_size),
                full_budget_tokens=page_size - 1,
                swa_budget_tokens=swa_budget,
            )
        )
        # Holding back nearly all of each side's own availability at once
        # leaves no shared bytes for the request.
        self.assertFalse(
            allocator.prealloc_fits(
                self._empty_tree(),
                token_demand(allocator, page_size, page_size),
                full_budget_tokens=page_size,
                swa_budget_tokens=page_size,
            )
        )


if __name__ == "__main__":
    unittest.main()
