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
"""Tri-pool composite (`UnifiedMambaSWATokenToKVPoolAllocator`) -- full KV +
SWA KV + mamba/conv state in ONE unified byte buffer, chain
``[mamba (up END) | swa (FLOAT) | full (down END)]``.

Pinned contracts (each guards a distinct failure mode):
  - chain wiring + the swa side being a `FloatMultiEndedAllocator`;
  - the JOINT `available_size()` feasibility contract: `alloc(N)` for
    N == available_size() must succeed (full extends into the high band
    first, then the float extends a single side — the predicate models
    exactly that order; over-promising here is the fail-loud
    `alloc_with_virtual` assert, i.e. a crash in production);
  - `free_swa` tombstones become float HOLES recycled IN PLACE by later
    allocs (steady-state SWA churn == zero copies), while the full side
    keeps the token;
  - the per-request state surface (`UnifiedMambaSlotAllocator` over the
    mamba END) and its independence from the token surface;
  - urgent flushes drain the two ENDS but never touch the float's holes.

Pure CPU; fakes stand in for the KV pools (data markers verify moves).

    python -m pytest test/registered/unit/mem_cache/test_unified_tri_pool.py -v
"""

import inspect
import unittest

import torch

import sglang.srt.mem_cache.allocator.unified_sub_pool as mea
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import FloatMultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMambaSlotAllocator,
    init_unified_mamba_swa_pools,
)
from sglang.test.ci.ci_register import register_cpu_ci

# Hermetic convention of this directory's pool tests: plain unittest.TestCase,
# only ci_register imported (no heavy sglang.test.test_utils chain).
register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_DEV = "cpu"


class _FakeKVCache:
    """buf[p] == virtual id stored at physical slot p (-1 free); moves copy it."""

    def __init__(self, max_slots: int):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class _FakeUnifiedSWAKVPool:
    class _SubKV(_FakeKVCache):
        def __init__(self, max_slots):
            super().__init__(max_slots)
            self.allocator = None

        def attach_allocator(self, allocator):
            self.allocator = allocator

    def __init__(self, shared_pool: UnifiedKVPool):
        self.full_kv_pool = self._SubKV(shared_pool.max_slots("full"))
        self.swa_kv_pool = self._SubKV(shared_pool.max_slots("swa"))
        self._full_allocator = None
        self._swa_allocator = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


def _tri_specs(
    full_layer_num=4, swa_layer_num=2, state_layer_num=2, head_num=2, head_dim=4
):
    full = MHASubPoolSpec(
        name="full",
        layer_num=full_layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction="down",
    )
    swa = MHASubPoolSpec(
        name="swa",
        layer_num=swa_layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction="float",
    )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=state_layer_num,
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(0, 0, 0),  # Inkling: conv-only, no SSM state
        temporal_dtype=torch.float32,
        grow_direction="up",
    )
    return full, swa, mamba


class TestUnifiedTriPool(unittest.TestCase):
    def _build(
        self,
        n_full=32,
        n_swa=16,
        n_state=8,
        lazy_compaction=False,
    ):
        full, swa, mamba = _tri_specs()
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        allocator = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=mamba_kv,
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy_compaction,
        )
        return pool, allocator, kvcache, mamba_kv

    def _stamp(self, allocator, kvcache, v):
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        kvcache.full_kv_pool.buf[fa.virtual_to_physical[v]] = v
        kvcache.swa_kv_pool.buf[sa.virtual_to_physical[v]] = v

    # -- construction --

    def test_chain_wiring_and_float_swa(self):
        pool, allocator, kvcache, _ = self._build()
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        ma = allocator.mamba_allocator
        self.assertIsInstance(sa, FloatMultiEndedAllocator)
        self.assertIs(ma.high_peer, sa)
        self.assertIs(sa.low_peer, ma)
        self.assertIs(sa.high_peer, fa)
        self.assertIs(fa.low_peer, sa)
        # Canonical chain order in the pool.
        self.assertEqual(
            [s.name for s in pool.sub_pool_specs], ["mamba", "swa", "full"]
        )
        # KV pool got the allocators.
        self.assertIs(kvcache._full_allocator, fa)
        self.assertIs(kvcache._swa_allocator, sa)

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

    def test_available_size_alloc_contract(self):
        for lazy in (False, True):
            _, allocator, kvcache, _ = self._build(lazy_compaction=lazy)
            avail = allocator.available_size()
            self.assertGreater(avail, 0)
            v = allocator.alloc(avail)
            self.assertIsNotNone(
                v, f"alloc(available_size()={avail}) must succeed (lazy={lazy})"
            )
            self.assertEqual(int(v.numel()), avail)
            # Both sides bound for every allocated virtual id.
            fa = allocator.full_attn_allocator
            sa = allocator.swa_attn_allocator
            self.assertTrue(bool((fa.virtual_to_physical[v] >= 0).all()))
            self.assertTrue(bool((sa.virtual_to_physical[v] >= 0).all()))

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
        """The block whose SWA-physical pages touch neither float boundary --
        `free_swa` on it must create interior holes (a boundary block would be
        absorbed instead; both are zero-copy, different mechanisms)."""
        sa = allocator.swa_attn_allocator
        for v in blocks:
            pages = set(int(x) for x in sa.virtual_to_physical[v].tolist())
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
        self.assertTrue(bool((fa.virtual_to_physical[v_mid] >= 0).all()))
        self.assertTrue(bool((sa.virtual_to_physical[v_mid] == -1).all()))
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
        # The OTHER zero-copy mechanism: a boundary block's tombstones shrink
        # the span, handing bytes back to the neighbours. The shrink is
        # DEFERRED out of the per-step free (it needs the hole set on the
        # host); the per-step opportunistic flush is where it lands.
        _, allocator, kvcache, _ = self._build()
        blocks = [allocator.alloc(4) for _ in range(2)]
        for v in blocks:
            self._stamp(allocator, kvcache, v)
        sa = allocator.swa_attn_allocator
        span_pages = sa._span_pages()
        # Pick a block holding a span-boundary page.
        boundary = None
        for v in blocks:
            pages = set(int(x) for x in sa.virtual_to_physical[v].tolist())
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
        self.assertTrue(bool((fa.virtual_to_physical[va] == -1).all()))
        self.assertTrue(bool((sa.virtual_to_physical[va] == -1).all()))
        # Fully-freed float parks and is transparent again.
        self.assertTrue(sa._is_frontier_transparent())

    # -- per-request state surface --

    def test_mamba_slot_allocator_surface(self):
        pool, allocator, _, mamba_kv = self._build()
        slot_alloc = UnifiedMambaSlotAllocator(
            allocator.mamba_allocator,
            max_size=pool.max_slots("mamba") - 1,
            device=_DEV,
        )
        v = slot_alloc.alloc(3)
        self.assertIsNotNone(v)
        p = slot_alloc.translate(v)
        self.assertTrue(bool((p >= 0).all()))
        mamba_kv.buf[p] = v
        self.assertEqual(
            slot_alloc.available_size(),
            (pool.max_slots("mamba") - 1) - 3,
        )
        slot_alloc.free(v)
        self.assertEqual(slot_alloc.available_size(), pool.max_slots("mamba") - 1)
        # Group prefetch draw-down + surplus return.
        slot_alloc.alloc_group_begin(4)
        s1 = slot_alloc.alloc(1)
        self.assertIsNotNone(s1)
        slot_alloc.alloc_group_end()
        self.assertEqual(
            slot_alloc.available_size(),
            (pool.max_slots("mamba") - 1) - 1,
        )

    # -- cost + flush semantics --

    def test_mamba_slot_full_token_cost_formula(self):
        _, allocator, _, _ = self._build()
        e_tok = (
            allocator.full_attn_allocator.entry_bytes
            + allocator.swa_attn_allocator.entry_bytes
        )
        m = allocator.mamba_allocator.entry_bytes_per_page
        self.assertEqual(allocator.mamba_slot_full_token_cost(), -(-m // e_tok))

    def test_urgent_flush_preserves_float_holes(self):
        _, allocator, kvcache, _ = self._build(lazy_compaction=True)
        blocks = [allocator.alloc(4) for _ in range(3)]
        for v in blocks:
            self._stamp(allocator, kvcache, v)
        allocator.free_swa(self._swa_interior_block(allocator, blocks))
        sa = allocator.swa_attn_allocator
        holes = sa._hole_pages()
        self.assertGreater(holes, 0)
        from sglang.srt.mem_cache.allocator.unified_sub_pool import _relieve_for_alloc

        _relieve_for_alloc(allocator, 1)
        self.assertEqual(sa._hole_pages(), holes)  # holes are assets, not backlog


class TestTriPagedFreeGroup(unittest.TestCase):
    """The tri composite at PAGE SIZE > 1, driven through the production free
    path: free_group_begin -> free_segment -> free_group_end.

    Regression (GPU eval_440, Inkling ps=128 boot crash): every other tri test
    runs at page_size=1, where the page-REPRESENTATIVE machinery
    (`free_page_reps_group` / `_release_page_reps`, added when the free path
    was made host-sync-free) is entirely dead code — `free_segment` frees
    directly. At ps>1 the composite releases reps by calling
    `swa_attn_allocator.free(..., _pages=...)`, and the float allocator was
    ported from a base that predates that keyword, so the first real decode
    batch died with `TypeError: unexpected keyword argument '_pages'`.

    `_pages` is not cosmetic: honouring it is what keeps the free path free of
    the data-dependent `torch.unique` host sync, so this also pins that the
    float takes the caller's page ids rather than re-deriving them.
    """

    def _build_paged(self, page_size=4, n_full=64, n_swa=32, n_state=8):
        full, swa, mamba = _tri_specs()
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        allocator = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=mamba_kv,
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

    def test_float_free_honours_caller_supplied_pages(self):
        """`_pages` must be USED, not merely accepted -- re-deriving it is the
        host sync the paged free path exists to avoid."""
        pool, allocator = self._build_paged()
        v = allocator.alloc(8)
        self.assertIsNotNone(v)
        sa = allocator.swa_attn_allocator
        ps = allocator.page_size
        pages = (v[::ps] // ps).clone()
        live_before = sa._live_pages()
        sa.free(v[::ps] * 0 + v[::ps], _pages=pages)
        self.assertEqual(sa._live_pages(), live_before - pages.numel())

    def test_ungrouped_segment_free_also_reaches_the_float(self):
        """`free_segment` outside a free group releases reps immediately --
        the same float call, one frame shallower."""
        pool, allocator = self._build_paged()
        v = allocator.alloc(8)
        self.assertIsNotNone(v)
        allocator.free_segment(v, start_pos=0)
        self.assertEqual(allocator.verify_byte_accounting(), [])


class TestTriFreeSwaNoHostSync(unittest.TestCase):
    """The tri's swa side is the FLOAT, and the float can never run the lazy
    event pipeline — so unless the per-step frees carry caller-derived page
    ids, the tri silently reintroduces the host syncs the sync-free free
    path removed. Poison the ops to pin the property.

    (Fixtures at page_size > 1 on purpose: ps==1 short-circuits the whole
    page machinery and hides exactly this class of bug.)
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
            alloc.free_swa(v[: 4 * self.PS], start_pos=0)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_float_free_has_no_stale_slot_item_sync(self):
        """The float's free must not `.item()`-assert per free (the lazy-path
        contract: callers must not double-free; the idle span == p2v-bound +
        holes conservation catches violations without a per-free sync)."""
        alloc = self._tri()
        v = alloc.alloc(4 * self.PS)
        sa = alloc.swa_attn_allocator
        from unittest import mock

        with mock.patch.object(
            torch.Tensor, "item", side_effect=AssertionError("item = host sync")
        ):
            sa.free(v[:: self.PS], _pages=v[:: self.PS] // self.PS)

    def test_fallback_free_swa_still_correct_for_radix_shapes(self):
        """Radix eviction hands arbitrary node values (no start_pos): the
        dedup fallback must keep working and end in the same state as the
        stride path."""
        a1, a2 = self._tri(), self._tri()
        v1, v2 = a1.alloc(6 * self.PS), a2.alloc(6 * self.PS)
        self.assertTrue(torch.equal(v1, v2))
        a1.free_swa(v1[: 4 * self.PS], start_pos=0)
        a2.free_swa(v2[: 4 * self.PS])
        self.assertTrue(
            torch.equal(
                a1.swa_attn_allocator.virtual_to_physical,
                a2.swa_attn_allocator.virtual_to_physical,
            )
        )
        self.assertEqual(a1.available_size(), a2.available_size())
        self.assertEqual(a1.verify_byte_accounting(), [])
        self.assertEqual(a2.verify_byte_accounting(), [])


class TestGeneralizedRebalance(unittest.TestCase):
    """The float must yield to WHICHEVER end is short, with the direction
    computed from the layout — not only to the token path's hard-coded side.

    The mechanism (`make_room`) was always side-agnostic; these pin the
    POLICY: any end pool's own-alloc shortfall reaches
    `_ask_float_for_room`, which derives the side from the caller's
    growth direction."""

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def test_state_end_shortfall_slides_the_float_low(self):
        """The previously-missing direction: mamba (grow-up END) starved while
        free bytes idle ABOVE the float. The remedy must slide the float up
        (open its LOW side) and let the state alloc succeed."""
        alloc = self._tri()
        v = alloc.alloc(4 * self.PS)  # places the float mid-region
        self.assertIsNotNone(v)
        ma = alloc.mamba_allocator
        sa = alloc.swa_attn_allocator
        self.assertFalse(sa._is_frontier_transparent())
        # Fill the LOW band exactly: as many state slots as fit below the
        # float's low frontier.
        e_m = ma.entry_bytes_per_page
        fit = (sa._byte_low_frontier() - ma._byte_high_frontier()) // e_m
        self.assertGreater(fit, 0)
        got = ma.alloc(int(fit) * ma.page_size)
        self.assertIsNotNone(got)
        low_before = sa.low_wm_page
        # One more slot does NOT fit below the float -- only a rebalance helps.
        more = ma.alloc(ma.page_size)
        self.assertIsNotNone(more, "state alloc must succeed via float rebalance")
        self.assertGreater(sa.low_wm_page, low_before)  # float slid UP
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_direction_is_derived_from_growth_on_both_ends(self):
        """Raw end+float+end chain, BOTH orientations in one fixture: the
        up-growing end opens the float's LOW side; the down-growing end opens
        its HIGH side. No layout assumption survives."""
        from test_multi_ended_allocator import TestFloatMultiEndedAllocator

        inst = TestFloatMultiEndedAllocator(
            [m for m in dir(TestFloatMultiEndedAllocator) if m.startswith("test_")][0]
        )
        _pool, up_end, fla, down_end, _kv = inst._build_tri()
        v = fla.alloc(8)  # opaque float mid-region
        self.assertIsNotNone(v)

        # UP end: exhaust its band below the float, then ask for more.
        e_up = up_end.entry_bytes_per_page
        fit = int((fla._byte_low_frontier() - up_end._byte_high_frontier()) // e_up)
        if fit > 0:
            self.assertIsNotNone(up_end.alloc(fit * up_end.page_size))
        low_before = fla.low_wm_page
        self.assertIsNotNone(up_end.alloc(up_end.page_size))
        self.assertGreater(fla.low_wm_page, low_before)  # opened LOW side

        # DOWN end: exhaust its band above the float, then ask for more.
        e_dn = down_end.entry_bytes_per_page
        fit = int((down_end._byte_low_frontier() - fla._byte_high_frontier()) // e_dn)
        if fit > 0:
            self.assertIsNotNone(down_end.alloc(fit * down_end.page_size))
        high_before = fla.high_wm_page
        self.assertIsNotNone(down_end.alloc(down_end.page_size))
        self.assertLess(fla.high_wm_page, high_before)  # opened HIGH side

    def test_two_pool_chain_rebalance_is_a_noop(self):
        """No float in the chain => the remedy must change nothing (the
        2-pool composites keep their exact pre-existing behavior)."""
        from test_multi_ended_allocator import (
            TestPagedMultiEndedAllocator as _PagedFixture,
        )

        inst = _PagedFixture(
            [m for m in dir(_PagedFixture) if m.startswith("test_")][0]
        )
        _pool, full, swa, _fkv, _skv = inst._build()
        v = full.alloc(full.page_size * 2)
        self.assertIsNotNone(v)
        wm = full.watermark_physical
        full._ask_float_for_room(full.page_size * 1000)  # absurd ask
        self.assertEqual(full.watermark_physical, wm)  # untouched

    def test_index_cap_guard_never_moves_data_uselessly(self):
        """When the caller's own INDEX space binds, no amount of float
        movement helps -- make_room must not be called (poisoned)."""
        from unittest import mock

        alloc = self._tri()
        alloc.alloc(4 * self.PS)
        ma = alloc.mamba_allocator
        sa = alloc.swa_attn_allocator
        huge = (ma.num_pages + 10) * ma.page_size  # beyond index space
        with mock.patch.object(
            sa, "make_room", side_effect=AssertionError("useless make_room")
        ):
            ma._ask_float_for_room(huge)


class TestComputedShortSide(unittest.TestCase):
    """`_ask_float_for_room` must open the side that MEASURES short -- never
    "the side facing full". These pin the per-side computation, including
    the coupled-ends-on-both-sides shape a DSV4-style composite
    (C128 | swa-float | C4) will need.
    """

    PS = 4

    def _tri(self):
        inst = TestTriPagedFreeGroup(
            [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
        )
        return inst._build_paged(page_size=self.PS)[1]

    def _sides(self, alloc):
        sa = alloc.swa_attn_allocator
        low = max(0, sa._byte_low_frontier() - sa._chain_high_frontier_below_bytes())
        high = max(0, sa._chain_low_frontier_above_bytes() - sa._byte_high_frontier())
        return low, high

    def test_float_share_short_opens_the_state_side(self):
        """RED-LINE: full's demand fits its band, the float's own share fits
        NEITHER band, and the state side has the larger surplus — the policy
        must open the STATE side (the float slides toward full during a
        TOKEN alloc), which the old "side facing full" policy could never do.
        """
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(6 * self.PS)  # places the float mid-region
        self.assertIsNotNone(v)
        sa = alloc.swa_attn_allocator
        fa = alloc.full_attn_allocator
        e_f, e_s = fa.entry_bytes_per_page, sa.entry_bytes_per_page

        # Position: slide the float LOW (setup uses the mechanism directly),
        # so the low band is small and the geometry below is expressible.
        b_low0, b_high0 = self._sides(alloc)
        # Two positioning moves: pack the float low (leapfrog over-opens by
        # design), then open the LOW side back to ~2 full-pages -- small
        # enough that F outgrows it, wide enough that the integer need_n
        # window below is non-empty.
        sa.make_room(side="high", min_bytes=b_low0 + b_high0 - 2 * e_f)
        sa.make_room(side="low", min_bytes=2 * e_f)

        # Find a need_n where: D_high = need_n*e_f fits band_high, F =
        # need_n*e_s exceeds BOTH surpluses, and low has the larger surplus.
        chosen = None
        for need_n in range(1, 64):
            b_low, b_high = self._sides(alloc)
            s_low = b_low  # no coupled end on the low side
            s_high = b_high - need_n * e_f
            if s_high < 0:
                break
            F = need_n * e_s
            if F > s_low and F > s_high and s_low >= s_high:
                chosen = need_n
                break
        self.assertIsNotNone(chosen, "fixture cannot express the geometry")

        calls = []
        real = sa.make_room
        with mock.patch.object(
            sa, "make_room", side_effect=lambda **kw: calls.append(kw) or real(**kw)
        ):
            alloc._ask_float_for_room(chosen * self.PS)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0]["side"], "low")  # the STATE side

    def test_full_side_short_target_matches_the_closed_form(self):
        """Equivalence: when the full side is the short one (today's only
        reachable end-shortage), the ask must equal the documented formula
        demand + max(0, F - far_surplus) + slack — i.e. the historical
        behavior is the special case, preserved."""
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(4 * self.PS)
        self.assertIsNotNone(v)
        sa, fa = alloc.swa_attn_allocator, alloc.full_attn_allocator
        e_f, e_s = fa.entry_bytes_per_page, sa.entry_bytes_per_page
        chosen = None
        for need_n in range(1, 256):
            b_low, b_high = self._sides(alloc)
            if b_high - need_n * e_f < 0 and b_low >= 0:
                chosen = need_n
                break
        self.assertIsNotNone(chosen)
        b_low, b_high = self._sides(alloc)
        F = max(0, chosen - sa._hole_pages()) * e_s
        want = chosen * e_f + max(0, F - b_low) + max(e_f, e_s)
        calls = []
        with mock.patch.object(
            sa, "make_room", side_effect=lambda **kw: calls.append(kw)
        ):
            alloc._ask_float_for_room(chosen * self.PS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["side"], "high")
        self.assertEqual(calls[0]["min_bytes"], want)

    def test_two_coupled_ends_lands_demand_on_both_sides(self):
        """DSV4 shape (C128 | float | C4): a coupled set with ends on BOTH
        sides. One-side-short must open that side; BOTH-sides-short must not
        move at all (relocation is zero-sum between the bands)."""
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(6 * self.PS)
        self.assertIsNotNone(v)
        sa, fa, ma = (
            alloc.swa_attn_allocator,
            alloc.full_attn_allocator,
            alloc.mamba_allocator,
        )
        # Synthetic coupling: the state end joins the demand vector, exactly
        # the override a DSV4-style composite would ship.
        need = lambda self, t: {
            fa: -(-t // self.page_size),
            sa: -(-t // self.page_size),
            ma: -(-t // self.page_size),
        }
        with mock.patch.object(type(alloc), "_alloc_demand", need):
            # (a) both sides short: absurd need -> both demands exceed their
            # bands -> make_room must NOT be called.
            with mock.patch.object(
                sa, "make_room", side_effect=AssertionError("zero-sum move")
            ):
                alloc._ask_float_for_room(10_000 * self.PS)

            # (b) one side short: find a need where the HIGH side (full) is
            # short while the LOW side (mamba demand) still fits.
            e_f, e_m = fa.entry_bytes_per_page, ma.entry_bytes_per_page
            chosen = None
            for need_n in range(1, 256):
                b_low, b_high = self._sides(alloc)
                if b_high - need_n * e_f < 0 and b_low - need_n * e_m >= 0:
                    chosen = need_n
                    break
            if chosen is not None:
                calls = []
                with mock.patch.object(
                    sa, "make_room", side_effect=lambda **kw: calls.append(kw)
                ):
                    alloc._ask_float_for_room(chosen * self.PS)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["side"], "high")

    def test_nothing_short_means_no_relocation(self):
        """Everything fits -> the policy must not move a single page."""
        from unittest import mock

        alloc = self._tri()
        alloc.alloc(4 * self.PS)
        sa = alloc.swa_attn_allocator
        with mock.patch.object(
            sa, "make_room", side_effect=AssertionError("needless move")
        ):
            alloc._ask_float_for_room(1)


class TestFloatPolicyTotalTarget(unittest.TestCase):
    """`make_room`'s min_bytes is a TARGET for the whole band, not a delta.

    Regression: the band-level policy passed `deficit + one page` — with a
    PARTIALLY free band that is below the current gap, so `make_room`
    no-oped and the allocation failed even though the float had room to
    slide. (Its own test missed this because it filled the band exactly,
    making deficit ≈ the whole need.) The demand-vector policy computes the
    total target, so a partial gap under-asks never.
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
        # Need = partial-gap + 3: the old delta-ask was BELOW the current
        # gap, so nothing moved and this returned None.
        got = ma.alloc((gap_slots + 3) * ma.page_size)
        self.assertIsNotNone(got, "partial-gap shortfall must relocate, not fail")
        self.assertGreater(sa.low_wm_page, low_before)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_zero_demand_bands_are_inert(self):
        """The tri's token vector carries {mamba: 0}: a zero entry must
        neither move the float for mamba's sake nor trip the index guard."""
        from unittest import mock

        alloc = self._tri()
        alloc.alloc(4 * self.PS)
        demand = alloc._alloc_demand(2 * self.PS)
        self.assertEqual(demand[alloc.mamba_allocator], 0)
        sa = alloc.swa_attn_allocator
        with mock.patch.object(
            sa, "make_room", side_effect=AssertionError("needless move")
        ):
            alloc._ask_float_for_room(1)  # nothing short -> no relocation


class TestTriDeferredAbsorption(unittest.TestCase):
    """Boundary absorption is deferred out of the per-step free and paid once
    at a quiescent point — the base allocator's model (its lazy free does "no
    boundary absorb" and `_flush` pays a single D2H). These pin WHERE it is
    now paid, and that skipping it stays merely conservative."""

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
        alloc.free_swa(v[6 * self.PS :], start_pos=6 * self.PS)  # high edge
        self.assertGreater(sa._hole_pages(), 0)  # deferred
        self.assertEqual(sa._span_pages(), span)
        moved = alloc.flush_opportunistic()
        self.assertGreater(moved, 0)
        self.assertLess(sa._span_pages(), span)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_shortfall_ladder_absorbs_before_the_deficit_math(self):
        """The zero-copy rung must run FIRST: a stale-wide span would inflate
        the rebalance deficit and buy a `make_room` relocation the shrink
        already covers."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        sa = alloc.swa_attn_allocator
        alloc.free_swa(v[6 * self.PS :], start_pos=6 * self.PS)
        self.assertGreater(sa._hole_pages(), 0)
        moves_before = len(sa._inverse_history)
        from sglang.srt.mem_cache.allocator.unified_sub_pool import _relieve_for_alloc

        _relieve_for_alloc(alloc, 1)  # the ladder
        self.assertEqual(sa._hole_pages(), 0)  # rung 0 ran
        self.assertEqual(len(sa._inverse_history), moves_before)  # zero copies

    def test_deferral_is_conservative_never_over_reports(self):
        """Availability with a stale-wide span must never EXCEED the absorbed
        value -- under-reporting is safe, over-reporting would over-admit."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        alloc.free_swa(v[6 * self.PS :], start_pos=6 * self.PS)
        deferred = alloc.available_size()
        alloc.swa_attn_allocator._flush(urgent=False)
        absorbed = alloc.available_size()
        self.assertLessEqual(deferred, absorbed)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_clean_flush_skips_the_d2h_entirely(self):
        """Only `free` can put a hole ON a boundary (alloc DRAINS holes into
        live pages; extension adds live pages), so with nothing freed since
        the last absorb the walk provably finds nothing — and must not pay
        the D2H. Steady churn with only interior holes then costs no sync."""
        from unittest import mock

        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        alloc.free_swa(v[2 * self.PS : 4 * self.PS], start_pos=2 * self.PS)
        alloc.flush_opportunistic()  # consumes the dirty flag
        sa = alloc.swa_attn_allocator
        self.assertGreater(sa._hole_pages(), 0)  # interior holes remain
        with mock.patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("tolist = D2H")
        ):
            self.assertEqual(alloc.flush_opportunistic(), 0)
            self.assertEqual(sa._flush(urgent=False), 0)

    def test_alloc_between_frees_cannot_hide_a_boundary_hole(self):
        """Soundness of the skip: an alloc drains holes and can change the
        hole COUNT back to a previously-seen value, so the flag must be armed
        by `free`, not inferred from `numel()`."""
        alloc = self._tri()
        v = alloc.alloc(8 * self.PS)
        sa = alloc.swa_attn_allocator
        alloc.free_swa(v[: 2 * self.PS], start_pos=0)  # low-edge holes
        n_after_free = sa._hole_pages()
        alloc.alloc(2 * self.PS)  # drains them back to live
        alloc.free_swa(v[6 * self.PS :], start_pos=6 * self.PS)  # high edge
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
            alloc.free_swa(v, start_pos=0)
        self.assertTrue(sa._is_frontier_transparent())
        self.assertEqual(sa._hole_pages(), 0)


class TestTriFactorySizing(unittest.TestCase):
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
            max_mamba_cache_size=4,
            model_context_len=16,
            extra_max_context_len=4,
            max_num_reqs=4,
            enable_memory_saver=False,
            enable_mamba_extra_buffer=False,
            disable_overlap_schedule=True,
            need_sort=False,
        )
        kw.update(over)
        return kw

    def test_budget_sizing_and_boot_signature(self):
        budget = 1 << 20
        bundle = init_unified_mamba_swa_pools(
            **self._factory_kwargs(unified_total_bytes=budget)
        )
        pool = bundle.unified_memory_pool
        # Buffer = budget + the state pool's bytes (budget captured AFTER the
        # state carve-out), never the token-count re-sum.
        state_bytes = 4 * pool.spec("mamba").entry_bytes()
        self.assertEqual(pool.total_bytes, budget + state_bytes)
        # Boot signature: 3 sub-pools in chain order, swa is the float.
        self.assertEqual(len(pool.sub_pool_specs), 3)
        self.assertEqual(
            [(sp.name, sp.grow_direction) for sp in pool.sub_pool_specs],
            [("mamba", "up"), ("swa", "float"), ("full", "down")],
        )

    def test_fallback_is_the_token_count_resum(self):
        bundle = init_unified_mamba_swa_pools(**self._factory_kwargs())
        pool = bundle.unified_memory_pool
        want = (
            64 * pool.spec("full").entry_bytes()
            + 32 * pool.spec("swa").entry_bytes()
            + 4 * pool.spec("mamba").entry_bytes()
        )
        self.assertEqual(pool.total_bytes, want)

    def test_bs1_floor_fails_loud_before_construction(self):
        """A budget far below one worst-case request must raise BEFORE any
        pool construction -- under-sizing is a retract LIVELOCK at runtime."""
        with self.assertRaisesRegex(RuntimeError, "bs=1 floor"):
            init_unified_mamba_swa_pools(
                **self._factory_kwargs(
                    unified_total_bytes=1024,  # << ctx * e_f alone
                    model_context_len=100_000,
                    sliding_window_size=64,
                )
            )


class TestTriPoolHardening(unittest.TestCase):
    """C1.7 pressure lanes: the planned-rebalance remedy in the alloc path
    (a mis-positioned float must not fail an alloc that fits in total bytes),
    retract-loop convergence through check_decode_capacity, and bounded copy
    traffic under alternating end pressure.
    """

    def _build(self, **kw):
        return TestUnifiedTriPool._build(self, **kw)

    def test_alloc_rebalances_a_blocking_float(self):
        # Fill much of the high band so the float (midpoint-placed) walls off
        # the low band's free bytes from `full`; the next alloc must succeed
        # by SLIDING the float, not fail while total bytes suffice.
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
        # The honest gate under-reports (no slide credit) -- asking BEYOND it
        # is what fires the rebalance remedy; the ask still fits total free
        # bytes because the LOW band holds them behind the float.
        avail = allocator.available_size()
        need = avail + 4
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
        # Simulated retract loop: requests' token blocks freed one at a time
        # until the next-step allocation fits; must converge before bs=1 and
        # never report capacity while the gate is short.
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
        self.assertGreaterEqual(allocator.available_size(), need)
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_alternating_pressure_copy_traffic_bounded(self):
        # Alternating full-grow / swa-churn cycles: total float moves stay
        # bounded (hole recycling + absorption do the steady-state work; the
        # rebalance fires only on real positional deficits).
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


class TestJointCapacityIsHonoured(unittest.TestCase):
    """`alloc(available_size())` must never fail.

    REGRESSION: the joint predicate priced the swa float's extension in RAW
    BYTES while `take_physical_pages` can only use whole pages on the float's
    OWN grid -- `_region_bounds_pages` rounds the band's low edge UP. The
    bounding frontier is a multiple of the NEIGHBOUR's entry size, which is
    unrelated to the float's, so the byte budget credited a page the grid could
    not yield and the very first alloc tripped `alloc_with_virtual`'s backstop
    assert. Swept over geometries rather than pinned to one, so a symmetric
    mistake on the FULL side would surface here too.
    """

    def _build(self, *, page_size, n_full, n_swa, n_state, lazy, specs):
        full, swa, mamba = specs
        total = (
            n_full * full.entry_bytes()
            + n_swa * swa.entry_bytes()
            + n_state * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        return pool, UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=_FakeKVCache(pool.max_slots("mamba")),
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )

    def test_fresh_boot_alloc_of_available_size_succeeds(self):
        # Geometries chosen so the mamba end's frontier (a multiple of the
        # STATE entry size) lands off the swa float's page grid -- the
        # misalignment the byte budget used to ignore.
        for page_size in (1, 2, 4):
            for fl, sl, ml in ((4, 3, 1), (4, 2, 2), (6, 3, 1), (3, 5, 2)):
                for n_full, n_swa, n_state in ((24, 16, 4), (32, 16, 8), (20, 12, 6)):
                    for lazy in (False, True):
                        specs = _tri_specs(
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

    def test_available_size_never_exceeds_the_float_page_grid(self):
        """Direct form: the joint answer, converted to float pages, must fit
        inside what `_region_bounds_pages` actually offers."""
        for page_size in (1, 4):
            specs = _tri_specs(
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


class TestFloatRelocationIsOrderedAgainstTheForward(unittest.TestCase):
    """Float relocation must settle the in-flight forward BEFORE its first copy.

    REGRESSION: `make_room` / `compact_holes` issued `move_kv_cache` and rebound
    `virtual_to_physical` with no ordering against the running forward, so the
    copy could carry pre-write bytes and the rebind then pointed every later
    reader at a destination that never received those writes -- silently wrong
    KV, no crash. The END pools guard exactly this hazard in
    `_flush(urgent=True)` via `_settle_inflight_forward`; the float had no
    `forward_stream` / `wait_event` / settle call anywhere in its body.
    """

    def _tri(self, lazy=True):
        full, swa, mamba = _tri_specs(head_num=1, head_dim=8)
        total = (
            48 * full.entry_bytes() + 32 * swa.entry_bytes() + 8 * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        alloc = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            mamba_kvcache=_FakeKVCache(pool.max_slots("mamba")),
            device=_DEV,
            full_max_total_num_tokens=48,
            swa_max_total_num_tokens=32,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=lazy,
        )
        return pool, alloc, kvcache

    def _trace(self, flt):
        """Record the order of (settle, move) on the float."""
        order = []
        real_settle = flt._settle_inflight_forward
        real_move = flt._move_pages_and_rebind

        def settle():
            order.append("settle")
            return real_settle()

        def move(src, dst):
            order.append("move")
            return real_move(src, dst)

        flt._settle_inflight_forward = settle
        flt._move_pages_and_rebind = move
        return order

    def test_make_room_settles_before_the_first_move(self):
        _pool, alloc, _kv = self._tri()
        flt = alloc.swa_attn_allocator
        # Occupy the float, then free an interior page so a relocation has
        # something to move and somewhere to move it.
        v = alloc.alloc(12)
        self.assertIsNotNone(v)
        alloc.free(v[:4])
        order = self._trace(flt)
        flt.make_room(side="low", min_bytes=flt.entry_bytes_per_page)
        self.assertIn("settle", order, "make_room never settled the forward")
        if "move" in order:
            self.assertLess(
                order.index("settle"),
                order.index("move"),
                f"a copy was issued before the settle: {order}",
            )

    def test_compact_holes_settles_before_the_first_move(self):
        _pool, alloc, _kv = self._tri()
        flt = alloc.swa_attn_allocator
        v = alloc.alloc(12)
        self.assertIsNotNone(v)
        alloc.free(v[2:6])  # interior holes, so compact_holes has work
        order = self._trace(flt)
        flt.compact_holes(retreat_side="high")
        if not order:
            self.skipTest("no holes reached compact_holes in this geometry")
        self.assertEqual(order[0], "settle", f"first action was not a settle: {order}")

    def test_the_settle_is_a_stream_wait_not_a_host_sync(self):
        """Pin the mechanism: `_settle_inflight_forward` must stream-wait, so
        the fix costs no host sync on the shortfall path."""
        src = inspect.getsource(
            mea.MultiEndedAllocator._settle_inflight_forward  # noqa: SLF001
        )
        self.assertIn("wait_event", src)
        self.assertNotIn(".item()", src)
        self.assertNotIn("synchronize()", src)


class TestFloatHoleCreditIsPerSide(unittest.TestCase):
    """A float's schedulable credit must follow the side the holes are on.

    REGRESSION: the base `_peer_drainable_hole_bytes` asks
    `_growth_side_neighbor()`, which reads `grow_direction`. A float's is
    "float", so the base fell through to `low_peer` -- it never saw the HIGH
    neighbour, and the single scalar it returned was then added to
    `max(gap_low, gap_high)`, landing a LOW neighbour's holes on the HIGH gap.
    Over-reporting `schedulable_available_size` makes the scheduler admit work
    the shortfall ladder cannot satisfy, which the caller treats as a
    memory-estimation bug.
    """

    def _float(self):
        full, swa, mamba = _tri_specs(head_num=1, head_dim=8)
        total = (
            48 * full.entry_bytes() + 32 * swa.entry_bytes() + 8 * mamba.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        alloc = UnifiedMambaSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=_FakeUnifiedSWAKVPool(pool),
            mamba_kvcache=_FakeKVCache(pool.max_slots("mamba")),
            device=_DEV,
            full_max_total_num_tokens=48,
            swa_max_total_num_tokens=32,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=True,
        )
        return alloc, alloc.swa_attn_allocator

    def test_credit_sees_both_neighbours(self):
        _alloc, flt = self._float()
        self.assertIsInstance(flt, FloatMultiEndedAllocator)
        low = flt._side_drainable_hole_bytes("low")
        high = flt._side_drainable_hole_bytes("high")
        self.assertEqual(flt._peer_drainable_hole_bytes(), max(low, high))
        # The base would have answered with the LOW side alone.
        self.assertGreaterEqual(flt._peer_drainable_hole_bytes(), high)

    def test_schedulable_never_exceeds_the_sum_of_the_two_sides(self):
        """Upper bound that the undirected scalar could violate: no side may be
        credited with the other side's holes on top of its own gap."""
        alloc, flt = self._float()
        v = alloc.alloc(10)
        self.assertIsNotNone(v)
        alloc.free(v[:3])
        epp = flt.entry_bytes_per_page
        gap_low, gap_high = flt._gap_pages()
        c_low = flt._side_drainable_hole_bytes("low") // epp
        c_high = flt._side_drainable_hole_bytes("high") // epp
        bound = (
            min(
                max(gap_low + c_low, gap_high + c_high),
                flt.num_pages - flt.min_page_index - flt._live_pages(),
            )
            + flt._hole_pages()
        ) * flt.page_size
        self.assertLessEqual(flt.schedulable_available_size(), bound)

    def test_memo_verifier_agrees_with_the_per_side_formula(self):
        """The staleness verifier recomputes through the same entry, so the
        override must not make the memo look stale."""
        _alloc, flt = self._float()
        flt.available_size()
        flt.schedulable_available_size()
        self.assertEqual(flt._byte_accounting_violations(), [])


if __name__ == "__main__":
    unittest.main()
