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
"""Tri-pool composite (`UnifiedMambaSWATokenToKVPoolAllocator`) — full KV +
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

import unittest

import torch

from sglang.srt.mem_cache.multi_ended_allocator import (
    FloatMultiEndedAllocator,
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMambaSlotAllocator,
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
        """The block whose SWA-physical pages touch neither float boundary —
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
        allocator._flush_both_for_alloc(1)
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
        """`_pages` must be USED, not merely accepted — re-deriving it is the
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
        """`free_segment` outside a free group releases reps immediately —
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

        with mock.patch.object(
            torch, "unique", side_effect=AssertionError("unique = host sync")
        ), mock.patch.object(
            torch.Tensor, "item", side_effect=AssertionError("item = host sync")
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

    def test_deferral_is_conservative_never_over_reports(self):
        """Availability with a stale-wide span must never EXCEED the absorbed
        value — under-reporting is safe, over-reporting would over-admit."""
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
        """Park-on-empty stays in `free` because it is sync-free — a float
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


if __name__ == "__main__":
    unittest.main()
