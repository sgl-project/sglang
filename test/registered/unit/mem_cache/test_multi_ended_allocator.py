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
"""Unit tests for UnifiedKVPool views and the unified sub-pool allocators:
virtual<->physical id mapping, compaction, and the SWA / Mamba composites.

Data copies go through a fake kvcache and the view math is pure torch, so the
suite runs on CPU apart from the cases that skip themselves without CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import contextlib
import random
import unittest

import torch

from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    FloatMultiEndedAllocator,
    MultiEndedAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
)
from sglang.srt.runtime_context import get_parallel

_DEV = "cpu"


def _make_mha_spec(name, grow, layer_num=2, head_num=2, head_dim=4):
    return MHASubPoolSpec(
        name=name,
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction=grow,
    )


def _make_mamba_spec(name, grow, layer_num=2):
    return MambaSubPoolSpec(
        name=name,
        layer_num=layer_num,
        conv_state_shapes=((4, 3),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2, 2, 2),
        temporal_dtype=torch.float32,
        grow_direction=grow,
    )


class _FakeKVCache:
    """Tracks, per physical slot, the virtual id whose data lives there, so a
    test can assert the data followed a compaction move."""

    def __init__(self, max_slots: int):
        # buf[p] == virtual id currently stored at physical slot p (-1 if free).
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class _RejectScalarIndexTensor:
    """Tensor proxy that rejects one-row-at-a-time mapping lookups."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        if isinstance(index, int):
            raise AssertionError("physical_to_virtual was read one row at a time")
        return self.tensor[index]

    def __setitem__(self, index, value):
        self.tensor[index] = value


class TestUnifiedKVPoolViews(unittest.TestCase):
    def test_min_slot_index_and_disjoint_bytes(self):
        full = _make_mha_spec("full", "up", layer_num=4)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        entry_max = max(full.entry_bytes(), mamba.entry_bytes())
        total = full.entry_bytes() * 64 + mamba.entry_bytes() * 16
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        for s in (full, mamba):
            min_idx = pool.min_slot_index(s.name)
            # real data of every pool begins at bytes >= entry_max
            self.assertGreaterEqual(min_idx * s.entry_bytes(), entry_max)
            self.assertGreater(pool.max_slots(s.name), min_idx)

    def test_mha_view_roundtrip(self):
        full = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        swa = _make_mha_spec("swa", "down", layer_num=2, head_num=2, head_dim=4)
        total = full.entry_bytes() * 32 + swa.entry_bytes() * 32
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device=_DEV,
            enable_memory_saver=False,
        )
        k_full, v_full = pool.mha_views_for("full")
        k_swa, v_swa = pool.mha_views_for("swa")
        self.assertEqual(len(k_full), 3)
        self.assertEqual(len(k_swa), 2)
        # "full" and "swa" share the buffer from byte 0 and grow toward each
        # other, so a low "swa" slot would byte-overlap "full" slot 5.
        swa_slot = pool.max_slots("swa") - 1
        for lyr in range(3):
            k_full[lyr][5] = float(lyr + 1)
            v_full[lyr][5] = float(-(lyr + 1))
        for lyr in range(2):
            k_swa[lyr][swa_slot] = 99.0
        for lyr in range(3):
            self.assertTrue(torch.all(k_full[lyr][5] == float(lyr + 1)))
            self.assertTrue(torch.all(v_full[lyr][5] == float(-(lyr + 1))))
        for lyr in range(2):
            self.assertTrue(torch.all(k_swa[lyr][swa_slot] == 99.0))
        # "full" slot 5 layer-0 K must not alias "full" slot 6 layer-0 K
        self.assertFalse(torch.all(k_full[0][6] == float(1)))

    def test_mamba_view_shapes(self):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=3)
        total = full.entry_bytes() * 16 + mamba.entry_bytes() * 8
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        conv_views, temporal_view = pool.mamba_views_for("mamba")
        max_slots = pool.max_slots("mamba")
        self.assertEqual(len(conv_views), 1)
        self.assertEqual(tuple(conv_views[0].shape), (3, max_slots, 4, 3))
        self.assertEqual(tuple(temporal_view.shape), (3, max_slots, 2, 2, 2))
        # roundtrip a write at (layer=1, slot=4)
        conv_views[0][1, 4] = 3.5
        temporal_view[2, 6] = -1.25
        self.assertTrue(torch.all(conv_views[0][1, 4] == 3.5))
        self.assertTrue(torch.all(temporal_view[2, 6] == -1.25))


class TestMultiEndedAllocator(unittest.TestCase):
    def _build_pair(self, n_full_slots=64, n_mamba_slots=16):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        return pool, full_alloc, mamba_alloc, full_kv, mamba_kv

    def _check_invariants(self, alloc: MultiEndedAllocator, kv: _FakeKVCache):
        v2p = alloc.virtual_to_physical
        p2v = alloc.physical_to_virtual
        # live virtual ids = those with v2p != -1, excluding the reserved id 0.
        live_v = [
            v for v in range(1, alloc.num_virtual_ids) if int(v2p[v].item()) != -1
        ]
        # mutual-inverse on the live set
        for v in live_v:
            p = int(v2p[v].item())
            self.assertEqual(int(p2v[p].item()), v, f"p2v[{p}] != {v}")
            # data followed any relocations
            self.assertEqual(int(kv.buf[p].item()), v, f"kv.buf[{p}] != {v}")
        # allocated physical range is hole-free + matches live count
        if alloc.grow_direction == "up":
            alloc_lo, alloc_hi = alloc.min_slot_index, alloc.watermark_physical
        else:
            alloc_lo, alloc_hi = alloc.watermark_physical + 1, alloc.max_slots
        self.assertEqual(alloc_hi - alloc_lo, len(live_v))
        for p in range(alloc_lo, alloc_hi):
            self.assertNotEqual(int(p2v[p].item()), -1, f"hole at physical {p}")
        # free virtual ids | live = [min_slot_index, max_slots)
        free_set = set(int(x) for x in alloc.free_virtual_ids.tolist())
        self.assertEqual(
            free_set | set(live_v),
            set(range(alloc.min_slot_index, alloc.max_slots)),
        )
        self.assertEqual(free_set & set(live_v), set())

    def _alloc(self, alloc: MultiEndedAllocator, kv: _FakeKVCache, n: int):
        avail = alloc.available_size()
        v = alloc.alloc(n)
        if n > avail:
            self.assertIsNone(v)
            return None
        self.assertIsNotNone(v)
        self.assertEqual(int(v.numel()), n)
        # stamp the data marker at each new physical slot
        p = alloc.virtual_to_physical[v]
        kv.buf[p] = v
        return v

    def _free(self, alloc: MultiEndedAllocator, kv: _FakeKVCache, v: torch.Tensor):
        p = alloc.virtual_to_physical[v]
        kv.buf[p] = -1  # the freed virtual id's data is gone
        alloc.free(v)

    def test_basic_alloc_free_compaction(self):
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair()
        # alloc three batches on the full side
        a = self._alloc(full_alloc, full_kv, 3)
        b = self._alloc(full_alloc, full_kv, 5)
        c = self._alloc(full_alloc, full_kv, 2)
        self._check_invariants(full_alloc, full_kv)
        # free the middle batch -> forces eager compaction (boundary slots move in)
        self._free(full_alloc, full_kv, b)
        self._check_invariants(full_alloc, full_kv)
        # `a` and `c` virtual ids unchanged; their physical slots may have moved.
        for v in a.tolist() + c.tolist():
            self.assertNotEqual(int(full_alloc.virtual_to_physical[v].item()), -1)
        # free the boundary batch (no relocation needed)
        self._free(full_alloc, full_kv, c)
        self._check_invariants(full_alloc, full_kv)
        self._free(full_alloc, full_kv, a)
        self._check_invariants(full_alloc, full_kv)
        self.assertEqual(full_alloc.allocated_count(), 0)

    def test_byte_frontier_coordination(self):
        # full has 8 slots' worth of bytes; mamba's entry is larger, so a few
        # mamba allocs should shrink full's available_size below its slot headroom.
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair(
            n_full_slots=8, n_mamba_slots=8
        )
        full_avail0 = full_alloc.available_size()
        self._alloc(mamba_alloc, mamba_kv, 3)
        self.assertLess(full_alloc.available_size(), full_avail0)
        # over-alloc the full side -> None
        self.assertIsNone(full_alloc.alloc(full_alloc.available_size() + 1))

    def test_randomized(self):
        rng = random.Random(0xC0FFEE)
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair(
            n_full_slots=48, n_mamba_slots=24
        )
        live_full = []  # list of virtual-id tensors still allocated
        live_mamba = []
        for _ in range(400):
            side = rng.random() < 0.6  # 60% full
            alloc, kv, live = (
                (full_alloc, full_kv, live_full)
                if side
                else (mamba_alloc, mamba_kv, live_mamba)
            )
            if rng.random() < 0.55 or not live:
                n = rng.randint(1, 5)
                v = self._alloc(alloc, kv, n)
                if v is not None:
                    live.append(v)
            else:
                idx = rng.randrange(len(live))
                v = live.pop(idx)
                self._free(alloc, kv, v)
            self._check_invariants(full_alloc, full_kv)
            self._check_invariants(mamba_alloc, mamba_kv)
        # drain
        for live, alloc, kv in (
            (live_full, full_alloc, full_kv),
            (live_mamba, mamba_alloc, mamba_kv),
        ):
            for v in live:
                self._free(alloc, kv, v)
            self._check_invariants(alloc, kv)
            self.assertEqual(alloc.allocated_count(), 0)

    def _build_lazy_full(self, n_full_slots=64, n_mamba_slots=16, move_cap=2):
        """A lazy-compaction 'full' (grow-up) allocator + its peer, with a
        small per-call move cap so every flush is partial."""
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=True,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=True,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        full_alloc._lazy_max_moves_per_call = move_cap
        return pool, full_alloc, full_kv

    def test_lazy_retract_churn_no_ghost(self):
        """Regression: heavy free/flush churn in lazy mode must never leave a
        'ghost' page -- p2v < 0 in neither _free_phys_pages nor _pending_reuse."""
        rng = random.Random(0x5373)
        _, alloc, kv = self._build_lazy_full(n_full_slots=64, move_cap=2)
        live = []
        for _ in range(600):
            avail = alloc.available_size()
            if (rng.random() < 0.55 and avail > 0) or not live:
                n = rng.randint(1, min(5, max(1, avail)))
                v = self._alloc(alloc, kv, n)
                if v is not None:
                    live.append(v)
            else:
                v = live.pop(rng.randrange(len(live)))
                self._free(alloc, kv, v)  # often non-boundary -> a hole
            if rng.random() < 0.5:
                alloc.flush_opportunistic()  # partial flush (move_cap)
        # Drain to quiescence; must end empty AND ghost-free.
        for v in live:
            self._free(alloc, kv, v)
        for _ in range(64):
            alloc.flush_opportunistic()
        self.assertEqual(alloc.allocated_count(), 0)

    def test_double_free_raises(self):
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 3)
        self._free(full_alloc, full_kv, v)
        with self.assertRaises(AssertionError):
            full_alloc.free(v)

    # -- `out=` parameter regression tests --

    def test_translate_kv_loc_with_out_writes_inplace(self):
        """Regression: `translate_kv_loc(virt, out=buf)` must write into `buf`
        and preserve `buf.data_ptr()` -- the cuda-graph capture invariant."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        ptr_before = buf.data_ptr()
        ret = full_alloc.translate_kv_loc(v, out=buf)
        self.assertIs(ret, buf, "must return the `out=` buffer, not a fresh tensor")
        self.assertEqual(
            buf.data_ptr(), ptr_before, "out= buffer's data_ptr must be stable"
        )
        # Result matches v2p directly (page_size == 1 here)
        expected = full_alloc.virtual_to_physical[v]
        self.assertTrue(bool((buf == expected).all().item()))

    def test_translate_kv_loc_out_guards(self):
        """REGRESSION: a malformed `out=` buffer raises AssertionError -- it
        must match both the v2p dtype the gather writes and the input shape."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        with self.subTest(guard="dtype"):
            wrong_dtype = torch.empty(v.shape, dtype=torch.int32, device=_DEV)
            with self.assertRaises(AssertionError):
                full_alloc.translate_kv_loc(v, out=wrong_dtype)
        with self.subTest(guard="shape"):
            wrong_shape = torch.empty((v.numel() + 1,), dtype=torch.int64, device=_DEV)
            with self.assertRaises(AssertionError):
                full_alloc.translate_kv_loc(v, out=wrong_shape)

    # `index_select(v2p, 0, virt, out=out)` rejects aliasing between `index`
    # and `out`, so the in-place form must gather into a temporary first.
    def test_translate_kv_loc_with_out_aliasing_input(self):
        """REGRESSION: in-place form `translate_kv_loc(buf, out=buf)` must
        succeed and produce identical results to the no-out form."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v_orig = self._alloc(full_alloc, full_kv, 5).clone()
        # Save the expected output (no-out form) before mutating `buf`.
        expected = full_alloc.translate_kv_loc(v_orig)
        # Now exercise the aliasing form: buf serves as BOTH input and out.
        buf = v_orig.clone()
        ptr_before = buf.data_ptr()
        ret = full_alloc.translate_kv_loc(buf, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(
            buf.data_ptr(),
            ptr_before,
            "out= buffer's data_ptr must be stable (cuda-graph invariant)",
        )
        self.assertTrue(
            bool((buf == expected).all().item()),
            "in-place result must equal no-out result",
        )

    # A captured cuda-graph input buffer can hold virtual ids tombstoned (-1)
    # between replays; slot 0 is the sink, as bytes [0, entry_max) hold no data.
    def test_translate_kv_loc_clamps_tombstoned_v2p(self):
        """`translate_kv_loc` must clamp `v2p[v] == -1` entries to 0 (the
        padding sink). Required for cuda-graph capture safety."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        # Inject a tombstone directly, not through `free` (which would also
        # touch p2v and run compaction).
        v_tombstoned = int(v[2].item())
        full_alloc.virtual_to_physical[v_tombstoned] = -1
        # No-out form: result must clamp.
        out = full_alloc.translate_kv_loc(v)
        self.assertTrue(
            bool((out >= 0).all().item()),
            f"translate_kv_loc must clamp tombstoned entries to >=0, got {out.tolist()}",
        )
        self.assertEqual(
            int(out[2].item()),
            0,
            "tombstoned virtual id must map to slot 0 (padding sink)",
        )
        # out= form (the captured-graph path) must clamp in place too.
        buf = torch.empty_like(v)
        ret = full_alloc.translate_kv_loc(v, out=buf)
        self.assertIs(ret, buf)
        self.assertTrue(
            bool((buf >= 0).all().item()),
            "out= path must clamp tombstoned entries",
        )
        self.assertEqual(int(buf[2].item()), 0)

    def test_slot_zero_sink_invariant_survives_churn(self):
        """Virtual 0 must stay mapped to physical 0 through alloc/free/compaction
        churn: cuda-graph capture copies a zero-filled static buffer instead of
        translating, so a nonzero v2p[0] would send pad lanes to a live slot."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        zeros = torch.zeros(4, dtype=torch.int64)

        self.assertEqual(int(full_alloc.virtual_to_physical[0].item()), 0)
        self.assertTrue(torch.equal(full_alloc.translate_kv_loc(zeros), zeros))

        # Churn: allocate, free interior (forces compaction moves), re-allocate.
        a = self._alloc(full_alloc, full_kv, 6)
        b = self._alloc(full_alloc, full_kv, 6)
        self._free(full_alloc, full_kv, a)
        c = self._alloc(full_alloc, full_kv, 4)
        self._free(full_alloc, full_kv, b)
        self._free(full_alloc, full_kv, c)

        self.assertEqual(int(full_alloc.virtual_to_physical[0].item()), 0)
        self.assertTrue(torch.equal(full_alloc.translate_kv_loc(zeros), zeros))


# ---------------------------------------------------------------------------
# Shared SWA composite -- unit tests
# ---------------------------------------------------------------------------


class _FakeUnifiedSWAKVPool:
    """Minimal stand-in for `UnifiedSWAKVPool`: a real one would construct
    `UnifiedMHATokenToKVPool`, which is heavier than these tests need."""

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


class TestUnifiedSWATokenToKVPoolAllocator(unittest.TestCase):
    """The SWA composite: joint byte-budget, slot-conservation, `free_swa`
    tombstone semantics, and divergent compaction of the two sub-pools."""

    def _build(
        self,
        n_full_slots=32,
        n_swa_slots=16,
        full_layer_num=4,
        swa_layer_num=2,
        head_num=2,
        head_dim=4,
    ):
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=full_layer_num,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=swa_layer_num,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        total = (
            n_full_slots * full_spec.entry_bytes()
            + n_swa_slots * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full_slots,
            swa_max_total_num_tokens=n_swa_slots,
            need_sort=False,
            forward_stream=None,
        )
        return pool, allocator, kvcache

    def _alloc(self, allocator, kvcache, n):
        """Allocate N virtual ids; stamp the data marker on both sub-pools."""
        v = allocator.alloc(n)
        if v is None:
            return None
        full_phys = allocator.full_attn_allocator.virtual_to_physical[v]
        swa_phys = allocator.swa_attn_allocator.virtual_to_physical[v]
        kvcache.full_kv_pool.buf[full_phys] = v
        kvcache.swa_kv_pool.buf[swa_phys] = v
        return v

    def _free(self, allocator, kvcache, v):
        """Erase markers on both sub-pools (mirroring compaction's
        no-data-at-a-freed-slot invariant), then call the composite's free."""
        full_phys = allocator.full_attn_allocator.virtual_to_physical[v]
        swa_phys = allocator.swa_attn_allocator.virtual_to_physical[v]
        # erase only the LIVE swa entries (`free_swa` may have already
        # tombstoned some of `v`).
        valid_swa = swa_phys[swa_phys >= 0]
        kvcache.full_kv_pool.buf[full_phys] = -1
        kvcache.swa_kv_pool.buf[valid_swa] = -1
        allocator.free(v)

    def _check_sub_pool_invariants(self, sub, kv):
        """Per-sub-pool: v2p/p2v mutual inverse on the live set, hole-free
        allocated band, data followed relocations."""
        v2p = sub.virtual_to_physical
        p2v = sub.physical_to_virtual
        live_v = [v for v in range(1, sub.num_virtual_ids) if int(v2p[v].item()) != -1]
        for v in live_v:
            p = int(v2p[v].item())
            self.assertEqual(int(p2v[p].item()), v)
            # data marker followed any relocation
            self.assertEqual(int(kv.buf[p].item()), v)
        if sub.grow_direction == "up":
            lo, hi = sub.min_slot_index, sub.watermark_physical
        else:
            lo, hi = sub.watermark_physical + 1, sub.max_slots
        self.assertEqual(hi - lo, len(live_v))
        for p in range(lo, hi):
            self.assertNotEqual(int(p2v[p].item()), -1)

    # 1. Both peers hold a physical slot per virtual after composite alloc.
    def test_swa_alloc_both_peers_hold(self):
        _, allocator, _ = self._build()
        v = allocator.alloc(3)
        self.assertIsNotNone(v)
        self.assertEqual(int(v.numel()), 3)
        full_v2p = allocator.full_attn_allocator.virtual_to_physical
        swa_v2p = allocator.swa_attn_allocator.virtual_to_physical
        for vi in v.tolist():
            self.assertGreaterEqual(int(full_v2p[vi].item()), 0)
            self.assertGreaterEqual(int(swa_v2p[vi].item()), 0)
        # Full sub-pool is id-owner -> the minted ids are out of free_virtual_ids.
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertTrue(set(v.tolist()).isdisjoint(free_full))
        # Swa sub-pool is non-owner -> free_virtual_ids is None.
        self.assertIsNone(allocator.swa_attn_allocator.free_virtual_ids)

    # 2. Composite `free` releases both sub-pools' v2p and recycles the virtual.
    def test_swa_free_releases_both(self):
        _, allocator, kvcache = self._build()
        v = self._alloc(allocator, kvcache, 3)
        self._free(allocator, kvcache, v)
        for vi in v.tolist():
            self.assertEqual(
                int(allocator.full_attn_allocator.virtual_to_physical[vi].item()), -1
            )
            self.assertEqual(
                int(allocator.swa_attn_allocator.virtual_to_physical[vi].item()), -1
            )
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertTrue(set(v.tolist()).issubset(free_full))

    # 3. `free_swa` tombstones swa side only; virtual + full-physical stay live.
    def test_swa_free_swa_keeps_virtual_alive(self):
        _, allocator, kvcache = self._build()
        v = self._alloc(allocator, kvcache, 3)
        # Tombstone the middle one. Erase its swa marker first (compaction
        # will run inside `free_swa`).
        target = v[1:2]
        target_swa = allocator.swa_attn_allocator.virtual_to_physical[target]
        kvcache.swa_kv_pool.buf[target_swa] = -1
        allocator.free_swa(target)
        tgt = int(target.item())
        # full side still bound:
        self.assertGreaterEqual(
            int(allocator.full_attn_allocator.virtual_to_physical[tgt].item()), 0
        )
        # swa side tombstoned:
        self.assertEqual(
            int(allocator.swa_attn_allocator.virtual_to_physical[tgt].item()), -1
        )
        # NOT recycled to the id-owner's free list yet:
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertNotIn(tgt, free_full)
        # composite `free` of the same virtual still works (filters out
        # already-tombstoned on the swa side).
        full_phys = int(allocator.full_attn_allocator.virtual_to_physical[tgt].item())
        kvcache.full_kv_pool.buf[full_phys] = -1
        allocator.free(target)
        # now in free list:
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertIn(tgt, free_full)

    def test_swa_free_full_defers_inside_a_free_group(self):
        """The full-only release joins the barrier, like `free`."""
        _, allocator, kvcache = self._build()
        v = self._alloc(allocator, kvcache, 3)
        target = v[1:2]
        tgt = int(target.item())
        # Tombstone the swa side, erasing each marker before its release
        # (compaction runs inside both).
        target_swa = allocator.swa_attn_allocator.virtual_to_physical[target]
        kvcache.swa_kv_pool.buf[target_swa] = -1
        allocator.free_swa(target)
        full_phys = int(allocator.full_attn_allocator.virtual_to_physical[tgt].item())
        kvcache.full_kv_pool.buf[full_phys] = -1

        allocator.free_group_begin()
        allocator.free_full(target)
        deferred = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertNotIn(tgt, deferred)

        allocator.free_group_end()
        drained = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertIn(tgt, drained)

    # 4. Compaction diverges between the two sub-pools (each runs its own).
    def test_swa_compaction_diverges_physical_layout(self):
        _, allocator, kvcache = self._build()
        a = self._alloc(allocator, kvcache, 1)
        b = self._alloc(allocator, kvcache, 1)
        c = self._alloc(allocator, kvcache, 1)
        # Snapshot swa-side physical for c BEFORE we free_swa(b).
        c_swa_before = int(allocator.swa_attn_allocator.virtual_to_physical[c].item())
        c_full_before = int(allocator.full_attn_allocator.virtual_to_physical[c].item())
        # Tombstone b on swa only.
        b_swa = allocator.swa_attn_allocator.virtual_to_physical[b]
        kvcache.swa_kv_pool.buf[b_swa] = -1
        allocator.free_swa(b)
        # c's full-physical UNCHANGED (full side did not compact):
        self.assertEqual(
            int(allocator.full_attn_allocator.virtual_to_physical[c].item()),
            c_full_before,
        )
        # c's swa-physical MUST have moved: b is interior to swa's allocated
        # band, so freeing it relocates c into b's slot.
        c_swa_after = int(allocator.swa_attn_allocator.virtual_to_physical[c].item())
        self.assertNotEqual(c_swa_after, c_swa_before)
        # Per-sub-pool invariants still hold.
        self._check_sub_pool_invariants(
            allocator.full_attn_allocator, kvcache.full_kv_pool
        )
        self._check_sub_pool_invariants(
            allocator.swa_attn_allocator, kvcache.swa_kv_pool
        )

    # 5. Byte-frontier coordination: available_size shrinks as the peer grows.
    def test_swa_byte_frontier_coordination(self):
        _, allocator, kvcache = self._build(n_full_slots=8, n_swa_slots=8)
        avail0 = allocator.available_size()
        # Allocate enough that the joint budget visibly tightens.
        self._alloc(allocator, kvcache, 3)
        self.assertLess(allocator.available_size(), avail0)
        # Joint budget enforcement: over-alloc returns None.
        self.assertIsNone(allocator.alloc(allocator.available_size() + 1))

    # 6. Randomized stress -- invariants under mixed alloc / free / free_swa.
    def test_swa_randomized_alloc_free_freeswa(self):
        rng = random.Random(0xBADBEE)
        _, allocator, kvcache = self._build(
            n_full_slots=48, n_swa_slots=24, full_layer_num=3, swa_layer_num=3
        )
        live = []  # list of (virtual-id tensor)
        for _ in range(400):
            r = rng.random()
            if r < 0.5 or not live:  # alloc
                n = rng.randint(1, 4)
                v = self._alloc(allocator, kvcache, n)
                if v is not None:
                    live.append(("live", v))
            elif r < 0.8:  # composite free
                idx = rng.randrange(len(live))
                kind, v = live.pop(idx)
                self._free(allocator, kvcache, v)
            else:  # free_swa on some entries
                idx = rng.randrange(len(live))
                kind, v = live[idx]
                if kind != "live":
                    continue
                # Tombstone all of v on swa only.
                swa_phys = allocator.swa_attn_allocator.virtual_to_physical[v]
                kvcache.swa_kv_pool.buf[swa_phys] = -1
                allocator.free_swa(v)
                live[idx] = ("swa_tomb", v)
            # Invariants after every op.
            self._check_sub_pool_invariants(
                allocator.full_attn_allocator, kvcache.full_kv_pool
            )
            self._check_sub_pool_invariants(
                allocator.swa_attn_allocator, kvcache.swa_kv_pool
            )
            # The leak view is `_conserve_*`; public `*_available_size()` is
            # `min(conserve, schedulable)` and can be strictly smaller.
            self.assertEqual(
                allocator._conserve_full_available_size(),
                allocator._full_max_total_num_tokens
                - allocator.full_attn_allocator.allocated_count(),
            )
            self.assertEqual(
                allocator._conserve_swa_available_size(),
                allocator._swa_max_total_num_tokens
                - allocator.swa_attn_allocator.allocated_count(),
            )
        # Drain.
        for _, v in live:
            self._free(allocator, kvcache, v)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)

    # -- `out=` parameter regression tests for the SWA composite --

    def test_swa_translate_loc_from_full_to_swa_with_out_writes_inplace(self):
        """Regression: `translate_loc_from_full_to_swa(v, out=buf)` must write
        into `buf` and preserve its data_ptr; `out=` is int64, as every id is."""
        _, allocator, _ = self._build()
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        ptr_before = buf.data_ptr()
        ret = allocator.translate_loc_from_full_to_swa(v, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(buf.data_ptr(), ptr_before)
        # Byte-identical to the no-out form:
        no_out = allocator.translate_loc_from_full_to_swa(v)
        self.assertEqual(no_out.dtype, torch.int64)
        self.assertTrue(bool((buf == no_out).all().item()))

    def test_swa_translate_loc_from_full_to_swa_dtype_assertion(self):
        """Regression: a wrong-dtype `out=` (int32) must raise; the allocator
        emits int64 and consumers narrow at their own buffer."""
        _, allocator, _ = self._build()
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        wrong_dtype = torch.empty(v.shape, dtype=torch.int32, device=_DEV)
        with self.assertRaises(AssertionError):
            allocator.translate_loc_from_full_to_swa(v, out=wrong_dtype)

    # SWA side of the tombstone clamp: a tombstoned `v2p_swa[v] == -1` would
    # make the captured kernel read `swa_k_buffer[-1]`.
    def test_swa_translate_loc_from_full_to_swa_clamps_tombstoned(self):
        _, allocator, _ = self._build()
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        # Inject a tombstone on the swa side at one of the live virtual ids.
        v_tomb = int(v[1].item())
        allocator.swa_attn_allocator.virtual_to_physical[v_tomb] = -1
        # No-out form: result must be int64 AND every entry >= 0.
        out = allocator.translate_loc_from_full_to_swa(v)
        self.assertEqual(out.dtype, torch.int64)
        self.assertTrue(
            bool((out >= 0).all().item()),
            "translate_loc_from_full_to_swa must clamp tombstoned to >=0",
        )
        self.assertEqual(int(out[1].item()), 0)
        # out= form must also clamp.
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        ret = allocator.translate_loc_from_full_to_swa(v, out=buf)
        self.assertIs(ret, buf)
        self.assertTrue(bool((buf >= 0).all().item()))
        self.assertEqual(int(buf[1].item()), 0)

    def test_swa_slot_zero_sink_invariant_survives_churn(self):
        """Both maps must send virtual 0 to physical 0 through churn: cuda-graph
        capture zero-fills its static buffers instead of translating, which is
        only equivalent while slot 0 stays the sink in both sub-pools."""
        _, allocator, kvcache = self._build()
        zeros64 = torch.zeros(4, dtype=torch.int64)

        def check():
            self.assertTrue(torch.equal(allocator.translate_kv_loc(zeros64), zeros64))
            self.assertTrue(
                torch.equal(allocator.translate_loc_from_full_to_swa(zeros64), zeros64)
            )

        check()
        a = self._alloc(allocator, kvcache, 5)
        b = self._alloc(allocator, kvcache, 5)
        allocator.free_swa(a)  # tombstone swa side only
        self._free(allocator, kvcache, b)  # full free (compaction on both)
        self._free(allocator, kvcache, a)
        c = self._alloc(allocator, kvcache, 3)
        self._free(allocator, kvcache, c)
        check()


# ---------------------------------------------------------------------------
# page_size > 1 -- paged unit tests
# ---------------------------------------------------------------------------


class TestPagedMultiEndedAllocator(unittest.TestCase):
    """`MultiEndedAllocator(page_size=8)`: free-list, v2p/p2v and compaction are
    page-granular, while the external API stays in token ids as at page_size 1."""

    PAGE_SIZE = 8

    def _build(self, n_full_pages=16, n_swa_pages=8, full_layer_num=2, swa_layer_num=2):
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=full_layer_num,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=swa_layer_num,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        # entry_bytes_per_page = layer_num * (k_row + v_row) * page_size
        # We size the buffer to fit `n_full_pages` full-pages + `n_swa_pages`
        # swa-pages (token-equivalent: n_*_pages * page_size).
        total = (
            n_full_pages * self.PAGE_SIZE * full_spec.entry_bytes()
            + n_swa_pages * self.PAGE_SIZE * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        swa_kv = _FakeKVCache(pool.max_slots("swa"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            page_size=self.PAGE_SIZE,
        )
        swa_alloc = MultiEndedAllocator(
            kvcache=swa_kv,
            unified_buffer=pool,
            sub_pool_name="swa",
            device=_DEV,
            is_id_owner=True,
            page_size=self.PAGE_SIZE,
        )
        full_alloc.bind_peer(swa_alloc)
        swa_alloc.bind_peer(full_alloc)
        return pool, full_alloc, swa_alloc, full_kv, swa_kv

    def _stamp_tokens(
        self, alloc: MultiEndedAllocator, kv: _FakeKVCache, v_tokens: torch.Tensor
    ):
        """Stamp each returned token's own id into `kv.buf` at its physical token."""
        if v_tokens.numel() == 0:
            return
        ps = alloc.page_size
        virt_pages = v_tokens // ps
        offsets = v_tokens % ps
        phys_pages = alloc.virtual_to_physical[virt_pages]
        phys_tokens = phys_pages * ps + offsets
        kv.buf[phys_tokens] = v_tokens

    def _check_invariants(
        self, alloc: MultiEndedAllocator, kv: _FakeKVCache, stamped_tokens: dict
    ):
        v2p = alloc.virtual_to_physical
        p2v = alloc.physical_to_virtual
        ps = alloc.page_size
        # Live virtual pages (excluding the reserved padding page 0).
        live_v_pages = [
            v for v in range(1, alloc.num_pages) if int(v2p[v].item()) != -1
        ]
        # Mutual inverse on the live page set.
        for v_page in live_v_pages:
            p_page = int(v2p[v_page].item())
            self.assertEqual(
                int(p2v[p_page].item()),
                v_page,
                f"p2v[{p_page}] != {v_page}",
            )
        # Allocated physical-page range is hole-free + matches live count.
        if alloc.grow_direction == "up":
            alloc_lo, alloc_hi = alloc.min_page_index, alloc.watermark_physical
        else:
            alloc_lo, alloc_hi = (
                alloc.watermark_physical + 1,
                alloc.num_pages,
            )
        self.assertEqual(alloc_hi - alloc_lo, len(live_v_pages))
        for p_page in range(alloc_lo, alloc_hi):
            self.assertNotEqual(
                int(p2v[p_page].item()), -1, f"hole at physical page {p_page}"
            )
        # Free virtual page ids | live = [min_page_index, num_pages).
        free_set = set(int(x) for x in alloc.free_virtual_ids.tolist())
        self.assertEqual(
            free_set | set(live_v_pages),
            set(range(alloc.min_page_index, alloc.num_pages)),
        )
        self.assertEqual(free_set & set(live_v_pages), set())
        # For every token we stamped, verify data followed any relocations.
        for v_tok, mark in stamped_tokens.items():
            v_page = v_tok // ps
            offset = v_tok % ps
            p_page_t = int(v2p[v_page].item())
            if p_page_t == -1:
                continue  # was freed; don't check
            phys_tok = p_page_t * ps + offset
            self.assertEqual(
                int(kv.buf[phys_tok].item()),
                mark,
                f"data drift: stamped {mark} at virtual token {v_tok} "
                f"(page {v_page}+offset {offset}) — found {int(kv.buf[phys_tok].item())}",
            )

    # 1. alloc(N) returns N TOKEN ids that are page-aligned.
    def test_paged_alloc_token_aligned(self):
        _, full_alloc, swa_alloc, full_kv, swa_kv = self._build()
        v = full_alloc.alloc(16)  # 2 pages x 8 tokens
        self.assertIsNotNone(v)
        self.assertEqual(int(v.numel()), 16)
        # The output must consist of exactly 2 contiguous page-ranges.
        v_pages = sorted(set((v // self.PAGE_SIZE).tolist()))
        self.assertEqual(len(v_pages), 2)
        for p in v_pages:
            page_tokens = sorted(int(t) for t in v if t // self.PAGE_SIZE == p)
            self.assertEqual(
                page_tokens,
                [p * self.PAGE_SIZE + i for i in range(self.PAGE_SIZE)],
                "Page contents should be contiguous token ids",
            )

    # 2. alloc(N) requires N % page_size == 0.
    def test_paged_alloc_non_aligned_raises(self):
        _, full_alloc, _, _, _ = self._build()
        with self.assertRaises(AssertionError):
            full_alloc.alloc(5)  # not a multiple of 8

    # 4. Compaction relocates a whole page at once (data follows).
    def test_paged_compaction_relocates_whole_pages(self):
        _, full_alloc, _, full_kv, _ = self._build()
        stamped = {}
        # Alloc 3 pages worth of tokens.
        a = full_alloc.alloc(self.PAGE_SIZE)  # tokens of page X
        b = full_alloc.alloc(self.PAGE_SIZE)  # tokens of page Y (middle)
        c = full_alloc.alloc(self.PAGE_SIZE)  # tokens of page Z

        # Stamp each token with its own virtual id so in-page siblings differ.
        for v in (a, b, c):
            self._stamp_tokens(full_alloc, full_kv, v)
            for t in v.tolist():
                stamped[t] = t

        # Free the MIDDLE page (token ids of `b`). This forces a compaction
        # where page `c` (boundary, grow-up) relocates into page `b`'s slot.
        full_alloc.free(b)
        for t in b.tolist():
            stamped.pop(t, None)
        self._check_invariants(full_alloc, full_kv, stamped)
        # `a` and `c` pages must still be live.
        for t in a.tolist():
            v_page = t // self.PAGE_SIZE
            self.assertNotEqual(int(full_alloc.virtual_to_physical[v_page].item()), -1)
        for t in c.tolist():
            v_page = t // self.PAGE_SIZE
            self.assertNotEqual(int(full_alloc.virtual_to_physical[v_page].item()), -1)

    # 5. Regression: `allocated_count()` reports TOKENS, not pages; the leak
    # invariant `available + evictable + ... == total` is all in tokens.
    def test_paged_free_unique_by_page(self):
        _, full_alloc, _, full_kv, _ = self._build()
        a = full_alloc.alloc(self.PAGE_SIZE * 2)  # 2 pages = 2*PS tokens
        allocated_count_before = full_alloc.allocated_count()
        # `allocated_count()` returns TOKENS.
        self.assertEqual(allocated_count_before, 2 * self.PAGE_SIZE)
        # Internal page count.
        self.assertEqual(full_alloc._allocated_pages(), 2)
        # The caller must pass page-coherent ranges; `a` covers both pages whole.
        full_alloc.free(a)
        self.assertEqual(full_alloc.allocated_count(), 0)
        self.assertEqual(full_alloc._allocated_pages(), 0)

    # 6. take_physical overflow check (grow-up direction).
    def test_paged_take_physical_overflow_check(self):
        _, full_alloc, _, _, _ = self._build(n_full_pages=4)
        avail = full_alloc.available_size()
        n_pages = avail // self.PAGE_SIZE
        result = full_alloc.take_physical(n_pages * self.PAGE_SIZE)
        self.assertIsNotNone(result)
        # Now one more page would overflow.
        overflow = full_alloc.take_physical(self.PAGE_SIZE)
        self.assertIsNone(overflow, "Overflow should return None, not crash")

    # 7. SWA composite joint byte-budget in page units.
    def test_paged_swa_joint_byte_budget(self):
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        n_full_pages, n_swa_pages = 8, 8
        total = (
            n_full_pages * self.PAGE_SIZE * full_spec.entry_bytes()
            + n_swa_pages * self.PAGE_SIZE * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full_pages * self.PAGE_SIZE,
            swa_max_total_num_tokens=n_swa_pages * self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            need_sort=False,
            forward_stream=None,
        )
        # available_size() is in TOKENS; the joint budget charges both sub-pools.
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        entry_sum_pp = fa.entry_bytes_per_page + sa.entry_bytes_per_page
        gap = sa._byte_low_frontier() - fa._byte_high_frontier()
        expected_pages_by_bytes = gap // entry_sum_pp
        expected = (
            min(
                expected_pages_by_bytes,
                fa.num_pages - fa.min_page_index,
                sa.num_pages - sa.min_page_index,
            )
            * self.PAGE_SIZE
        )
        self.assertEqual(allocator.available_size(), expected)
        # The joint cost is heavier than either single-side cost, so the joint
        # budget can never exceed either sub-pool's own available_size.
        self.assertLessEqual(
            allocator.available_size(),
            min(fa.available_size(), sa.available_size()),
        )

    # 9. Regression: alloc_extend must bind v2p / p2v; an unbound page leaves
    # v2p at -1, and translate_kv_loc then emits negative (OOB) token ids.
    def test_paged_alloc_extend_binds_v2p_p2v(self):
        from sglang.srt.mem_cache.allocator import unified_sub_pool as mea_mod

        _, full_alloc, _, _, _ = self._build()
        PS = self.PAGE_SIZE
        free_before = full_alloc.free_virtual_ids.clone()
        watermark_before = full_alloc.watermark_physical
        allocated_count_before = full_alloc.allocated_count()

        # Stub the kernel: driving the real Triton one needs a GPU, and the
        # binding contract holds regardless of what it writes to out_indices.
        original_kernel = mea_mod.alloc_extend_kernel

        class _NoOpKernelGrid:
            def __getitem__(self, _grid):
                return self

            def __call__(self, *a, **kw):
                pass

        mea_mod.alloc_extend_kernel = _NoOpKernelGrid()
        try:
            # bs=1, prefix=0, seq=2 pages worth, so num_new_pages=2.
            prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEV)
            prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
            seq_lens = torch.tensor([2 * PS], dtype=torch.int64, device=_DEV)
            seq_lens_cpu = torch.tensor([2 * PS], dtype=torch.int64)
            last_loc = torch.tensor([-1], dtype=torch.int64, device=_DEV)

            out = full_alloc.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                2 * PS,
                num_new_pages=2,
            )
        finally:
            mea_mod.alloc_extend_kernel = original_kernel

        self.assertIsNotNone(out)
        consumed_pages = free_before[:2]
        v2p_values = full_alloc.virtual_to_physical[consumed_pages]
        for v_page, p_page in zip(consumed_pages.tolist(), v2p_values.tolist()):
            self.assertNotEqual(
                p_page,
                -1,
                f"REGRESSION: virtual page {v_page} not bound after "
                f"alloc_extend (translate_kv_loc would return negative)",
            )
        # And p2v_page must round-trip.
        for v_page, p_page in zip(consumed_pages.tolist(), v2p_values.tolist()):
            self.assertEqual(int(full_alloc.physical_to_virtual[p_page].item()), v_page)
        # `allocated_count()` is in TOKENS (advances by 2 * PAGE_SIZE);
        # `_allocated_pages()` is the page count.
        self.assertEqual(
            full_alloc.allocated_count(),
            allocated_count_before + 2 * PS,
        )
        self.assertEqual(
            full_alloc._allocated_pages(),
            (allocated_count_before // PS) + 2,
        )
        if full_alloc.grow_direction == "up":
            self.assertEqual(full_alloc.watermark_physical, watermark_before + 2)
        else:
            self.assertEqual(full_alloc.watermark_physical, watermark_before - 2)
        # Free-list must have shrunk by 2.
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()) - 2,
        )

    # 10. Regression: alloc_decode must bind v2p / p2v when it wraps to a new
    # page; most decode steps reuse the prefix's tail page and bind nothing.
    def test_paged_alloc_decode_binds_v2p_p2v_on_page_wrap(self):
        from sglang.srt.mem_cache.allocator import unified_sub_pool as mea_mod

        _, full_alloc, _, _, _ = self._build()
        PS = self.PAGE_SIZE
        # Pre-allocate ~1 page so an arbitrary `seq_len % page_size == 1`
        # decode step triggers a new-page consumption.
        v = full_alloc.alloc(PS)
        self.assertIsNotNone(v)
        free_before = full_alloc.free_virtual_ids.clone()
        watermark_before = full_alloc.watermark_physical
        allocated_count_before = full_alloc.allocated_count()

        # A decode wraps to a new page exactly when seq_len % page_size == 1;
        # the kernel then consumes `free_virtual_ids[0]`.
        seq_lens = torch.tensor([PS + 1], dtype=torch.int64, device=_DEV)
        seq_lens_cpu = torch.tensor([PS + 1], dtype=torch.int64)
        last_loc = torch.tensor(
            # last token of page-N at offset page_size-1.
            [int(v[-1].item())],
            dtype=torch.int64,
            device=_DEV,
        )

        original_kernel = mea_mod.alloc_decode_kernel

        class _NoOpKernelGrid:
            def __getitem__(self, _grid):
                return self

            def __call__(self, *a, **kw):
                pass

        mea_mod.alloc_decode_kernel = _NoOpKernelGrid()
        try:
            out = full_alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        finally:
            mea_mod.alloc_decode_kernel = original_kernel

        self.assertIsNotNone(out)
        # 1 virtual page consumed from the head of free_virtual_ids.
        consumed_page = int(free_before[0].item())
        # v2p_page must now map to a valid physical page (not -1).
        p_page = int(full_alloc.virtual_to_physical[consumed_page].item())
        self.assertNotEqual(
            p_page,
            -1,
            f"REGRESSION: virtual page {consumed_page} not bound after "
            f"alloc_decode (translate_kv_loc would return negative)",
        )
        # p2v round-trip.
        self.assertEqual(
            int(full_alloc.physical_to_virtual[p_page].item()), consumed_page
        )
        # `allocated_count()` is in TOKENS (advances by PAGE_SIZE);
        # `_allocated_pages()` is the page count.
        self.assertEqual(
            full_alloc.allocated_count(),
            allocated_count_before + PS,
        )
        self.assertEqual(
            full_alloc._allocated_pages(),
            (allocated_count_before // PS) + 1,
        )
        if full_alloc.grow_direction == "up":
            self.assertEqual(full_alloc.watermark_physical, watermark_before + 1)
        else:
            self.assertEqual(full_alloc.watermark_physical, watermark_before - 1)
        # Free-list must have shrunk by 1.
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()) - 1,
        )

    # 11. Regression: alloc_decode with num_new_pages == 0 must not advance
    # the watermark or touch v2p / p2v.
    def test_paged_alloc_decode_no_op_when_no_new_page(self):
        from sglang.srt.mem_cache.allocator import unified_sub_pool as mea_mod

        _, full_alloc, _, _, _ = self._build()
        PS = self.PAGE_SIZE
        # Alloc one page, then decode within it: no new page is consumed.
        v = full_alloc.alloc(PS)
        free_before = full_alloc.free_virtual_ids.clone()
        watermark_before = full_alloc.watermark_physical
        allocated_count_before = full_alloc.allocated_count()

        # seq_len = PS - 1 (just inside the prefix page), pre-prefix-len = PS - 2.
        # `(seq_lens % page_size == 1)` is FALSE here, so num_new_pages == 0.
        seq_lens = torch.tensor([PS - 1], dtype=torch.int64, device=_DEV)
        seq_lens_cpu = torch.tensor([PS - 1], dtype=torch.int64)
        last_loc = torch.tensor(
            [int(v[PS - 2].item())],
            dtype=torch.int64,
            device=_DEV,
        )

        original_kernel = mea_mod.alloc_decode_kernel

        class _NoOpKernelGrid:
            def __getitem__(self, _grid):
                return self

            def __call__(self, *a, **kw):
                pass

        mea_mod.alloc_decode_kernel = _NoOpKernelGrid()
        try:
            out = full_alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        finally:
            mea_mod.alloc_decode_kernel = original_kernel

        self.assertIsNotNone(out)
        # Nothing should have moved -- no new page consumed.
        self.assertEqual(full_alloc.watermark_physical, watermark_before)
        self.assertEqual(full_alloc.allocated_count(), allocated_count_before)
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()),
        )

    # Regression: at page_size > 1 the `out=` page-math branch must match the
    # no-`out=` form byte-for-byte.
    def test_paged_translate_kv_loc_with_out(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v = full_alloc.alloc(2 * ps)
        self.assertIsNotNone(v)
        # Compare with-out vs no-out.
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        ptr_before = buf.data_ptr()
        with_out = full_alloc.translate_kv_loc(v, out=buf)
        no_out = full_alloc.translate_kv_loc(v)
        self.assertIs(with_out, buf, "must return the out= buffer")
        self.assertEqual(
            buf.data_ptr(), ptr_before, "out= buffer's data_ptr must be stable"
        )
        # Page-math correctness: result equals virt_page * ps + offset
        # against the real v2p table.
        virt_pages = v // ps
        offsets = v % ps
        phys_pages = full_alloc.virtual_to_physical[virt_pages]
        expected = phys_pages * ps + offsets
        self.assertTrue(bool((buf == expected).all().item()))
        self.assertTrue(bool((with_out == no_out).all().item()))

    # Regression: at page_size > 1 the aliasing form `translate_kv_loc(buf,
    # out=buf)` must derive `virt_pages` / `offsets` before writing into `out`.
    def test_paged_translate_kv_loc_with_out_aliasing_input(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v_orig = full_alloc.alloc(2 * ps)
        self.assertIsNotNone(v_orig)
        # Expected (no-out form) computed BEFORE mutating buf.
        expected = full_alloc.translate_kv_loc(v_orig)
        # In-place form: buf serves as both input and out.
        buf = v_orig.clone()
        ptr_before = buf.data_ptr()
        ret = full_alloc.translate_kv_loc(buf, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(buf.data_ptr(), ptr_before)
        self.assertTrue(
            bool((buf == expected).all().item()),
            "page>1 in-place result must equal no-out result",
        )

    # At ps > 1 a tombstoned page yields `-1 * ps + offset`, i.e. a value in
    # [-ps, -1]; the clamp must lift every output token to >= 0.
    def test_paged_translate_kv_loc_clamps_tombstoned_v2p(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v = full_alloc.alloc(2 * ps)
        self.assertIsNotNone(v)
        # Tombstone one page (any v2p_page entry) -> all ps tokens in that
        # page should clamp to 0 in the translate output.
        tomb_page = int((v[0] // ps).item())
        full_alloc.virtual_to_physical[tomb_page] = -1
        # No-out form.
        out = full_alloc.translate_kv_loc(v)
        self.assertTrue(
            bool((out >= 0).all().item()),
            f"paged translate_kv_loc must clamp tombstoned to >=0; got {out.tolist()}",
        )
        # The first `ps` tokens belong to the tombstoned page -> all 0.
        self.assertTrue(
            bool((out[:ps] == 0).all().item()),
            "all tokens in a tombstoned page must map to slot 0 (padding sink)",
        )
        # The second page is still bound; its outputs must be > 0.
        self.assertTrue(
            bool((out[ps:] > 0).all().item()),
            "non-tombstoned pages must still translate to live physical slots",
        )
        # out= form must also clamp.
        buf = torch.empty_like(v)
        ret = full_alloc.translate_kv_loc(v, out=buf)
        self.assertIs(ret, buf)
        self.assertTrue(
            bool((buf >= 0).all().item()),
            "paged out= path must clamp tombstoned entries",
        )
        self.assertTrue(bool((buf[:ps] == 0).all().item()))

    # 14. Regression: the leak-invariant terms the scheduler checks are all in
    # TOKENS -- `full_available_size() + allocated_tokens == static_cap`.
    def test_paged_swa_full_available_size_in_tokens(self):
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        PS = self.PAGE_SIZE
        n_full_pages, n_swa_pages = 16, 16
        total = (
            n_full_pages * PS * full_spec.entry_bytes()
            + n_swa_pages * PS * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        full_max = n_full_pages * PS
        swa_max = n_swa_pages * PS
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=full_max,
            swa_max_total_num_tokens=swa_max,
            page_size=PS,
            need_sort=False,
            forward_stream=None,
        )
        # The leak invariant reads `_conserve_*`; public `*_available_size()`
        # is `min(conserve, schedulable)` and may be smaller.
        self.assertEqual(allocator._conserve_full_available_size(), full_max)
        self.assertEqual(allocator._conserve_swa_available_size(), swa_max)

        # Alloc 2 pages = 2*PS tokens.
        v = allocator.alloc(2 * PS)
        self.assertIsNotNone(v)

        # conserve view must drop by 2*PS TOKENS, not by 2 (pages).
        self.assertEqual(
            allocator._conserve_full_available_size(),
            full_max - 2 * PS,
            "REGRESSION: the conserve view must drop by token-count, "
            "not page-count. A 'pool memory leak detected' crash is "
            "caused by a page-count drop here.",
        )
        self.assertEqual(
            allocator._conserve_swa_available_size(),
            swa_max - 2 * PS,
        )

        # No eviction yet, so total == conserve + allocated_tokens.
        allocated_tokens = full_max - allocator._conserve_full_available_size()
        self.assertEqual(allocated_tokens, 2 * PS)
        self.assertEqual(
            allocated_tokens + allocator._conserve_full_available_size(),
            full_max,
        )

    # 15. Regression: `UnifiedMambaTokenToKVPoolAllocator.size` must be TOTAL
    # TOKENS -- available + allocated, both in tokens, never a page count.
    def test_paged_mamba_size_in_tokens(self):
        from sglang.srt.mem_cache.allocator.unified_mamba import (
            UnifiedMambaTokenToKVPoolAllocator,
        )

        # The mamba sub-allocator always uses page_size=1; only the full
        # sub-allocator takes self.PAGE_SIZE.
        PS = self.PAGE_SIZE
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        mamba_spec = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            conv_state_shapes=((4, 3),),
            conv_dtype=torch.float32,
            temporal_state_shape=(2, 2, 2),
            temporal_dtype=torch.float32,
            grow_direction="down",
        )
        n_full_pages, n_mamba_slots = 16, 8
        total = (
            n_full_pages * PS * full_spec.entry_bytes()
            + n_mamba_slots * mamba_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, mamba_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        full_kv.attach_allocator = lambda allocator: None
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        mamba_kv.attach_allocator = lambda allocator: None
        # The physical-move contract for the mamba sub-pool (un-translated);
        # _FakeKVCache already provides move_kv_cache, so nothing extra.

        class _FakeHybridLinearKVPool:
            full_kv_pool = full_kv
            mamba_pool = mamba_kv

        allocator = UnifiedMambaTokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=_FakeHybridLinearKVPool(),
            device=_DEV,
            page_size=PS,
            need_sort=False,
            forward_stream=None,
        )

        # Idle: size == full_available_size() (entirely in tokens).
        full_avail_before = allocator.full_attn_allocator.available_size()
        self.assertEqual(allocator.size, full_avail_before)
        # available_size == size (no allocations yet).
        self.assertEqual(allocator.available_size(), allocator.size)

        # Alloc 2 pages = 2*PS tokens on full side.
        v = allocator.alloc(2 * PS)
        self.assertIsNotNone(v)

        # `.size` is dynamic -- it shrinks as the peer consumes bytes -- but the
        # peer is idle here, so the conserved sum still equals the initial total.
        self.assertEqual(
            allocator.full_attn_allocator.available_size()
            + allocator.full_attn_allocator.allocated_count(),
            full_avail_before,
            "REGRESSION: full.available_size() + full.allocated_count() must "
            "be conserved at TOKEN granularity (was `tokens + pages` in the "
            "buggy revision).",
        )
        # And .size matches this conserved sum.
        self.assertEqual(allocator.size, full_avail_before)

    # 16. Regression: `_virt_tokens_to_phys_tokens` must do the page math --
    # indexing the page-granular v2p with token ids gives OOB kernel inputs.
    # A real `UnifiedSWAKVPool` cannot be built in this CPU shim, so the
    # production instance methods are covered through the static helper.
    def test_paged_pool_translate_helper_returns_physical_tokens(self):
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedSWATokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        PS = self.PAGE_SIZE
        n_pages = 8
        total = (
            n_pages * PS * full_spec.entry_bytes()
            + n_pages * PS * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_pages * PS,
            swa_max_total_num_tokens=n_pages * PS,
            page_size=PS,
            need_sort=False,
            forward_stream=None,
        )

        v_tokens = allocator.alloc(2 * PS)
        self.assertIsNotNone(v_tokens)

        # The static helper does the page math: same as the instance methods.
        swa_phys = UnifiedSWAKVPool._virt_tokens_to_phys_tokens(
            v_tokens, allocator.swa_attn_allocator
        )

        self.assertTrue(
            bool((swa_phys >= 0).all().item()),
            "REGRESSION: _virt_tokens_to_phys_tokens returned negative "
            "physical token ids (page-math fix likely reverted).",
        )
        self.assertEqual(
            int(torch.unique(swa_phys).numel()),
            int(swa_phys.numel()),
            "Physical token ids must be unique (one-to-one mapping).",
        )
        virt_pages_in = v_tokens // PS
        offsets_in = v_tokens % PS
        swa_phys_pages_direct = allocator.swa_attn_allocator.virtual_to_physical[
            virt_pages_in
        ]
        expected = swa_phys_pages_direct * PS + offsets_in
        self.assertTrue(
            bool((swa_phys == expected).all().item()),
            "REGRESSION: _virt_tokens_to_phys_tokens output must equal "
            "v2p_page[virt_pages] * page_size + offsets.",
        )

        # The composite emits KERNEL-FACING ids, not the physical token ids this
        # helper returns; they coincide only at multiplier 1, which nothing uses.
        swa_mult = allocator.swa_kernel_page_multiplier
        self.assertEqual(swa_mult, 2 * swa_spec.layer_num)
        composite_out = allocator.translate_loc_from_full_to_swa(v_tokens)
        expected_kernel = swa_phys_pages_direct * (PS * swa_mult) + offsets_in
        self.assertTrue(
            bool((composite_out.long() == expected_kernel.long()).all().item()),
            "REGRESSION: translate_loc_from_full_to_swa must emit the swa "
            "sub-pool's kernel-facing ids (phys_page * ps * blocks_per_page + offset).",
        )


class TestLazyCompaction(unittest.TestCase):
    """Lazy compaction invariants and the lazy-vs-eager equivalence harness.
    `_flush` only waits on `forward_stream` when one is passed, so these
    allocators stay CPU-only."""

    def _make_full(self, *, lazy: bool, n_full_slots=64, n_mamba_slots=16):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        return pool, full_alloc, full_kv

    def _stamp_kv(self, kv: _FakeKVCache, alloc: MultiEndedAllocator, tokens) -> None:
        """Write a marker into KV[phys] for each freshly-alloced virtual
        token id, so we can later check the data followed any relocation.
        """
        for v in tokens.tolist():
            p = int(alloc.virtual_to_physical[v].item())
            kv.buf[p] = int(v)

    def test_lazy_free_inward_walk(self):
        """The inward walk is deferred to `_flush`, which then absorbs every
        contiguous-from-boundary hole in one pass, whatever the free order."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)
        wm_before = fa.watermark_physical
        # Free a middle slot first -> hole.
        fa.free(a[2:3].clone())
        # Free the topmost ids -- in eager mode these would absorb inline,
        # but in lazy mode they cat onto the free list.
        fa.free(a[4:5].clone())
        fa.free(a[3:4].clone())
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 3)
        fa._flush(urgent=True)
        self.assertEqual(fa.watermark_physical, wm_before - 3)
        self.assertEqual(len(fa._free_phys_pages), 0)
        self.assertEqual(fa.live_page_count, 2)

    def test_lazy_take_physical_drains_holes_first(self):
        """alloc reuses holes before extending the watermark."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)
        # Free two non-boundary virtuals to populate _free_phys_pages.
        fa.free(a[1:2].clone())
        fa.free(a[3:4].clone())
        n_holes_before = len(fa._free_phys_pages)
        self.assertEqual(n_holes_before, 2)
        wm_before = fa.watermark_physical
        # Allocate 2 more -- both should come from holes, watermark unchanged.
        a2 = fa.alloc(2)
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 0)
        # Allocate 1 more -- must extend (no holes left).
        a3 = fa.alloc(1)
        self.assertEqual(fa.watermark_physical, wm_before + 1)
        # All three new alloc batches are non-None.
        self.assertIsNotNone(a2)
        self.assertIsNotNone(a3)

    def test_lazy_available_size_includes_holes(self):
        """available_size counts drainable holes + extension capacity."""
        _pool, fa, _kv = self._make_full(lazy=True)
        avail_initial = fa.available_size()
        a = fa.alloc(5)
        # 3 non-boundary frees -> 3 holes; watermark unchanged.
        fa.free(a[0:1].clone())
        fa.free(a[1:2].clone())
        fa.free(a[2:3].clone())
        self.assertEqual(len(fa._free_phys_pages), 3)
        avail_after = fa.available_size()
        # Holes stay available capacity, so only the 2 tokens that stayed live
        # are subtracted from the initial availability.
        self.assertEqual(avail_after, avail_initial - 2)

    def test_lazy_flush_compacts_holes_into_gap(self):
        """_flush(urgent=True) moves a survivor into a hole and shrinks the
        watermark, freeing bytes back into the shared gap.
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)
        # Stamp KV so we can assert the data followed the relocation.
        self._stamp_kv(_kv, fa, a)
        # Free a low-index hole; keep the topmost live.
        fa.free(a[1:2].clone())
        self.assertEqual(len(fa._free_phys_pages), 1)
        wm_before = fa.watermark_physical
        n_moves = fa._flush(urgent=True)
        # At least one move should have happened (topmost survivor -> hole).
        self.assertGreaterEqual(n_moves, 1)
        # Watermark shrunk; hole list is now empty.
        self.assertLess(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 0)
        # live_page_count invariant under compaction.
        self.assertEqual(fa.live_page_count, 4)

    def test_lazy_flush_gathers_survivor_mappings_as_one_batch(self):
        """Compaction must not synchronize once per relocated survivor."""
        _pool, fa, kv = self._make_full(lazy=True)
        values = fa.alloc(12)
        self._stamp_kv(kv, fa, values)
        fa.free(values[1:5].clone())

        physical_to_virtual = fa.physical_to_virtual
        fa.physical_to_virtual = _RejectScalarIndexTensor(physical_to_virtual)
        try:
            self.assertEqual(fa._flush(urgent=True), 4)
        finally:
            fa.physical_to_virtual = physical_to_virtual

        for virtual in values.tolist():
            physical = int(fa.virtual_to_physical[virtual].item())
            if physical == -1:
                continue
            self.assertEqual(int(fa.physical_to_virtual[physical].item()), virtual)
            self.assertEqual(int(kv.buf[physical].item()), virtual)

    def _replay_sequence(self, ops, lazy: bool):
        """Run a given alloc/free op trace under eager OR lazy mode and
        return the final (live virtual set, alloc-time KV stamps)."""
        _pool, fa, kv = self._make_full(lazy=lazy)
        live = set()  # set of virtual ids
        kv_stamps = {}  # v -> stamp (the data we wrote at alloc time)
        next_stamp = 100
        for kind, n in ops:
            if kind == "alloc":
                tokens = fa.alloc(n)
                if tokens is None:
                    continue
                for v in tokens.tolist():
                    p = int(fa.virtual_to_physical[v].item())
                    kv.buf[p] = next_stamp
                    kv_stamps[v] = next_stamp
                    live.add(v)
                    next_stamp += 1
            elif kind == "free":
                if not live:
                    continue
                # Take up to n from live, deterministically by id.
                victims = sorted(live)[:n]
                live.difference_update(victims)
                fa.free(torch.tensor(victims, dtype=torch.int64))
        # Force final compaction on lazy so the comparison is at quiescence.
        if lazy:
            fa._flush(urgent=True)
        # Read back the data for each live id.
        live_data = {}
        for v in live:
            p = int(fa.virtual_to_physical[v].item())
            live_data[v] = int(kv.buf[p].item())
        return live, live_data, kv_stamps

    def test_lazy_vs_eager_equivalence(self):
        """Same random alloc/free sequence under lazy and eager modes must
        yield identical live virtual sets AND identical KV reads (the data
        followed any relocation).
        """
        rng = random.Random(42)
        ops = []
        for _ in range(200):
            if rng.random() < 0.6:
                ops.append(("alloc", rng.randint(1, 6)))
            else:
                ops.append(("free", rng.randint(1, 4)))
        eager_live, eager_data, eager_stamps = self._replay_sequence(ops, lazy=False)
        lazy_live, lazy_data, lazy_stamps = self._replay_sequence(ops, lazy=True)
        self.assertEqual(eager_live, lazy_live, "live virtual set diverged")
        self.assertEqual(eager_stamps, lazy_stamps, "alloc-time stamps diverged")
        # For every live id, the data we read back must match what we wrote.
        for v in eager_live:
            self.assertEqual(
                eager_data[v], eager_stamps[v], f"eager: KV[v={v}] != stamp"
            )
            self.assertEqual(lazy_data[v], lazy_stamps[v], f"lazy: KV[v={v}] != stamp")

    def test_lazy_non_urgent_stops_at_write_set_blocker(self):
        """When the topmost survivor is in an in-flight batch's write-set, a
        non-urgent `_flush` stops the boundary walk: skipping past it would
        shuffle holes without shrinking the watermark."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)

        class _FakeEvent:
            def __init__(self):
                self.fired = False

            def query(self):
                return self.fired

        ev = _FakeEvent()
        fa.set_latest_forward_done_event(ev)
        # Free a non-boundary slot to create a compactable hole.
        fa.free(a[1:2].clone())
        self.assertEqual(len(fa._free_phys_pages), 1)
        # `set_inflight_forward` takes the virtual `out_cache_loc` tensor, not a
        # materialized physical set; `_flush` translates it when it classifies.
        topmost_phys = int(fa.virtual_to_physical[int(a[-1].item())].item())
        oclv = a[-1:].clone()  # virtual id that translates to topmost_phys
        fa.set_inflight_forward(ev, oclv)
        # Non-urgent flush -> case A blocker at the top -> STOP.
        n_moves = fa._flush(urgent=False)
        self.assertEqual(
            n_moves,
            0,
            "non-urgent flush must STOP when the topmost survivor is in "
            "an in-flight write-set",
        )
        # State untouched: hole still present.
        self.assertEqual(len(fa._free_phys_pages), 1)
        self.assertEqual(len(fa._pending_reuse), 0)
        # Firing the event prunes the write-set entry; the next flush then
        # releases src directly, with no pending-reuse entry needed.
        ev.fired = True
        n_moves2 = fa._flush(urgent=False)
        self.assertGreaterEqual(n_moves2, 1)
        self.assertEqual(len(fa._pending_reuse), 0)

    def test_lazy_non_urgent_read_race_uses_pending_reuse(self):
        """With no in-flight write-set, a non-urgent `_flush` compacts at once
        (read+read on KV[src] is safe) and parks `(src, latest_event)` in
        `_pending_reuse`, so no later alloc writes KV[src] under a live read."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)

        class _FakeEvent:
            def __init__(self):
                self.fired = False

            def query(self):
                return self.fired

        ev = _FakeEvent()
        fa.set_latest_forward_done_event(ev)
        # Free a non-boundary slot -> 1 hole.
        fa.free(a[1:2].clone())
        # `out_cache_loc_virtual=None` means "no write race on this pool" -- the
        # path Mamba uses, since its forward writes state through its own
        # kernels rather than `out_cache_loc`.
        fa.set_inflight_forward(ev, None)
        n_moves = fa._flush(urgent=False)
        self.assertGreaterEqual(n_moves, 1)
        # `_pending_reuse` holds one entry per BATCH (keyed by event), not per
        # src; the page count lives in `_pending_reuse_pages_cpu`.
        self.assertEqual(len(fa._pending_reuse), 1)
        self.assertEqual(len(fa._pending_reuse_pages_cpu), n_moves)
        # Fire the event and drain -- srcs return to availability.
        ev.fired = True
        fa._drain_pending_reuse(urgent=False)
        self.assertEqual(len(fa._pending_reuse), 0)
        self.assertEqual(len(fa._pending_reuse_pages_cpu), 0)


class TestO3FusedAllocBind(unittest.TestCase):
    """Fused take_physical_pages + bind_pages via `_alloc_bind_fast_or_slow`.
    GPU-only: the fused kernel is Triton."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("O3 fused alloc-bind kernel requires CUDA")

    def _make_full(
        self,
        *,
        lazy: bool = True,
        n_full_slots: int = 64,
        n_mamba_slots: int = 16,
        page_size: int = 1,
    ):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device="cuda",
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        fa = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device="cuda",
            is_id_owner=True,
            page_size=page_size,
            lazy_compaction=lazy,
        )
        ma = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device="cuda",
            is_id_owner=True,
            page_size=1,  # mamba is per-request, always page=1
            lazy_compaction=lazy,
        )
        fa.bind_peer(ma)
        ma.bind_peer(fa)
        return pool, fa, full_kv

    def test_fast_path_when_no_holes(self):
        """When `_free_phys_pages` is empty, the fused fast path fires."""
        _pool, fa, _kv = self._make_full(lazy=True)
        # Sanity: empty holeset.
        self.assertEqual(len(fa._free_phys_pages), 0)
        wm_before = fa.watermark_physical
        # Pick virtual page ids the kernel will bind.
        v_pages = torch.tensor([20, 21, 22, 23], dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 4)
        # Watermark advanced by N.
        self.assertEqual(fa.watermark_physical, wm_before + 4)
        # Returned phys ids match the grow-up arange [wm_before, wm_before+4).
        expected_phys = torch.arange(
            wm_before, wm_before + 4, dtype=torch.int64, device="cuda"
        )
        self.assertTrue(torch.equal(phys, expected_phys))
        # v2p table: each virtual -> its physical.
        for v, p in zip(v_pages.tolist(), expected_phys.tolist()):
            self.assertEqual(int(fa.virtual_to_physical[v].item()), p)
        # p2v table: each physical -> its virtual.
        for v, p in zip(v_pages.tolist(), expected_phys.tolist()):
            self.assertEqual(int(fa.physical_to_virtual[p].item()), v)
        # live_page_count updated.
        self.assertEqual(fa.live_page_count, 4)
        # Another fast-path call accumulates.
        v_pages2 = torch.tensor([24, 25], dtype=torch.int64, device="cuda")
        fa._alloc_bind_fast_or_slow(v_pages2, 2)
        self.assertEqual(fa.live_page_count, 6)

    def test_slow_path_when_holes_exist(self):
        """Greedy hole reuse: an existing hole must be drained before the
        watermark extends, so the fast path must not fire."""
        _pool, fa, _kv = self._make_full(lazy=True)
        # Build a hole by alloc-then-free-non-boundary.
        a = fa.alloc(3)
        self.assertEqual(fa.live_page_count, 3)
        fa.free(a[0:1].clone())  # frees a non-boundary slot -> enters holeset
        self.assertEqual(fa.live_page_count, 2)
        self.assertEqual(len(fa._free_phys_pages), 1)
        # `_free_phys_pages` is a torch.Tensor; read the hole via `.tolist()`.
        hole_pos = int(fa._free_phys_pages.tolist()[0])
        wm_before = fa.watermark_physical
        # Alloc 1 page via the helper. Slow path should drain the hole.
        v_pages = torch.tensor([42], dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 1)
        # Hole drained, NOT a watermark extension.
        self.assertEqual(int(phys[0].item()), hole_pos)
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 0)
        # v2p/p2v updated.
        self.assertEqual(int(fa.virtual_to_physical[42].item()), hole_pos)
        self.assertEqual(int(fa.physical_to_virtual[hole_pos].item()), 42)
        # The slow path advances live_page_count via take_physical_pages.
        self.assertEqual(fa.live_page_count, 3)

    def test_fast_path_in_eager_mode(self):
        """Eager mode never accumulates holes, so it always takes the fast path;
        `live_page_count` is not maintained there and stays 0."""
        _pool, fa, _kv = self._make_full(lazy=False)
        self.assertFalse(fa.lazy_compaction)
        self.assertEqual(fa.live_page_count, 0)
        wm_before = fa.watermark_physical
        v_pages = torch.tensor([30, 31, 32], dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 3)
        self.assertEqual(fa.watermark_physical, wm_before + 3)
        expected_phys = torch.arange(
            wm_before, wm_before + 3, dtype=torch.int64, device="cuda"
        )
        self.assertTrue(torch.equal(phys, expected_phys))
        # live_page_count UNCHANGED (eager mode invariant).
        self.assertEqual(fa.live_page_count, 0)

    def test_index_space_overflow_returns_none(self):
        """When the requested allocation would overflow `num_pages`,
        the helper returns None and leaves the allocator unchanged."""
        _pool, fa, _kv = self._make_full(lazy=True, n_full_slots=8, n_mamba_slots=2)
        # Try to alloc more pages than exist.
        N = fa.num_pages + 100
        wm_before = fa.watermark_physical
        # Only v_pages' numel matters to the helper.
        v_pages = torch.zeros(N, dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, N)
        self.assertIsNone(phys)
        # Allocator state unchanged.
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(fa.live_page_count, 0)

    def test_empty_alloc_returns_empty_tensor(self):
        """N=0 returns an empty tensor (no kernel launch, no state change)."""
        _pool, fa, _kv = self._make_full(lazy=True)
        wm_before = fa.watermark_physical
        v_pages = torch.empty(0, dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 0)
        self.assertIsNotNone(phys)
        self.assertEqual(phys.numel(), 0)
        self.assertEqual(fa.watermark_physical, wm_before)

    def test_fast_path_equivalent_to_slow_path(self):
        """On an empty holeset, the fast path must produce identical v2p / p2v /
        return tensors to the unfused take_physical_pages + bind sequence."""
        # Two identical allocators; one takes fast path, one takes slow.
        _pool_a, fa_a, _kv_a = self._make_full(lazy=True)
        _pool_b, fa_b, _kv_b = self._make_full(lazy=True)
        v_pages = torch.tensor([50, 51, 52, 53, 54], dtype=torch.int64, device="cuda")
        # Fast path on fa_a.
        phys_a = fa_a._alloc_bind_fast_or_slow(v_pages, 5)
        # Slow path on fa_b: directly call take_physical_pages + bind
        # (the unfused reference implementation).
        phys_b = fa_b.take_physical_pages(5)
        fa_b.bind(v_pages, phys_b)
        # take_physical_pages already advances live_page_count (matching the
        # fused fast path), so no manual bump here.
        # Identical return tensors.
        self.assertTrue(torch.equal(phys_a, phys_b))
        # Identical v2p / p2v after the operation.
        self.assertTrue(torch.equal(fa_a.virtual_to_physical, fa_b.virtual_to_physical))
        self.assertTrue(torch.equal(fa_a.physical_to_virtual, fa_b.physical_to_virtual))
        # Identical watermark + live_page_count.
        self.assertEqual(fa_a.watermark_physical, fa_b.watermark_physical)
        self.assertEqual(fa_a.live_page_count, fa_b.live_page_count)

    def test_page_size_gt_1(self):
        """At page_size > 1 the kernel must scatter one v2p entry per PAGE,
        not per token."""
        _pool, fa, _kv = self._make_full(
            lazy=True, n_full_slots=64, n_mamba_slots=16, page_size=4
        )
        self.assertEqual(fa.page_size, 4)
        v_pages = torch.tensor([3, 4, 5], dtype=torch.int64, device="cuda")
        wm_before = fa.watermark_physical
        phys = fa._alloc_bind_fast_or_slow(v_pages, 3)
        self.assertIsNotNone(phys)
        self.assertEqual(phys.shape, (3,))
        # Watermark advances by N PAGES (not N tokens).
        self.assertEqual(fa.watermark_physical, wm_before + 3)
        # v2p table updated at page granularity.
        for v, p in zip(v_pages.tolist(), phys.tolist()):
            self.assertEqual(int(fa.virtual_to_physical[v].item()), p)
            self.assertEqual(int(fa.physical_to_virtual[p].item()), v)

    def test_grow_down_fast_path(self):
        """The mamba sub-pool is grow-down. Verifies fast-path arithmetic
        in the descending direction."""
        _pool, _fa, _kv = self._make_full(lazy=True)
        # Build a grow-down allocator standalone for the test.
        from sglang.srt.mem_cache.unified_memory_pool import (
            UnifiedKVPool,
        )

        full = _make_mha_spec("full", "up", layer_num=2)
        swa = _make_mha_spec("swa", "down", layer_num=2)  # grow-down
        total = (full.entry_bytes() + swa.entry_bytes()) * 32
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device="cuda",
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        swa_kv = _FakeKVCache(pool.max_slots("swa"))
        fa = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device="cuda",
            is_id_owner=True,
            lazy_compaction=True,
        )
        sa = MultiEndedAllocator(
            kvcache=swa_kv,
            unified_buffer=pool,
            sub_pool_name="swa",
            device="cuda",
            is_id_owner=False,  # non-owner, grow-down
            lazy_compaction=True,
        )
        fa.bind_peer(sa)
        sa.bind_peer(fa)
        # Grow-down: watermark starts at num_pages - 1, decreases.
        self.assertEqual(sa.grow_direction, "down")
        wm_before = sa.watermark_physical
        v_pages = torch.tensor([5, 6, 7], dtype=torch.int64, device="cuda")
        phys = sa._alloc_bind_fast_or_slow(v_pages, 3)
        # Grow-down: the kernel emits ASCENDING, matching
        # `_take_physical_eager`'s `torch.arange(wm - N + 1, wm + 1)`.
        expected = torch.tensor(
            [wm_before - 2, wm_before - 1, wm_before],
            dtype=torch.int64,
            device="cuda",
        )
        self.assertTrue(torch.equal(phys, expected))
        # Watermark decreased by N.
        self.assertEqual(sa.watermark_physical, wm_before - 3)
        for v, p in zip(v_pages.tolist(), expected.tolist()):
            self.assertEqual(int(sa.virtual_to_physical[v].item()), p)
            self.assertEqual(int(sa.physical_to_virtual[p].item()), v)


class TestSWACompositeKernelIdSurface(unittest.TestCase):
    """The SWA composite's kernel-facing id surface. Attention backends probe for
    `translate_kv_loc_for_kernel` / `full_v2p_page_table`, and every id must
    follow `kernel_id(t) = v2p[t // ps] * (ps * mult) + t % ps`."""

    PS = 4
    FULL_L = 4
    SWA_L = 2

    def _build(self):
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=self.FULL_L,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=self.SWA_L,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        n_full, n_swa = 64, 32  # tokens = 16 / 8 pages at PS=4
        total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        return UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            page_size=self.PS,
            need_sort=False,
            forward_stream=None,
        )

    def test_full_kernel_translate_matches_formula(self):
        """Both sides scale by their OWN sub-pool's block count, and the full
        kernel id follows v2p[t // ps] * (ps * mult) + t % ps."""
        mult = 2 * self.FULL_L
        a = self._build()
        self.assertEqual(a.kernel_page_multiplier, 2 * self.FULL_L)
        self.assertEqual(a.swa_kernel_page_multiplier, 2 * self.SWA_L)
        v = a.alloc(3 * self.PS)
        self.assertIsNotNone(v)
        v2p = a.full_attn_allocator.virtual_to_physical
        expected = v2p[v // self.PS] * (self.PS * mult) + v % self.PS
        self.assertTrue(torch.equal(a.translate_kv_loc_for_kernel(v), expected))
        # The PHYSICAL translate must stay unscaled -- compaction and the byte
        # machinery depend on it staying in physical space.
        phys = v2p[v // self.PS] * self.PS + v % self.PS
        self.assertTrue(torch.equal(a.translate_kv_loc(v), phys))

    def test_kernel_translate_accepts_an_int32_page_table(self):
        """Regression: fa3 passes its own page table, which is int32 and 2-D, so
        the gather must not require an int64 index."""
        for ps in (1, 4):
            with self.subTest(page_size=ps):
                self.PS = ps
                mult = 2 * self.FULL_L
                a = self._build()
                v = a.alloc(4 * ps)
                self.assertIsNotNone(v)
                v2p = a.full_attn_allocator.virtual_to_physical
                expected = v2p[v // ps] * (ps * mult) + v % ps
                page_table = v.to(torch.int32).view(2, -1)
                got = a.translate_kv_loc_for_kernel(page_table)
                self.assertEqual(got.shape, page_table.shape)
                self.assertTrue(torch.equal(got.reshape(-1), expected))
                # `out=` takes the same int32 index; the buffer stays int64.
                dst = torch.empty(page_table.shape, dtype=torch.int64, device=_DEV)
                a.translate_kv_loc_for_kernel(page_table, out=dst)
                self.assertTrue(torch.equal(dst.reshape(-1), expected))

    def test_swa_translate_scales_page_stride(self):
        mult = 2 * self.SWA_L
        a = self._build()
        v = a.alloc(3 * self.PS)
        self.assertIsNotNone(v)
        v2p_swa = a.swa_attn_allocator.virtual_to_physical
        expected = v2p_swa[v // self.PS] * (self.PS * mult) + v % self.PS
        self.assertTrue(torch.equal(a.translate_loc_from_full_to_swa(v), expected))

    def test_swa_kernel_tombstone_still_lands_on_sink(self):
        """The scaled stride must not break the tombstone clamp: a tombstoned
        page's ids (v2p == -1 -> -stride + offset, negative for every in-page
        offset) still land on the sink, never negative."""
        mult = 2 * self.SWA_L
        a = self._build()
        v = a.alloc(2 * self.PS)
        self.assertIsNotNone(v)
        tomb_page = int(v[0].item()) // self.PS
        a.swa_attn_allocator.virtual_to_physical[tomb_page] = -1
        got = a.translate_loc_from_full_to_swa(v)
        self.assertTrue(bool((got >= 0).all().item()))
        in_tomb = v // self.PS == tomb_page
        self.assertTrue(bool((got[in_tomb] == 0).all().item()))


class TestPs64MLACompositeFeasibility(unittest.TestCase):
    """MLA + mamba composite at page_size=64, the size flashmla snaps to. Large
    pages stress the sink-page floor, the per-layer-view tail pad and the
    page-granular alloc at once."""

    PS = 64
    LAYERS = 3

    def _build(self):
        full = MLASubPoolSpec(
            name="full",
            layer_num=self.LAYERS,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        mamba = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            conv_state_shapes=((8, 16),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(4, 8, 8),
            temporal_dtype=torch.float32,
            grow_direction="up",
        )
        n_full = 8 * self.PS  # 8 pages incl. the sink page
        total = n_full * full.entry_bytes() + 16 * mamba.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        full_kv.attach_allocator = lambda allocator: None
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        mamba_kv.attach_allocator = lambda allocator: None
        mamba_kv._copy_from_physical = lambda src, dst: None

        class _FakeHybridLinearKVPool:
            full_kv_pool = full_kv
            mamba_pool = mamba_kv

        return UnifiedMambaTokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=_FakeHybridLinearKVPool(),
            device=_DEV,
            page_size=self.PS,
            need_sort=False,
            forward_stream=None,
        )

    def test_construction_alloc_and_kernel_formula(self):
        a = self._build()
        # MLA: one latent row per layer, so the spec reports LAYERS blocks.
        self.assertEqual(a.kernel_page_multiplier, self.LAYERS)
        v = a.alloc(2 * self.PS)
        self.assertIsNotNone(v, "2-page alloc infeasible at ps=64")
        # Page-aligned virtual run (page-granular allocator invariant).
        self.assertEqual(int(v[0].item()) % self.PS, 0)
        # The kernel translate follows the affine formula at ps=64, and every id
        # fits int32 (the canonical narrows on store).
        v2p = a.full_v2p_page_table
        want = v2p[v // self.PS] * (self.PS * self.LAYERS) + v % self.PS
        got = a.translate_kv_loc_for_kernel(v)
        self.assertTrue(torch.equal(got, want), "kernel-facing formula broke at ps=64")
        self.assertTrue(bool((got < 2**31).all().item()))


class _ChainStub:
    """Duck-typed chain member carrying only what the frontier walk touches, so a
    test can place an opaque or transparent middle at exact byte coordinates."""

    def __init__(self, *, low_byte: int, high_byte: int, transparent: bool):
        self._low_byte = low_byte
        self._high_byte = high_byte
        self.transparent = transparent
        self.low_peer = None
        self.high_peer = None
        self.lazy_compaction = False
        self._free_phys_pages = torch.empty(0, dtype=torch.int64)
        self.entry_bytes_per_page = 1
        self.sub_pool_name = "stub"
        self.grow_direction = "float"

    def _is_frontier_transparent(self):
        return self.transparent

    def _byte_low_frontier(self):
        return self._low_byte

    def _byte_high_frontier(self):
        return self._high_byte


class TestChainFrontierWalk(unittest.TestCase):
    """N-pool chain walk: 2-pool byte-identity with the single-peer formulas,
    transparent-middle skipping, and growth-side-neighbor credit routing."""

    def _build_pair(self):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * 64 + mamba.entry_bytes() * 16
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        fa = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("full")),
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
        )
        ma = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("mamba")),
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
        )
        fa.bind_peer(ma)
        ma.bind_peer(fa)
        return pool, fa, ma

    def test_bind_peer_rejects_float_members(self):
        _, fa, ma = self._build_pair()
        stub = _ChainStub(low_byte=0, high_byte=0, transparent=True)
        with self.assertRaisesRegex(AssertionError, "END-pool-only"):
            fa.bind_peer(stub)
        fa.grow_direction = "float"
        try:
            with self.assertRaisesRegex(AssertionError, "END-pool-only"):
                fa.bind_peer(ma)
        finally:
            fa.grow_direction = "up"

    def test_two_pool_gap_equals_old_single_peer_formula(self):
        # With no middles the walk must equal the closed form
        # gap_up = peer_low - my_high; gap_down = my_low - peer_high.
        _, fa, ma = self._build_pair()
        for n_full, n_mamba in ((0, 0), (8, 0), (8, 4), (32, 16)):
            fa.clear()
            ma.clear()
            if n_full:
                self.assertIsNotNone(fa.alloc(n_full))
            if n_mamba:
                self.assertIsNotNone(ma.alloc(n_mamba))
            self.assertEqual(
                fa._current_gap_bytes(),
                max(0, ma._byte_low_frontier() - fa._byte_high_frontier()),
            )
            # Old down-side closed form: my_low - peer_high (symmetric band).
            self.assertEqual(
                ma._current_gap_bytes(),
                max(0, ma._byte_low_frontier() - fa._byte_high_frontier()),
            )

    def test_transparent_middle_is_skipped(self):
        pool, fa, ma = self._build_pair()
        mid_lo = fa.entry_bytes_per_page * 8
        mid_hi = fa.entry_bytes_per_page * 12
        stub = _ChainStub(low_byte=mid_lo, high_byte=mid_hi, transparent=True)
        fa.bind_high_peer(stub)
        stub.low_peer = fa
        stub.high_peer = ma
        ma.bind_low_peer(stub)

        # Transparent: both ends see straight through to each other.
        self.assertEqual(
            fa._current_gap_bytes(),
            ma._byte_low_frontier() - fa._byte_high_frontier(),
        )
        self.assertEqual(
            ma._current_gap_bytes(),
            ma._byte_low_frontier() - fa._byte_high_frontier(),
        )

        # Opaque: each end's gap stops at the middle's near frontier.
        stub.transparent = False
        self.assertEqual(fa._current_gap_bytes(), mid_lo - fa._byte_high_frontier())
        self.assertEqual(ma._current_gap_bytes(), ma._byte_low_frontier() - mid_hi)

    def test_multi_hop_walk_stops_at_first_opaque(self):
        _, fa, ma = self._build_pair()
        t1 = _ChainStub(low_byte=100, high_byte=100, transparent=True)
        t2 = _ChainStub(low_byte=200, high_byte=260, transparent=False)
        fa.bind_high_peer(t1)
        t1.low_peer = fa
        t1.high_peer = t2
        t2.low_peer = t1
        t2.high_peer = ma
        ma.bind_low_peer(t2)
        self.assertEqual(fa._current_gap_bytes(), 200 - fa._byte_high_frontier())
        self.assertIs(fa._growth_side_neighbor(), t2)
        t2.transparent = True
        self.assertIs(fa._growth_side_neighbor(), ma)
        self.assertEqual(
            fa._current_gap_bytes(),
            ma._byte_low_frontier() - fa._byte_high_frontier(),
        )

    def test_drainable_credit_reads_walked_neighbor(self):
        _, fa, ma = self._build_pair()
        stub = _ChainStub(low_byte=64, high_byte=128, transparent=True)
        fa.bind_high_peer(stub)
        stub.low_peer = fa
        stub.high_peer = ma
        ma.bind_low_peer(stub)

        # Walked-through to the far end: its holes are credited iff lazy.
        self.assertEqual(fa._peer_drainable_hole_bytes(), 0)  # ma not lazy
        ma.lazy_compaction = True
        ma._free_phys_pages = torch.arange(3, dtype=torch.int64)
        self.assertEqual(fa._peer_drainable_hole_bytes(), 3 * ma.entry_bytes_per_page)
        # Opaque non-lazy middle blocks the far end's credit.
        stub.transparent = False
        self.assertEqual(fa._peer_drainable_hole_bytes(), 0)


class TestFloatMultiEndedAllocator(unittest.TestCase):
    """Holes-first float middle: midpoint placement, in-place hole recycling,
    larger-gap extension, boundary absorption with park-on-empty transparency,
    and the on-demand movers `make_room` and `compact_holes`."""

    def _build_tri(self, n_state=8, n_float=32, n_full=32):
        state = _make_mamba_spec("state", "up", layer_num=2)
        fl = _make_mha_spec("swa", "float", layer_num=2)
        full = _make_mha_spec("full", "down", layer_num=2)
        total = (
            state.entry_bytes() * n_state
            + fl.entry_bytes() * n_float
            + full.entry_bytes() * n_full
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, fl, state],
            device=_DEV,
            enable_memory_saver=False,
        )
        sa = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("state")),
            unified_buffer=pool,
            sub_pool_name="state",
            device=_DEV,
            is_id_owner=True,
        )
        fkv = _FakeKVCache(pool.max_slots("swa"))
        fla = FloatMultiEndedAllocator(
            kvcache=fkv,
            unified_buffer=pool,
            sub_pool_name="swa",
            device=_DEV,
            is_id_owner=True,
        )
        dkv = _FakeKVCache(pool.max_slots("full"))
        da = MultiEndedAllocator(
            kvcache=dkv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
        )
        # Chain wiring state <-> float <-> full.
        sa.bind_high_peer(fla)
        fla.bind_low_peer(sa)
        fla.bind_high_peer(da)
        da.bind_low_peer(fla)
        return pool, sa, fla, da, fkv

    def _stamp(self, alloc, kv, v):
        kv.buf[alloc.virtual_to_physical[v]] = v

    def _interior_block(self, fla, blocks):
        """The allocated block whose physical pages touch neither span
        boundary (robust to the extend-direction policy)."""
        for v in blocks:
            pages = set(int(x) for x in fla.virtual_to_physical[v].tolist())
            if fla.low_wm_page not in pages and (fla.high_wm_page - 1) not in pages:
                return v
        raise AssertionError("no interior block in layout")

    def _check_float_state(self, fla, kv):
        holes = set(int(x) for x in fla._free_phys_pages.tolist())
        span = range(fla.low_wm_page, fla.high_wm_page)
        live = [p for p in span if p not in holes]
        self.assertEqual(fla._live_pages(), len(live))
        for h in holes:
            self.assertTrue(fla.low_wm_page < h < fla.high_wm_page - 1)
        for p in live:
            v = int(fla.physical_to_virtual[p].item())
            self.assertNotEqual(v, -1, f"live page {p} unbound")
            self.assertEqual(int(fla.virtual_to_physical[v].item()), p)
            self.assertEqual(int(kv.buf[p].item()), v, f"data lost at {p}")

    def test_midpoint_initial_placement(self):
        _, _, fla, _, _ = self._build_tri()
        self.assertTrue(fla._is_frontier_transparent())
        v = fla.alloc(4)
        self.assertIsNotNone(v)
        lo, hi = fla._region_bounds_pages()
        self.assertEqual(fla.low_wm_page, lo + (hi - lo - 4) // 2)
        self.assertEqual(fla._span_pages(), 4)
        # Gap on BOTH sides.
        gap_low, gap_high = fla._gap_pages()
        self.assertGreater(gap_low, 0)
        self.assertGreater(gap_high, 0)

    def test_holes_first_reuse_is_zero_copy(self):
        _, _, fla, _, kv = self._build_tri()
        va = fla.alloc(2)
        vb = fla.alloc(2)
        vc = fla.alloc(2)
        for v in (va, vb, vc):
            self._stamp(fla, kv, v)
        span_before = (fla.low_wm_page, fla.high_wm_page)
        fla.free(self._interior_block(fla, (va, vb, vc)))  # interior -> holes
        self.assertEqual(fla._hole_pages(), 2)
        self.assertEqual((fla.low_wm_page, fla.high_wm_page), span_before)
        vd = fla.alloc(2)  # must recycle the holes in place
        self._stamp(fla, kv, vd)
        self.assertEqual(fla._hole_pages(), 0)
        self.assertEqual((fla.low_wm_page, fla.high_wm_page), span_before)
        self.assertEqual(len(fla._inverse_history), 0)  # zero copies
        self._check_float_state(fla, kv)

    def test_boundary_free_absorbed_at_the_deferred_point(self):
        """Boundary holes shrink the span zero-copy, but the shrink is deferred
        out of `free`, which must stay host-sync-free; skipping the deferred
        absorb is only ever conservative."""
        _, _, fla, _, kv = self._build_tri()
        va = fla.alloc(2)
        vb = fla.alloc(2)  # extends one side; frees at that edge absorb
        self._stamp(fla, kv, va)
        self._stamp(fla, kv, vb)
        span = fla._span_pages()
        fla.free(vb)
        # Deferred: span still claims the freed edge, the pages are holes.
        self.assertEqual(fla._hole_pages(), 2)
        self.assertEqual(fla._span_pages(), span)
        self.assertEqual(fla._live_pages(), 2)  # exact regardless
        absorbed = fla._flush(urgent=False)
        self.assertEqual(absorbed, 2)
        self.assertEqual(fla._hole_pages(), 0)
        self.assertEqual(fla._span_pages(), span - 2)
        self.assertEqual(len(fla._inverse_history), 0)  # zero copies
        self._check_float_state(fla, kv)

    def test_free_is_host_sync_free(self):
        """No D2H anywhere in the float's free: absorption is deferred to
        `_flush`, so the per-decode-step path never syncs with the host."""
        from unittest import mock

        _, _, fla, _, kv = self._build_tri()
        v = fla.alloc(4)
        self._stamp(fla, kv, v)
        with (
            mock.patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("tolist = D2H")
            ),
            mock.patch.object(
                torch.Tensor, "item", side_effect=AssertionError("item = D2H")
            ),
            mock.patch.object(
                torch, "unique", side_effect=AssertionError("unique = host sync")
            ),
        ):
            fla.free(v[:2], _pages=v[:2])

    def test_deferred_absorption_reaches_the_same_state_as_eager(self):
        """Deferring absorption must not change WHERE the span lands, only when."""
        _, _, f1, _, kv1 = self._build_tri()
        _, _, f2, _, kv2 = self._build_tri()
        for f, kv, absorb_each in ((f1, kv1, True), (f2, kv2, False)):
            blocks = [f.alloc(2) for _ in range(3)]
            for v in blocks:
                self._stamp(f, kv, v)
            for v in (blocks[2], blocks[0]):
                f.free(v)
                if absorb_each:
                    f._flush(urgent=False)
        f2._flush(urgent=False)
        self.assertEqual(f1.low_wm_page, f2.low_wm_page)
        self.assertEqual(f1.high_wm_page, f2.high_wm_page)
        self.assertEqual(f1._hole_pages(), f2._hole_pages())
        self.assertEqual(f1.available_size(), f2.available_size())

    def test_park_on_empty_restores_transparency(self):
        _, sa, fla, da, _ = self._build_tri()
        base_gap = da._current_gap_bytes()
        v = fla.alloc(4)
        self.assertLess(da._current_gap_bytes(), base_gap)  # float blocks
        fla.free(v)
        self.assertTrue(fla._is_frontier_transparent())
        self.assertEqual(fla._hole_pages(), 0)
        self.assertEqual(da._current_gap_bytes(), base_gap)  # sees through again
        self.assertEqual(sa._current_gap_bytes(), base_gap)

    def test_extends_toward_larger_gap(self):
        _, _, fla, da, kv = self._build_tri()
        v = fla.alloc(4)
        self._stamp(fla, kv, v)
        # Consume most of the high gap with the full end; low gap now larger.
        self.assertIsNotNone(da.alloc(24))
        gap_low, gap_high = fla._gap_pages()
        self.assertGreater(gap_low, gap_high)
        lo_before = fla.low_wm_page
        hi_before = fla.high_wm_page
        v2 = fla.alloc(2)
        self.assertIsNotNone(v2)
        self._stamp(fla, kv, v2)
        self.assertEqual(fla.low_wm_page, lo_before - 2)  # grew low side
        self.assertEqual(fla.high_wm_page, hi_before)
        self._check_float_state(fla, kv)

    def test_available_is_max_gap_plus_holes(self):
        _, _, fla, da, _ = self._build_tri()
        va = fla.alloc(2)
        vb = fla.alloc(2)
        vc = fla.alloc(2)
        fla.free(self._interior_block(fla, (va, vb, vc)))
        self.assertEqual(fla._hole_pages(), 2)
        gap_low, gap_high = fla._gap_pages()
        self.assertEqual(fla.available_size(), max(gap_low, gap_high) + 2)
        del da

    def test_make_room_boundary_relocation(self):
        _, _, fla, da, kv = self._build_tri()
        v = fla.alloc(6)
        self._stamp(fla, kv, v)
        epp = fla.entry_bytes_per_page
        _, gap_high = fla._gap_pages()
        ask = (gap_high + 3) * epp  # 3 pages beyond the current high gap
        opened = fla.make_room(side="high", min_bytes=ask)
        self.assertGreaterEqual(opened, ask)
        # Cost min(L_live, G): moved exactly the 3 boundary pages, not 6.
        moved = sum(int(s.numel()) for s, _, _ in fla._inverse_history)
        self.assertEqual(moved, 3)
        self._check_float_state(fla, kv)
        # The opened space is real: the full end can now take it.
        self.assertGreaterEqual(da.available_size(), 3)

    def test_make_room_leapfrog_cost_bounded_by_live(self):
        _, _, fla, _, kv = self._build_tri()
        v = fla.alloc(2)  # tiny live mass
        self._stamp(fla, kv, v)
        epp = fla.entry_bytes_per_page
        _, gap_high = fla._gap_pages()
        ask = (gap_high + 10) * epp  # demand >> live
        opened = fla.make_room(side="high", min_bytes=ask)
        self.assertGreaterEqual(opened, ask)
        moved = sum(int(s.numel()) for s, _, _ in fla._inverse_history)
        self.assertEqual(moved, 2)  # min(L_live, G) == L_live
        self._check_float_state(fla, kv)

    def test_make_room_impossible_leaves_state_unchanged(self):
        _, _, fla, _, kv = self._build_tri(n_float=8)
        v = fla.alloc(6)
        self._stamp(fla, kv, v)
        lo, hi = fla._region_bounds_pages()
        epp = fla.entry_bytes_per_page
        snapshot = (
            fla.low_wm_page,
            fla.high_wm_page,
            fla._hole_pages(),
            len(fla._inverse_history),
        )
        opened = fla.make_room(side="high", min_bytes=(hi - lo) * epp)
        self.assertLess(opened, (hi - lo) * epp)
        self.assertEqual(
            snapshot,
            (
                fla.low_wm_page,
                fla.high_wm_page,
                fla._hole_pages(),
                len(fla._inverse_history),
            ),
        )
        self._check_float_state(fla, kv)

    def test_make_room_uses_far_holes_before_far_gap(self):
        _, _, fla, _, kv = self._build_tri()
        va = fla.alloc(2)
        vb = fla.alloc(2)
        vc = fla.alloc(2)
        for v in (va, vb, vc):
            self._stamp(fla, kv, v)
        lo_before = fla.low_wm_page
        fla.free(self._interior_block(fla, (va, vb, vc)))  # 2 interior holes
        self.assertEqual(fla._hole_pages(), 2)
        epp = fla.entry_bytes_per_page
        _, gap_high = fla._gap_pages()
        opened = fla.make_room(side="high", min_bytes=(gap_high + 2) * epp)
        self.assertGreaterEqual(opened, (gap_high + 2) * epp)
        # The two holes absorbed the two moved pages: low side untouched.
        self.assertEqual(fla.low_wm_page, lo_before)
        self.assertEqual(fla._hole_pages(), 0)
        self._check_float_state(fla, kv)

    def test_compact_holes_ordered_pack(self):
        _, _, fla, _, kv = self._build_tri()
        vs = [fla.alloc(2) for _ in range(4)]
        for v in vs:
            self._stamp(fla, kv, v)
        fla.free(vs[1])  # interleaved holes
        span_before = fla._span_pages()
        moved = fla.compact_holes(retreat_side="high")
        self.assertEqual(fla._hole_pages(), 0)
        self.assertEqual(fla._span_pages(), span_before - 2)
        self.assertGreater(moved, 0)
        self._check_float_state(fla, kv)


class TestDcpWidening(unittest.TestCase):
    """`dcp_size > 1`: the alloc surface speaks a widened virtual id space while
    the pool keeps storing one row per `dcp_size` logical ids."""

    @contextlib.contextmanager
    def _dcp(self, dcp_size, dcp_rank=0):
        """The width comes from the parallel context, not a constructor
        argument, so one scope has to hold construction and every read."""
        with get_parallel().override(
            dcp_enabled=dcp_size > 1,
            attn_dcp_size=dcp_size,
            attn_dcp_rank=dcp_rank,
        ):
            yield

    def _build_pair(self, *, page_size, n_full_slots=64):
        """(full, mamba) as the composite wires them: only full shards."""
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        pool = UnifiedKVPool(
            total_bytes=full.entry_bytes() * n_full_slots + mamba.entry_bytes() * 16,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        alloc = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("full")),
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            page_size=page_size,
            shards_under_dcp=True,
        )
        # The peer stays slot-granular: mamba state is replicated, not sharded.
        mamba = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("mamba")),
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
        )
        alloc.bind_peer(mamba)
        return alloc, mamba

    def _build(self, *, page_size, n_full_slots=64):
        return self._build_pair(page_size=page_size, n_full_slots=n_full_slots)[0]

    def test_replicated_peer_stays_slot_granular_under_dcp(self):
        """Regression: only the sharding sub-allocator widens under DCP. A
        widened Mamba page size fails the page-multiple check, since state is
        allocated one slot per request."""
        with self._dcp(4):
            _, mamba = self._build_pair(page_size=2)
            self.assertEqual(mamba.page_size, 1)
            self.assertEqual(mamba.page_size, mamba.pool_page_size)
            self.assertIsNotNone(mamba.alloc(1))

    def test_capacity_scales_but_physical_pages_do_not(self):
        for page_size in (1, 8):
            with self._dcp(1):
                base = self._build(page_size=page_size)
                base_pages = base.num_pages
                base_page_bytes = base.entry_bytes_per_page
                base_avail = base.available_size()
            for dcp_size in (2, 4):
                with self._dcp(dcp_size):
                    a = self._build(page_size=page_size)
                    self.assertEqual(a.page_size, page_size * dcp_size)
                    self.assertEqual(a.pool_page_size, page_size)
                    # Same rows, same bytes per page; only the id space grows.
                    self.assertEqual(a.num_pages, base_pages)
                    self.assertEqual(a.entry_bytes_per_page, base_page_bytes)
                    self.assertEqual(a.available_size(), base_avail * dcp_size)

    def test_alloc_returns_whole_widened_pages(self):
        with self._dcp(2, 1):
            a = self._build(page_size=4)
            ids = a.alloc(3 * 8)  # 3 widened pages of 4*2 ids
            self.assertIsNotNone(ids)
            pages = ids.view(3, 8)
            self.assertTrue(
                torch.equal(pages[:, 1:] - pages[:, :-1], torch.ones(3, 7).long())
            )
            self.assertTrue(bool((pages[:, 0] % 8 == 0).all()))
            # Freeing the widened ids releases exactly the pages they came from.
            before = a.available_size()
            a.free(ids)
            self.assertEqual(a.available_size(), before + 3 * 8)

    def test_every_rank_maps_a_widened_page_to_one_physical_page(self):
        """The DCP ranks must agree on the physical page a widened page uses;
        only the row WITHIN it differs, by `(loc % dcp) -> loc // dcp`."""
        dcp_size = 4
        with self._dcp(dcp_size):
            allocs = [self._build(page_size=2) for _ in range(dcp_size)]
            ids = [a.alloc(2 * dcp_size * 2) for a in allocs]
            for i in ids:
                self.assertIsNotNone(i)
            # Same allocation order -> same widened ids on every rank.
            for i in ids[1:]:
                self.assertTrue(torch.equal(i, ids[0]))
        for rank, (a, i) in enumerate(zip(allocs, ids)):
            with self._dcp(dcp_size, rank):
                owned = (i % dcp_size) == rank
                self.assertEqual(int(owned.sum()), i.numel() // dcp_size)
                phys = a.translate_kv_loc(i[owned] // dcp_size)
                # Collapsed ids land inside this rank's physical rows,
                # contiguously within each page, never on the reserved sink.
                self.assertTrue(bool((phys > 0).all()))
                self.assertTrue(bool((phys < a.max_slots).all()))
                self.assertEqual(len(set(phys.tolist())), phys.numel())

    def test_write_translate_tombstones_unowned_ids(self):
        dcp_size = 2
        for rank in range(dcp_size):
            with self._dcp(dcp_size, rank):
                a = self._build(page_size=2)
                ids = a.alloc(2 * dcp_size * 3)
                written = a.translate_write_loc_for_kernel(ids)
                owned = (ids % dcp_size) == rank
                # Owned ids agree with the read translate of the collapsed id...
                self.assertTrue(
                    torch.equal(
                        written[owned],
                        a.translate_kv_loc_for_kernel(ids[owned] // dcp_size),
                    )
                )
                # ...and the rest go to the sink the write kernels skip.
                self.assertTrue(bool((written[~owned] == 0).all()))
                self.assertTrue(bool((written[owned] > 0).all()))

    def _build_composite(self, *, page_size):
        from sglang.srt.mem_cache.allocator.unified_mamba import (
            UnifiedMambaTokenToKVPoolAllocator,
        )

        full_spec = _make_mha_spec("full", "up", layer_num=2)
        mamba_spec = _make_mamba_spec("mamba", "down", layer_num=2)
        pool = UnifiedKVPool(
            total_bytes=16 * page_size * full_spec.entry_bytes()
            + 8 * mamba_spec.entry_bytes(),
            sub_pool_specs=[full_spec, mamba_spec],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        mamba_kv._copy_from_physical = lambda src, dst: None

        class _FakeHybridLinearKVPool:
            full_kv_pool = full_kv
            mamba_pool = mamba_kv

        return UnifiedMambaTokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=_FakeHybridLinearKVPool(),
            device=_DEV,
            page_size=page_size,
            need_sort=False,
            forward_stream=None,
        )

    def test_mamba_slot_cost_is_in_the_same_units_as_available_size(self):
        """The planner charges `mamba_slot_full_token_cost()` against a budget
        fed by `available_size()`. Both are bytes/entry_bytes conversions, so
        both carry `dcp_size`; if only the budget widens, every Mamba state is
        under-reserved by that factor and a batch is admitted whose later
        allocations cross the shared byte frontier."""
        for page_size in (1, 8):
            with self._dcp(1):
                base = self._build_composite(page_size=page_size)
                base_cost = base.mamba_slot_full_token_cost()
                base_avail = base.available_size()
                self.assertGreater(base_cost, 0)
            for dcp_size in (2, 4):
                with self._dcp(dcp_size):
                    a = self._build_composite(page_size=page_size)
                    self.assertEqual(a.available_size(), base_avail * dcp_size)
                    mamba_bytes = a.mamba_allocator.entry_bytes_per_page
                    full_entry = a.full_attn_allocator.entry_bytes
                    cost = a.mamba_slot_full_token_cost()
                    # A widened token is `full_entry / dcp_size` bytes, so the
                    # reservation covers the slot...
                    self.assertGreaterEqual(cost * full_entry, mamba_bytes * dcp_size)
                    # ...and stays tight (rounds up by less than one token).
                    self.assertLess((cost - 1) * full_entry, mamba_bytes * dcp_size)
                    # The un-scaled cost -- the bug -- would not have covered it.
                    self.assertLess(base_cost * full_entry, mamba_bytes * dcp_size)


if __name__ == "__main__":
    unittest.main()
