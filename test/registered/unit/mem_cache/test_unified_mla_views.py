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
"""MLA views for the unified memory pool (MLA-hybrid-Mamba, Kimi K3).

Covers, CPU-only (pure torch — no GPU / Triton kernels):
  - `MLASubPoolSpec` byte math (aligned entry, one latent row per layer);
  - `build_dense_views` addressing: view_l[t] for PHYSICAL token t must land
    at the envelope byte `t * entry_bytes + l * row_bytes`, the per-layer
    views must not alias at equal ids, and a short buffer must fail loud;
  - `UnifiedKVPool` MLA plumbing: the allocation is exactly the budget and
    the reserved sink floor covers the whole page-0 envelope;
  - `UnifiedMLATokenToKVPool`: buffer wiring, V-as-prefix-slice, and the
    page-envelope `move_kv_cache` (physical token ids, page-major runs);
  - `MultiEndedAllocator.translate_kv_loc`: the v2p formula, tombstone clamp
    to the sink, `out=` contract, int32 2-D page tables, and correctness
    across eager compaction.

GPU parity of the actual read/write kernels (set_mla_kv_buffer TMA path etc.)
lives in the server-level tests, not here.

    python -m pytest test/registered/unit/mem_cache/test_unified_mla_views.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.layout.page_major import (
    ENTRY_ALIGN_BYTES,
    build_dense_views,
    mla_entry_bytes,
    paged_row_view,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMLATokenToKVPool,
)

_DEV = "cpu"

# Small-but-nontrivial MLA geometry: L=3 layers, D=8 (=6+2), so every byte
# offset is hand-checkable. Real K3 is L=24, D=576 (=512+64). 3 rows x 16 B
# = 48 B of payload round up to one 64 B entry.
_L = 3
_LORA = 6
_ROPE = 2
_D = _LORA + _ROPE
_DTYPE = torch.bfloat16
_ITEM = _DTYPE.itemsize
_ROW = _D * _ITEM
_ENTRY = mla_entry_bytes(layer_num=_L, kv_cache_dim=_D, itemsize=_ITEM)
_E_ELEMS = _ENTRY // _ITEM


def _mla_spec(grow="down", layer_num=_L):
    return MLASubPoolSpec(
        name="full",
        layer_num=layer_num,
        kv_lora_rank=_LORA,
        qk_rope_head_dim=_ROPE,
        store_dtype=_DTYPE,
        grow_direction=grow,
    )


def _mamba_spec(grow="up", layer_num=2):
    return MambaSubPoolSpec(
        name="mamba",
        layer_num=layer_num,
        conv_state_shapes=((4, 3),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2, 2, 2),
        temporal_dtype=torch.float32,
        grow_direction=grow,
    )


def _make_unified(page_size=1, n_full_tokens=64, n_mamba_slots=8):
    full = _mla_spec()
    mamba = _mamba_spec()
    total = full.entry_bytes() * n_full_tokens + mamba.entry_bytes() * n_mamba_slots
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full, mamba],
        device=_DEV,
        enable_memory_saver=False,
        page_size=page_size,
    )
    return pool, full, mamba


def _build_views(raw, ps, num_pages):
    layout = _mla_spec().layout()
    return build_dense_views(
        raw, layout=layout, part=layout.part("kv"), page_size=ps, num_pages=num_pages
    )


class TestMLASubPoolSpec(unittest.TestCase):
    def test_entry_bytes_and_dim(self):
        spec = _mla_spec()
        self.assertEqual(spec.kv_cache_dim, _D)
        self.assertEqual(spec.row_bytes(), _ROW)
        self.assertEqual(spec.entry_bytes(), _ENTRY)
        self.assertGreaterEqual(spec.entry_bytes(), _L * _ROW)
        self.assertEqual(spec.entry_bytes() % ENTRY_ALIGN_BYTES, 0)
        self.assertEqual(spec.get_dtype(), _DTYPE)
        layout = spec.layout()
        self.assertEqual(layout.entry_bytes, _ENTRY)
        for l in range(_L):
            self.assertEqual(layout.part("kv").layer_offset_bytes(l), l * _ROW)

    def test_rejects_nonpositive_dims(self):
        with self.assertRaises(AssertionError):
            MLASubPoolSpec(
                name="full",
                layer_num=_L,
                kv_lora_rank=0,
                qk_rope_head_dim=_ROPE,
                store_dtype=_DTYPE,
                grow_direction="down",
            )


class TestMLAViews(unittest.TestCase):
    def _make_raw(self, ps, num_pages, short=0):
        n = num_pages * ps * _ENTRY - short
        return torch.zeros(n, dtype=torch.uint8, device=_DEV)

    def test_view_addressing_matches_envelope_formula(self):
        for ps in (1, 4):
            num_pages = 6
            raw = self._make_raw(ps, num_pages)
            views = _build_views(raw, ps, num_pages)
            self.assertEqual(len(views), _L)
            n_rows = num_pages * ps
            for v in views:
                self.assertEqual(tuple(v.shape), (n_rows, 1, _D))
                self.assertEqual(v.stride(), (_E_ELEMS, _D, 1))
            flat = raw.view(_DTYPE)
            for p, l, s in [(0, 0, 0), (1, 2, ps - 1), (4, 1, ps // 2), (5, 2, 0)]:
                t = p * ps + s
                marker = float(p * 100 + l * 10 + s + 1)
                views[l][t] = marker
                elem = t * _E_ELEMS + l * _D  # envelope formula, in elements
                self.assertTrue(
                    torch.all(flat[elem : elem + _D] == marker),
                    f"(p={p}, l={l}, s={s}, ps={ps}) landed off-formula",
                )

    def test_paged_row_view_keeps_the_slot_stride(self):
        """BUG REGRESSION at page_size 1: the paged MLA backends hand the
        kernels `paged_row_view(kv)`, whose dim 1 must carry the entry stride
        at every page size; a `view`-built split gave the size-1 slot dim the
        row stride instead."""
        num_pages = 3
        for ps in (1, 4):
            views = _build_views(self._make_raw(ps, num_pages), ps, num_pages)
            paged = paged_row_view(views[1], ps)
            self.assertEqual(tuple(paged.shape), (num_pages, ps, _D), ps)
            self.assertEqual(tuple(paged.stride()), (ps * _E_ELEMS, _E_ELEMS, 1), ps)
            for t in range(num_pages * ps):
                self.assertEqual(
                    paged[t // ps, t % ps].data_ptr(),
                    views[1][t].data_ptr(),
                    (ps, t),
                )

    def test_views_do_not_alias_across_layers(self):
        ps, num_pages = 4, 4
        views = _build_views(self._make_raw(ps, num_pages), ps, num_pages)
        t = 2 * ps + 1  # page 2, slot 1
        for l in range(_L):
            views[l][t] = float(l + 1)
        for l in range(_L):
            self.assertTrue(torch.all(views[l][t] == float(l + 1)))

    def test_short_buffer_fails_loud(self):
        ps, num_pages = 2, 4
        with self.assertRaises(AssertionError):
            _build_views(self._make_raw(ps, num_pages, short=1), ps, num_pages)


class TestUnifiedKVPoolMLA(unittest.TestCase):
    def test_raw_is_exactly_the_budget(self):
        pool, full, mamba = _make_unified(page_size=4)
        total = full.entry_bytes() * 64 + mamba.entry_bytes() * 8
        self.assertEqual(pool.max_slots("full"), total // full.entry_bytes())
        self.assertEqual(pool.max_slots("mamba"), total // mamba.entry_bytes())
        self.assertEqual(pool._raw.numel(), total)

    def test_reserved_floor_covers_page0_envelope(self):
        ps = 4
        pool, full, mamba = _make_unified(page_size=ps)
        floor = max(
            max(full.entry_bytes(), mamba.entry_bytes()), ps * full.entry_bytes()
        )
        for spec in (full, mamba):
            self.assertGreaterEqual(
                pool.min_slot_index(spec.name) * spec.entry_bytes(), floor
            )

    def test_mla_views_accessor(self):
        pool, full, _ = _make_unified(page_size=1)
        views = pool.mla_views_for("full")
        self.assertEqual(len(views), _L)
        self.assertIs(pool.mla_spec("full"), full)
        self.assertEqual(views[0].stride(0) * _ITEM, full.entry_bytes())


class TestUnifiedMLATokenToKVPool(unittest.TestCase):
    def _make(self, ps=1):
        pool, full, mamba = _make_unified(page_size=ps)
        kv_pool = UnifiedMLATokenToKVPool(
            unified_buffer=pool,
            sub_pool_name="full",
            kv_cache_dtype=_DTYPE,
            page_size=ps,
        )
        return pool, kv_pool

    def test_buffers_and_prefix_value_slice(self):
        pool, kv_pool = self._make(ps=1)
        self.assertEqual(len(kv_pool.kv_buffer), _L)
        self.assertEqual(kv_pool.get_kv_size_bytes(), 0)
        self.assertEqual(kv_pool.size, pool.max_slots("full") - 1)
        k = kv_pool.get_key_buffer(1)
        v = kv_pool.get_value_buffer(1)
        self.assertEqual(k.shape[-1], _D)
        self.assertEqual(v.shape[-1], _LORA)
        # V is a prefix slice of K's storage: writing K shows up in V
        k[7] = 2.5
        self.assertTrue(torch.all(v[7] == 2.5))

    def test_move_kv_cache_moves_page_envelopes(self):
        for ps in (1, 4):
            pool, kv_pool = self._make(ps=ps)
            num_pages = pool.max_slots("full") // ps
            page_bytes = ps * pool.mla_spec("full").entry_bytes()
            env = pool._raw[: num_pages * page_bytes].view(num_pages, page_bytes)
            src_pages = torch.tensor([num_pages - 2, num_pages - 4])
            dst_pages = torch.tensor([2, 3])
            env[src_pages[0]] = 7
            env[src_pages[1]] = 9
            # page-major token runs, exactly how compaction expands pages
            offsets = torch.arange(ps, dtype=torch.int64)
            src_t = (src_pages[:, None] * ps + offsets).reshape(-1)
            dst_t = (dst_pages[:, None] * ps + offsets).reshape(-1)
            kv_pool.move_kv_cache(dst_t, src_t)
            self.assertTrue(torch.all(env[dst_pages[0]] == 7), f"ps={ps}")
            self.assertTrue(torch.all(env[dst_pages[1]] == 9), f"ps={ps}")

    def test_move_then_readback(self):
        ps = 4
        pool, kv_pool = self._make(ps=ps)
        num_pages = pool.max_slots("full") // ps
        src_page, dst_page = num_pages - 3, 5
        # write through the views at src, expect it at dst after the move
        for l in range(_L):
            for s in range(ps):
                kv_pool.kv_buffer[l][src_page * ps + s] = float(l * ps + s + 1)
        offsets = torch.arange(ps, dtype=torch.int64)
        kv_pool.move_kv_cache(
            (torch.tensor([dst_page])[:, None] * ps + offsets).reshape(-1),
            (torch.tensor([src_page])[:, None] * ps + offsets).reshape(-1),
        )
        for l in range(_L):
            for s in range(ps):
                got = kv_pool.kv_buffer[l][dst_page * ps + s]
                self.assertTrue(
                    torch.all(got == float(l * ps + s + 1)), f"(l={l}, s={s})"
                )


class _FakeKVCache:
    def __init__(self, max_slots: int):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class TestTranslateKvLoc(unittest.TestCase):
    def _build(self, ps=1, n_full_tokens=64):
        pool, full, mamba = _make_unified(page_size=ps, n_full_tokens=n_full_tokens)
        full_alloc = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("full")),
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            page_size=ps,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("mamba")),
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        return full_alloc

    def test_translate_matches_v2p_formula(self):
        for ps in (1, 4):
            alloc = self._build(ps=ps)
            v = alloc.alloc(3 * ps)
            self.assertIsNotNone(v)
            v2p = alloc.virtual_to_physical
            want = v2p[v // ps] * ps + v % ps
            self.assertTrue(torch.equal(alloc.translate_kv_loc(v), want), f"ps={ps}")

    def test_tombstone_clamps_to_sink(self):
        alloc = self._build(ps=1)
        # never-allocated virtual ids -> v2p == -1 -> id 0
        virt = torch.tensor([alloc.min_slot_index + 1], dtype=torch.int64)
        self.assertTrue(torch.all(alloc.translate_kv_loc(virt) == 0))

    def test_out_matches_and_aliases(self):
        for ps in (1, 4):
            alloc = self._build(ps=ps)
            v = alloc.alloc(2 * ps)
            self.assertIsNotNone(v)
            no_out = alloc.translate_kv_loc(v)
            out = torch.empty_like(v)
            ret = alloc.translate_kv_loc(v, out=out)
            self.assertIs(ret, out)
            self.assertTrue(torch.all(out == no_out))
            # canonical in-place aliasing: translate(x, out=x)
            x = v.clone()
            alloc.translate_kv_loc(x, out=x)
            self.assertTrue(torch.all(x == no_out))

    def test_accepts_an_int32_2d_page_table(self):
        """fa3 translates its own page table, which is int32 and 2-D; a gather
        that needs a 1-D int64 index would crash the scheduler there."""
        for ps in (1, 4):
            alloc = self._build(ps=ps)
            v = alloc.alloc(4 * ps)
            self.assertIsNotNone(v)
            want = alloc.translate_kv_loc(v)
            page_table = v.to(torch.int32).view(2, -1)
            got = alloc.translate_kv_loc(page_table)
            self.assertEqual(got.shape, page_table.shape)
            self.assertTrue(torch.equal(got.reshape(-1), want))
            dst = torch.empty(page_table.shape, dtype=torch.int64)
            alloc.translate_kv_loc(page_table, out=dst)
            self.assertTrue(torch.equal(dst.reshape(-1), want))

    def test_translate_follows_compaction(self):
        alloc = self._build(ps=1)
        a = alloc.alloc(4)
        b = alloc.alloc(4)
        c = alloc.alloc(4)
        self.assertIsNotNone(c)
        alloc.free(b)  # eager compaction relocates survivors
        for run in (a, c):
            self.assertTrue(
                torch.equal(alloc.translate_kv_loc(run), alloc.virtual_to_physical[run])
            )


if __name__ == "__main__":
    unittest.main()
