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
"""MLA views for the unified memory pool (MLA-hybrid-Mamba, Kimi K3), CPU-only.

Addressing law under test: the (page, layer, slot) cell sits at envelope byte
offset `p*(L*ps*D) + l*(ps*D) + s*D`, reached through the kernel-facing id
`(t // ps) * (ps * L) + t % ps`.

GPU parity of the read/write kernels (set_mla_kv_buffer TMA path etc.) lives in
`test_unified_mla_gpu_parity.py`.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.layout.page_major import build_mla_views
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMLATokenToKVPool,
)

_DEV = "cpu"

# Geometry kept tiny so every byte offset is hand-checkable; real K3 is
# L=24, D=576 (=512+64).
_L = 3
_LORA = 6
_ROPE = 2
_D = _LORA + _ROPE
_DTYPE = torch.bfloat16
_ITEM = _DTYPE.itemsize


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


def _kernel_id(t, ps, layer_num):
    return (t // ps) * (ps * layer_num) + t % ps


class TestMLASubPoolSpec(unittest.TestCase):
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
    def _make_raw(self, ps, num_pages, pad_pages=1):
        page_bytes = ps * _L * _D * _ITEM
        raw = torch.zeros(
            (num_pages + pad_pages) * page_bytes, dtype=torch.uint8, device=_DEV
        )
        return raw, page_bytes

    def test_view_addressing_matches_envelope_formula(self):
        for ps in (1, 4):
            num_pages = 6
            raw, _ = self._make_raw(ps, num_pages)
            views = build_mla_views(
                raw,
                layer_num=_L,
                kv_cache_dim=_D,
                store_dtype=_DTYPE,
                page_size=ps,
                num_pages=num_pages,
            )
            self.assertEqual(len(views), _L)
            n_rows = num_pages * _L * ps
            for v in views:
                self.assertEqual(tuple(v.shape), (n_rows, 1, _D))
                # contiguous in the (row, dim) sense: .view(-1, ps, D) legality
                self.assertEqual(v.stride(0), _D)
                self.assertEqual(v.stride(2), 1)
            flat = raw.view(_DTYPE)
            for p, l, s in [(0, 0, 0), (1, 2, ps - 1), (4, 1, ps // 2), (5, 2, 0)]:
                t = p * ps + s
                marker = float(p * 100 + l * 10 + s + 1)
                views[l][_kernel_id(t, ps, _L)] = marker
                # envelope formula, in elements
                elem = p * (_L * ps * _D) + l * (ps * _D) + s * _D
                self.assertTrue(
                    torch.all(flat[elem : elem + _D] == marker),
                    f"(p={p}, l={l}, s={s}, ps={ps}) landed off-formula",
                )

    def test_views_do_not_alias_across_layers(self):
        ps = 4
        num_pages = 4
        raw, _ = self._make_raw(ps, num_pages)
        views = build_mla_views(
            raw,
            layer_num=_L,
            kv_cache_dim=_D,
            store_dtype=_DTYPE,
            page_size=ps,
            num_pages=num_pages,
        )
        t = 2 * ps + 1  # page 2, slot 1
        d = _kernel_id(t, ps, _L)
        for l in range(_L):
            views[l][d] = float(l + 1)
        for l in range(_L):
            self.assertTrue(torch.all(views[l][d] == float(l + 1)))

    def test_missing_tail_pad_fails_loud(self):
        ps = 2
        num_pages = 4
        raw, _ = self._make_raw(ps, num_pages, pad_pages=0)
        with self.assertRaises(AssertionError):
            build_mla_views(
                raw,
                layer_num=_L,
                kv_cache_dim=_D,
                store_dtype=_DTYPE,
                page_size=ps,
                num_pages=num_pages,
            )


class TestUnifiedKVPoolMLA(unittest.TestCase):
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
        k = kv_pool.get_key_buffer(1)
        v = kv_pool.get_value_buffer(1)
        self.assertEqual(k.shape[-1], _D)
        self.assertEqual(v.shape[-1], _LORA)
        # V is a prefix slice of K's storage: writing K shows up in V
        k[7] = 2.5
        self.assertTrue(torch.all(v[7] == 2.5))

    def test_move_kv_cache_moves_page_envelopes(self):
        """Whole page envelopes relocate, in raw bytes and (at ps=4) as read
        back through the per-layer views at the destination kernel ids."""
        for ps in (1, 4):
            pool, kv_pool = self._make(ps=ps)
            num_pages = pool.max_slots("full") // ps
            page_bytes = ps * _L * _D * _ITEM
            env = pool._raw[: num_pages * page_bytes].view(num_pages, page_bytes)
            src_pages = torch.tensor([num_pages - 2, num_pages - 4, num_pages - 3])
            dst_pages = torch.tensor([2, 3, 5])
            env[src_pages[0]] = 7
            env[src_pages[1]] = 9
            if ps == 4:
                # write through the views at src, expect it at dst after the move
                for l in range(_L):
                    for s in range(ps):
                        kv_pool.kv_buffer[l][
                            _kernel_id(int(src_pages[2]) * ps + s, ps, _L)
                        ] = float(l * ps + s + 1)
            # page-major token runs, exactly how compaction expands pages
            offsets = torch.arange(ps, dtype=torch.int64)
            src_t = (src_pages[:, None] * ps + offsets).reshape(-1)
            dst_t = (dst_pages[:, None] * ps + offsets).reshape(-1)
            kv_pool.move_kv_cache(dst_t, src_t)
            self.assertTrue(torch.all(env[dst_pages[0]] == 7), f"ps={ps}")
            self.assertTrue(torch.all(env[dst_pages[1]] == 9), f"ps={ps}")
            if ps == 4:
                for l in range(_L):
                    for s in range(ps):
                        got = kv_pool.kv_buffer[l][
                            _kernel_id(int(dst_pages[2]) * ps + s, ps, _L)
                        ]
                        self.assertTrue(
                            torch.all(got == float(l * ps + s + 1)), f"(l={l}, s={s})"
                        )


class _FakeKVCache:
    def __init__(self, max_slots: int):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class TestTranslateKvLocForKernel(unittest.TestCase):
    def _build(self, ps=1, n_full_tokens=64, multiplier=_L):
        pool, full, mamba = _make_unified(page_size=ps, n_full_tokens=n_full_tokens)
        full_alloc = MultiEndedAllocator(
            kvcache=_FakeKVCache(pool.max_slots("full")),
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            page_size=ps,
            kernel_page_multiplier=multiplier,
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

    def test_kernel_id_matches_formula(self):
        """kernel id = (phys // ps) * (ps * multiplier) + phys % ps, across
        page sizes, the multiplier-1 physical fallback, and eager compaction."""
        for ps, multiplier in ((1, _L), (4, _L), (1, 1)):
            with self.subTest(page_size=ps, multiplier=multiplier):
                alloc = self._build(ps=ps, multiplier=multiplier)
                a = alloc.alloc(4 * ps)
                b = alloc.alloc(4 * ps)
                c = alloc.alloc(4 * ps)
                self.assertIsNotNone(c)

                def check(virt):
                    phys = alloc.translate_kv_loc(virt)
                    expected = (phys // ps) * (ps * multiplier) + phys % ps
                    self.assertTrue(
                        torch.all(alloc.translate_kv_loc_for_kernel(virt) == expected)
                    )

                for virt in (a, b, c):
                    check(virt)
                alloc.free(b)  # eager compaction relocates survivors
                for virt in (a, c):
                    check(virt)

    def test_tombstone_clamps_to_sink(self):
        alloc = self._build(ps=1)
        # never-allocated virtual ids -> v2p == -1 -> kernel-facing id 0
        virt = torch.tensor([alloc.min_slot_index + 1], dtype=torch.int64)
        kernel = alloc.translate_kv_loc_for_kernel(virt)
        self.assertTrue(torch.all(kernel == 0))

    def test_out_matches_and_aliases(self):
        for ps in (1, 4):
            alloc = self._build(ps=ps)
            v = alloc.alloc(2 * ps)
            self.assertIsNotNone(v)
            no_out = alloc.translate_kv_loc_for_kernel(v)
            out = torch.empty_like(v)
            ret = alloc.translate_kv_loc_for_kernel(v, out=out)
            self.assertIs(ret, out)
            self.assertTrue(torch.all(out == no_out))
            # canonical in-place aliasing: translate(x, out=x)
            x = v.clone()
            alloc.translate_kv_loc_for_kernel(x, out=x)
            self.assertTrue(torch.all(x == no_out))


if __name__ == "__main__":
    unittest.main()
