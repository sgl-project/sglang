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
"""Dense fused-draft geometry: the per-PAGE fusion of a draft KV region.

Derived-property pins for the fused page layout

    [ host: 2*L_h blocks @ w_h | draft: 2*L_d blocks @ w_d | pad ]

with per-slot lcm(w_h, w_d) padding: both families' dense page strides must
be integral in their OWN row units (that is what lets the existing
`translate_kv_loc_for_kernel` / KVIndexTranslator machinery serve the draft with just
a different `kernel_page_multiplier`), the generalized view builder must
reproduce the unfused layout byte-identically at its defaults, and writes
through host and draft views must land in their own page regions (compaction
moves whole page envelopes, so region confinement IS the correctness of the
fused move).

    python -m pytest test/registered/unit/mem_cache/test_dense_fused_draft_geometry.py -v
"""

import math
import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import build_mha_views
from sglang.srt.mem_cache.unified_memory_pool import (
    DenseDraftRegion,
    MHASubPoolSpec,
    UnifiedDraftKVPool,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"


def _host_spec(draft_region=None):
    return MHASubPoolSpec(
        name="full",
        layer_num=2,
        head_num=2,
        head_dim=4,
        store_dtype=torch.bfloat16,
        grow_direction="down",
        draft_region=draft_region,
    )


def _draft_region():
    # Deliberately NOT a divisor/multiple of the host row width:
    # w_h = 2*4*2 = 16 B, w_d = 1*3*2 = 6 B, lcm = 48 B — real padding.
    return DenseDraftRegion(
        layer_num=1, head_num=1, head_dim=3, store_dtype=torch.bfloat16
    )


class TestFusedSpecMath(unittest.TestCase):
    def test_unfused_spec_is_byte_identical_to_before(self):
        s = _host_spec()
        self.assertEqual(s.entry_bytes(), 2 * (16 + 16))
        self.assertEqual(s.blocks_per_page(), 4)
        self.assertEqual(s.view_tail_pad_bytes(4), 4 * s.entry_bytes())

    def test_fused_entry_is_lcm_padded_and_both_strides_integral(self):
        f = _host_spec(_draft_region())
        w_h, w_d = 16, 6
        raw = f.host_entry_bytes() + f.draft_region.entry_bytes()
        quantum = math.lcm(w_h, w_d)
        self.assertEqual(f.entry_bytes(), -(-raw // quantum) * quantum)
        self.assertEqual(f.entry_bytes() % w_h, 0)
        self.assertEqual(f.entry_bytes() % w_d, 0)
        self.assertEqual(f.blocks_per_page(), f.entry_bytes() // w_h)
        self.assertEqual(f.draft_kernel_page_multiplier(), f.entry_bytes() // w_d)
        # The draft region begins exactly where the host blocks end.
        ps = 4
        self.assertEqual(f.draft_region_offset_in_page(ps), ps * f.host_entry_bytes())


class TestGeneralizedBuilderDefaults(unittest.TestCase):
    def test_default_params_reproduce_the_unfused_views(self):
        # Same geometry, built with and without the new parameters spelled
        # out: every view's size/stride/storage_offset must match.
        kwargs = dict(
            layer_num=2,
            head_num=2,
            head_dim=4,
            v_head_dim=4,
            store_dtype=torch.bfloat16,
            page_size=4,
            num_pages=3,
        )
        spec = _host_spec()
        total = 3 * 4 * spec.entry_bytes() + spec.view_tail_pad_bytes(4)
        raw = torch.zeros(total, dtype=torch.uint8)
        k0, v0 = build_mha_views(raw, **kwargs)
        k1, v1 = build_mha_views(
            raw, **kwargs, page_stride_blocks=4, region_offset_bytes=0
        )
        for a, b in zip(k0 + v0, k1 + v1):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.storage_offset(), b.storage_offset())


class TestFusedRegionConfinement(unittest.TestCase):
    """Writes through host/draft views must land in their own page regions."""

    PS = 2
    PAGES = 3

    def _build(self):
        spec = _host_spec(_draft_region())
        total = self.PAGES * self.PS * spec.entry_bytes() + spec.view_tail_pad_bytes(
            self.PS
        )
        raw = torch.zeros(total, dtype=torch.uint8)
        hk, hv = build_mha_views(
            raw,
            layer_num=spec.layer_num,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            v_head_dim=spec.v_head_dim,
            store_dtype=spec.store_dtype,
            page_size=self.PS,
            num_pages=self.PAGES,
            page_stride_blocks=spec.blocks_per_page(),
        )
        r = spec.draft_region
        dk, dv = build_mha_views(
            raw,
            layer_num=r.layer_num,
            head_num=r.head_num,
            head_dim=r.head_dim,
            v_head_dim=r.head_dim,
            store_dtype=r.store_dtype,
            page_size=self.PS,
            num_pages=self.PAGES,
            page_stride_blocks=spec.draft_kernel_page_multiplier(),
            region_offset_bytes=spec.draft_region_offset_in_page(self.PS),
        )
        return spec, raw, (hk, hv), (dk, dv)

    def _page_region(self, spec, page, *, draft):
        page_bytes = self.PS * spec.entry_bytes()
        start = page * page_bytes
        split = start + spec.draft_region_offset_in_page(self.PS)
        if draft:
            return split, split + self.PS * spec.draft_region.entry_bytes()
        return start, split

    def test_host_and_draft_writes_confine_to_their_regions(self):
        spec, raw, (hk, hv), (dk, dv) = self._build()
        for page in range(self.PAGES):
            for off in range(self.PS):
                token_dense_h = page * self.PS * spec.blocks_per_page() + off
                token_dense_d = (
                    page * self.PS * spec.draft_kernel_page_multiplier() + off
                )
                for views, dense_id, draft in (
                    ((hk, hv), token_dense_h, False),
                    ((dk, dv), token_dense_d, True),
                ):
                    for family in views:
                        for layer_view in family:
                            raw.zero_()
                            layer_view[dense_id] = 1.0
                            nz = raw.nonzero()
                            self.assertGreater(nz.numel(), 0)
                            lo, hi = self._page_region(spec, page, draft=draft)
                            self.assertTrue(
                                bool((nz >= lo).all() and (nz < hi).all()),
                                f"write leaked outside its region: page={page} "
                                f"off={off} draft={draft} "
                                f"bytes=[{int(nz.min())},{int(nz.max())}] "
                                f"region=[{lo},{hi})",
                            )

    def test_pool_helper_builds_the_draft_views(self):
        spec = _host_spec(_draft_region())
        swa = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=4,
            store_dtype=torch.bfloat16,
            grow_direction="up",
        )
        total = 8 * self.PS * spec.entry_bytes() + 8 * self.PS * swa.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[spec, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
        )
        dk, dv = pool.build_dense_draft_views("full")
        self.assertEqual(len(dk), 1)
        self.assertEqual(dk[0].shape[1:], (1, 3))
        with self.assertRaises(AssertionError):
            pool.build_dense_draft_views("swa")  # no fused region there


class TestUnifiedDraftKVPool(unittest.TestCase):
    PS = 2
    PAGES = 8

    def _pool(self):
        spec = _host_spec(_draft_region())
        swa = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=4,
            store_dtype=torch.bfloat16,
            grow_direction="up",
        )
        total = self.PAGES * self.PS * (spec.entry_bytes() + swa.entry_bytes())
        return UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[spec, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
        )

    def _draft_pool(self, pool):
        sentinel_allocator = object()
        dp = UnifiedDraftKVPool(
            unified_buffer=pool,
            host_sub_pool_name="full",
            host_allocator=sentinel_allocator,
            page_size=self.PS,
        )
        return dp, sentinel_allocator

    def test_probe_surface_and_view_binding(self):
        pool = self._pool()
        dp, alloc = self._draft_pool(pool)
        spec = pool.mha_spec("full")
        self.assertIs(dp.host_allocator, alloc)
        self.assertEqual(
            dp.draft_kernel_page_multiplier, spec.draft_kernel_page_multiplier()
        )
        self.assertEqual(len(dp.k_buffer), 1)
        self.assertEqual(dp.k_buffer[0].shape[1:], (1, 3))

    def test_host_page_move_carries_the_draft_bytes(self):
        # THE fused-layout property: compaction relocates whole page envelopes
        # on the HOST pool; a draft marker written in page A must arrive at
        # page B after host.move_kv_cache(B, A) — with zero draft-side moves.
        pool = self._pool()
        dp, _ = self._draft_pool(pool)
        host = UnifiedMHATokenToKVPool(
            unified_buffer=pool, sub_pool_name="full", page_size=self.PS
        )
        src_page, dst_page = 3, 5
        mult_d = dp.draft_kernel_page_multiplier
        src_dense_d = src_page * self.PS * mult_d
        dst_dense_d = dst_page * self.PS * mult_d
        dp.k_buffer[0][src_dense_d] = 7.0
        self.assertEqual(float(dp.k_buffer[0][dst_dense_d].sum()), 0.0)

        ps = self.PS
        to_tokens = lambda p: torch.arange(p * ps, (p + 1) * ps, dtype=torch.int64)
        host.move_kv_cache(to_tokens(dst_page), to_tokens(src_page))
        self.assertEqual(float(dp.k_buffer[0][dst_dense_d].sum()), 7.0 * 3)

    def test_draft_side_moves_and_transfers_fail_loudly(self):
        pool = self._pool()
        dp, _ = self._draft_pool(pool)
        one = torch.zeros(self.PS, dtype=torch.int64)
        with self.assertRaises(NotImplementedError):
            dp.move_kv_cache(one, one)
        with self.assertRaises(NotImplementedError):
            dp.get_contiguous_buf_infos()
        with self.assertRaises(NotImplementedError):
            dp.get_cpu_copy(one)


if __name__ == "__main__":
    unittest.main()
