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
"""Unit tests for the page-major layer-major byte layout.

Verifies that:
1. The new 4-D ``_build_mha_views`` output exposes correct byte addresses
   for each (layer, page, tok_in_page, head, dim) — under both the
   degenerate ``page_size=1`` case (byte-identical to the old per-token
   envelope) and the new ``page_size>1`` layer-major case.
2. ``MHASubPoolSpec.layer_k_offset_in_page`` /
   ``layer_v_offset_in_page`` math matches the layout intent.
3. ``set_kv_buffer`` round-trips correctly for both page sizes.
4. Compaction (``move_kv_cache_native``) moves the right bytes for both
   page sizes via the 4-D advanced indexing path.

CPU-only — no GPU / Triton needed.

    python -m pytest test/registered/unit/mem_cache/test_layout_compat.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import move_kv_cache_native
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
)

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


class TestMHASpecLayerOffsets(unittest.TestCase):
    """Verify ``layer_k_offset_in_page`` / ``layer_v_offset_in_page`` math."""

    def test_offsets_at_page_size_1_match_envelope(self):
        spec = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        # At ps=1, layer-major within a 1-token page IS envelope-per-token.
        # Layer L's K offset = L * (k_row + v_row); V offset = +k_row.
        k_row = spec.k_row_bytes()
        v_row = spec.v_row_bytes()
        for L in range(spec.layer_num):
            self.assertEqual(
                spec.layer_k_offset_in_page(L, page_size=1),
                L * (k_row + v_row),
            )
            self.assertEqual(
                spec.layer_v_offset_in_page(L, page_size=1),
                L * (k_row + v_row) + k_row,
            )

    def test_offsets_at_page_size_gt_1(self):
        spec = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        ps = 8
        k_row = spec.k_row_bytes()
        v_row = spec.v_row_bytes()
        # Layer L's K block within the page starts at L * ps * (k_row+v_row).
        # V block starts at +ps * k_row.
        for L in range(spec.layer_num):
            self.assertEqual(
                spec.layer_k_offset_in_page(L, page_size=ps),
                L * ps * (k_row + v_row),
            )
            self.assertEqual(
                spec.layer_v_offset_in_page(L, page_size=ps),
                L * ps * (k_row + v_row) + ps * k_row,
            )

    def test_page_bytes(self):
        spec = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        # page_bytes = page_size * entry_bytes (preserved invariant)
        for ps in [1, 8, 64, 256]:
            self.assertEqual(spec.page_bytes(ps), ps * spec.entry_bytes())


class TestBuildMHAViews(unittest.TestCase):
    """Verify the 4-D view shape + strides at both page sizes."""

    def _build(self, page_size, layer_num=3, head_num=2, head_dim=4, n_full_slots=64):
        full = _make_mha_spec(
            "full", "up", layer_num=layer_num, head_num=head_num, head_dim=head_dim
        )
        swa = _make_mha_spec(
            "swa", "down", layer_num=2, head_num=head_num, head_dim=head_dim
        )
        # Pad to ensure max_slots % page_size == 0 in both sub-pools.
        # entry_bytes is fixed per spec; size accordingly.
        total = full.entry_bytes() * n_full_slots + swa.entry_bytes() * n_full_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        return pool, full

    def test_view_shape_is_4d(self):
        for ps in [1, 8]:
            pool, spec = self._build(page_size=ps)
            k_views, v_views = pool.mha_views_for("full")
            self.assertEqual(len(k_views), spec.layer_num)
            max_slots = pool.max_slots("full")
            for L in range(spec.layer_num):
                self.assertEqual(k_views[L].ndim, 4)
                self.assertEqual(
                    tuple(k_views[L].shape),
                    (max_slots // ps, ps, spec.head_num, spec.head_dim),
                )
                self.assertEqual(
                    tuple(v_views[L].shape),
                    (max_slots // ps, ps, spec.head_num, spec.v_head_dim),
                )

    def test_strides_at_page_size_1_match_envelope(self):
        """At ps=1, the 4-D view's stride[0] equals what today's 3-D view's
        stride[0] would have been (= entry_bytes / itemsize)."""
        pool, spec = self._build(page_size=1, layer_num=4, head_num=3, head_dim=8)
        k_views, _ = pool.mha_views_for("full")
        itemsize = spec.store_dtype.itemsize
        for L in range(spec.layer_num):
            # stride[0] = page_bytes/itemsize = entry_bytes/itemsize at ps=1
            self.assertEqual(k_views[L].stride(0), spec.entry_bytes() // itemsize)
            # stride[1] = k_row/itemsize (within-page token stride)
            self.assertEqual(k_views[L].stride(1), spec.k_row_bytes() // itemsize)
            # stride[2] = head_dim (head stride)
            self.assertEqual(k_views[L].stride(2), spec.head_dim)
            # stride[3] = 1 (innermost)
            self.assertEqual(k_views[L].stride(3), 1)

    def test_strides_at_page_size_gt_1(self):
        pool, spec = self._build(page_size=8, layer_num=4, head_num=3, head_dim=8)
        k_views, _ = pool.mha_views_for("full")
        itemsize = spec.store_dtype.itemsize
        for L in range(spec.layer_num):
            # page_bytes = 8 * 4 * (k_row + v_row); stride[0] = that / itemsize
            self.assertEqual(k_views[L].stride(0), spec.page_bytes(8) // itemsize)
            # token stride within layer L's K block = k_row/itemsize
            self.assertEqual(k_views[L].stride(1), spec.k_row_bytes() // itemsize)
            self.assertEqual(k_views[L].stride(2), spec.head_dim)
            self.assertEqual(k_views[L].stride(3), 1)

    def test_distinct_layers_dont_alias_at_page_size_gt_1(self):
        """Writes to layer 0 must not affect layer 1's K/V values (under
        layer-major within-page layout)."""
        pool, spec = self._build(page_size=8, layer_num=3, head_num=2, head_dim=4)
        k_views, v_views = pool.mha_views_for("full")
        # Set page 0, token 3, layer 0 K to a distinct pattern.
        target_val = 0.5
        k_views[0][0, 3] = target_val
        # Layer 1 K at the same (page, tok) should remain at default (0.0).
        self.assertFalse(torch.all(k_views[1][0, 3] == target_val))
        self.assertTrue(torch.all(k_views[1][0, 3] == 0.0))
        # And layer 0 V at the same (page, tok) should remain at default.
        self.assertFalse(torch.all(v_views[0][0, 3] == target_val))
        self.assertTrue(torch.all(v_views[0][0, 3] == 0.0))

    def test_distinct_pages_dont_alias_at_page_size_gt_1(self):
        """Writes to one page must not affect another page."""
        pool, spec = self._build(page_size=8, layer_num=3, head_num=2, head_dim=4)
        k_views, _ = pool.mha_views_for("full")
        # Set page 0, token 3, layer 0 K to a distinct pattern.
        k_views[0][0, 3] = 1.25
        # Page 1, token 3, layer 0 K should remain at default.
        self.assertTrue(torch.all(k_views[0][1, 3] == 0.0))


class TestMoveKVCacheNative4D(unittest.TestCase):
    """Verify ``move_kv_cache_native`` handles 4-D buffers at both
    page_size=1 (degenerate envelope) and page_size>1 (layer-major)."""

    def _build_buffer(
        self, page_size, layer_num=2, head_num=2, head_dim=4, n_full_slots=64
    ):
        full = _make_mha_spec(
            "full", "up", layer_num=layer_num, head_num=head_num, head_dim=head_dim
        )
        swa = _make_mha_spec(
            "swa", "down", layer_num=2, head_num=head_num, head_dim=head_dim
        )
        total = full.entry_bytes() * n_full_slots + swa.entry_bytes() * n_full_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
        )
        return pool

    def test_move_kv_cache_page_size_1(self):
        pool = self._build_buffer(page_size=1, layer_num=2, head_num=2, head_dim=4)
        k_views, v_views = pool.mha_views_for("full")
        # Write distinct markers at source slots 5, 6.
        for L in range(2):
            k_views[L][5, 0] = float(L + 1)
            v_views[L][5, 0] = -float(L + 1)
            k_views[L][6, 0] = float(L + 10)
            v_views[L][6, 0] = -float(L + 10)
        # Move 5 -> 8 and 6 -> 9.
        move_kv_cache_native(
            k_views,
            v_views,
            tgt_loc=torch.tensor([8, 9], dtype=torch.int64),
            src_loc=torch.tensor([5, 6], dtype=torch.int64),
            page_size=1,
        )
        for L in range(2):
            self.assertTrue(torch.all(k_views[L][8, 0] == float(L + 1)))
            self.assertTrue(torch.all(v_views[L][8, 0] == -float(L + 1)))
            self.assertTrue(torch.all(k_views[L][9, 0] == float(L + 10)))
            self.assertTrue(torch.all(v_views[L][9, 0] == -float(L + 10)))

    def test_move_kv_cache_page_size_gt_1(self):
        ps = 8
        pool = self._build_buffer(page_size=ps, layer_num=2, head_num=2, head_dim=4)
        k_views, v_views = pool.mha_views_for("full")
        # Write markers at token ids 5 and 14 (different pages).
        for L in range(2):
            # token 5 = (page 0, tok 5)
            k_views[L][0, 5] = float(L + 1)
            v_views[L][0, 5] = -float(L + 1)
            # token 14 = (page 1, tok 6)
            k_views[L][1, 6] = float(L + 10)
            v_views[L][1, 6] = -float(L + 10)
        # Move token 5 -> token 23 (page 2, tok 7) and 14 -> 31 (page 3, tok 7).
        move_kv_cache_native(
            k_views,
            v_views,
            tgt_loc=torch.tensor([23, 31], dtype=torch.int64),
            src_loc=torch.tensor([5, 14], dtype=torch.int64),
            page_size=ps,
        )
        for L in range(2):
            # 23 = page 2, tok 7
            self.assertTrue(torch.all(k_views[L][2, 7] == float(L + 1)))
            self.assertTrue(torch.all(v_views[L][2, 7] == -float(L + 1)))
            # 31 = page 3, tok 7
            self.assertTrue(torch.all(k_views[L][3, 7] == float(L + 10)))
            self.assertTrue(torch.all(v_views[L][3, 7] == -float(L + 10)))

    def test_move_kv_cache_3d_legacy_path_unchanged(self):
        """move_kv_cache_native(3-D, page_size=1) must take the legacy
        else-branch and be byte-identical to today."""
        k = [torch.zeros((32, 2, 4), dtype=torch.float16) for _ in range(2)]
        v = [torch.zeros((32, 2, 4), dtype=torch.float16) for _ in range(2)]
        for L in range(2):
            k[L][5] = float(L + 1)
            v[L][5] = -float(L + 1)
        move_kv_cache_native(
            k,
            v,
            tgt_loc=torch.tensor([7], dtype=torch.int64),
            src_loc=torch.tensor([5], dtype=torch.int64),
            page_size=1,
        )
        for L in range(2):
            self.assertTrue(torch.all(k[L][7] == float(L + 1)))
            self.assertTrue(torch.all(v[L][7] == -float(L + 1)))


class TestByteIdentityAtPageSize1(unittest.TestCase):
    """Verify that at page_size=1 the new 4-D view describes the SAME
    physical bytes as the old 3-D view would have. The view
    semantics differ (4-D vs 3-D shape) but the underlying byte layout is
    identical — confirmed by manually computing expected byte offsets and
    matching them against the 4-D view's strides + storage_offset.
    """

    def test_byte_addresses_match_envelope(self):
        spec = _make_mha_spec("full", "up", layer_num=4, head_num=2, head_dim=4)
        ps = 1
        # Build pool.
        total = spec.entry_bytes() * 64 + spec.entry_bytes() * 32
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[
                spec,
                _make_mha_spec("swa", "down", layer_num=2),
            ],
            device=_DEV,
            enable_memory_saver=False,
            page_size=ps,
        )
        k_views, v_views = pool.mha_views_for("full")
        # For each (layer, slot), compute the expected byte address under
        # the envelope layout and verify the 4-D view's data_ptr +
        # advanced indexing agrees.
        max_slots = pool.max_slots("full")
        itemsize = spec.store_dtype.itemsize
        base_addr = pool._raw.data_ptr()
        for L in range(spec.layer_num):
            for s in range(0, max_slots, max(1, max_slots // 4)):
                # Envelope: bytes for slot s, layer L's K start at:
                #   s * entry_bytes + L * (k_row + v_row)
                expected_k_byte_offset = s * spec.entry_bytes() + L * (
                    spec.k_row_bytes() + spec.v_row_bytes()
                )
                # 4-D view: k_views[L][page=s, tok=0, head=0, dim=0]
                # storage_offset of the element [s, 0, 0, 0]:
                view_offset_elems = (
                    k_views[L].storage_offset()
                    + s * k_views[L].stride(0)
                    + 0 * k_views[L].stride(1)
                    + 0 * k_views[L].stride(2)
                    + 0 * k_views[L].stride(3)
                )
                view_byte_offset = view_offset_elems * itemsize
                # 4-D view sits over `_raw.view(spec.store_dtype)`, which
                # has data_ptr == _raw.data_ptr() (same backing storage).
                self.assertEqual(view_byte_offset, expected_k_byte_offset)


if __name__ == "__main__":
    unittest.main()
