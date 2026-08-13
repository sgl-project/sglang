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
"""Dense MHA K/V views for the unified memory pool (uniform-row hybrid models).

Covers, CPU-only (pure torch — no GPU / Triton kernels):
  - `MHASubPoolSpec` dense surface: `is_uniform_row` is exactly
    `k_row_bytes == v_row_bytes`, and `dense_blocks_per_page == 2*L` refuses
    asymmetric specs;
  - `build_dense_mha_views` addressing: view_l[dense(t)] must land exactly at
    the page-major envelope byte offset the STRIDED builder assigns to the same
    (page, slot, layer, K|V) cell — the two builders are views over one truth;
  - K and V of one token share ONE dense id (per-layer origin shift does the
    disambiguation), with no aliasing across the 2*L overlapping views;
  - the missing-tail-pad and asymmetric-dims cases fail loud at construction.

Addressing law under test (the derived property everything else builds on):

    dense(t) = (t // ps) * (ps * 2L) + t % ps
    K of layer l at block 2l, V at block 2l+1, blocks are ps rows of
    head_num*head_dim elements — offsets identical to
    MHASubPoolSpec.layer_k/v_offset_in_page when rows are uniform.

    python -m pytest test/registered/unit/mem_cache/test_unified_mha_dense_views.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_dense_mha_views,
    build_page_major_mha_views,
    mha_entry_bytes,
)
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec

_DEV = "cpu"

# Small-but-nontrivial MHA geometry: L=2 layers, H=2 heads, D=4, so every byte
# offset is hand-checkable. blocks = 2L = 4 per page.
_L = 2
_H = 2
_D = 4
_ROW = _H * _D  # row elements
_DTYPE = torch.bfloat16
_ITEM = _DTYPE.itemsize
_BLOCKS = 2 * _L


def _mha_spec(head_dim=_D, v_head_dim=None, layer_num=_L, grow="down"):
    return MHASubPoolSpec(
        name="full",
        layer_num=layer_num,
        head_num=_H,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DTYPE,
        grow_direction=grow,
    )


def _dense(t, ps):
    return (t // ps) * (ps * _BLOCKS) + t % ps


def _make_raw(ps, num_pages, pad_pages=1):
    page_bytes = ps * _BLOCKS * _ROW * _ITEM
    raw = torch.zeros(
        (num_pages + pad_pages) * page_bytes, dtype=torch.uint8, device=_DEV
    )
    return raw


def _build_dense(raw, ps, num_pages, head_dim=_D, v_head_dim=_D, layer_num=_L):
    return build_dense_mha_views(
        raw,
        layer_num=layer_num,
        head_num=_H,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DTYPE,
        page_size=ps,
        num_pages=num_pages,
    )


class TestMHADenseSpecSurface(unittest.TestCase):
    def test_uniform_row_is_exactly_row_byte_equality(self):
        self.assertTrue(_mha_spec().is_uniform_row())
        # Asymmetric K/V (the MiMoV2 shape, scaled down) must NOT be uniform.
        self.assertFalse(_mha_spec(head_dim=6, v_head_dim=4).is_uniform_row())

    def test_dense_blocks_refuses_asymmetric(self):
        self.assertEqual(_mha_spec().dense_blocks_per_page(), _BLOCKS)
        with self.assertRaises(AssertionError):
            _mha_spec(head_dim=6, v_head_dim=4).dense_blocks_per_page()

    def test_spec_offsets_equal_dense_block_origins(self):
        """The spec's byte math and the view builder's origins are two
        independent derivations of the envelope; under uniform rows they must
        agree: layer_k_offset(l) == (2l)*ps*row, layer_v_offset(l) == (2l+1)*ps*row."""
        spec = _mha_spec()
        for ps in (1, 4):
            row = spec.k_row_bytes()
            for l in range(_L):
                self.assertEqual(spec.layer_k_offset_in_page(l, ps), (2 * l) * ps * row)
                self.assertEqual(
                    spec.layer_v_offset_in_page(l, ps), (2 * l + 1) * ps * row
                )

    def test_entry_bytes_matches_layout_helper(self):
        spec = _mha_spec()
        self.assertEqual(
            spec.entry_bytes(),
            mha_entry_bytes(
                layer_num=_L, head_num=_H, head_dim=_D, v_head_dim=_D, itemsize=_ITEM
            ),
        )


class TestDenseMHAViews(unittest.TestCase):
    def test_view_shapes_are_stock_mha(self):
        ps, num_pages = 4, 6
        k_views, v_views = _build_dense(_make_raw(ps, num_pages), ps, num_pages)
        n_dense = num_pages * _BLOCKS * ps
        self.assertEqual(len(k_views), _L)
        self.assertEqual(len(v_views), _L)
        for v in (*k_views, *v_views):
            # The stock MHATokenToKVPool per-layer signature: 3-D, packed rows.
            self.assertEqual(tuple(v.shape), (n_dense, _H, _D))
            self.assertEqual(v.stride(), (_ROW, _D, 1))

    def test_dense_addressing_matches_strided_builder(self):
        """Cross-readback: bytes written through the STRIDED views at
        (page, slot) must be read back through the DENSE views at dense(t),
        for both K and V of every layer — and vice versa. This pins that the
        two builders describe the same physical envelope."""
        for ps in (1, 4):
            num_pages = 5
            raw = _make_raw(ps, num_pages)
            sk, sv = build_page_major_mha_views(
                raw,
                layer_num=_L,
                head_num=_H,
                head_dim=_D,
                v_head_dim=_D,
                store_dtype=_DTYPE,
                page_size=ps,
                num_pages=num_pages,
            )
            dk, dv = _build_dense(raw, ps, num_pages)
            probes = [(0, 0, 0), (1, 1, ps - 1), (4, 0, ps // 2), (3, 1, 0)]
            # strided-write -> dense-read
            for p, l, s in probes:
                t = p * ps + s
                d = _dense(t, ps)
                sk[l][p, s] = float(p * 100 + l * 10 + s + 1)
                sv[l][p, s] = float(p * 100 + l * 10 + s + 2)
                self.assertTrue(
                    torch.all(dk[l][d] == float(p * 100 + l * 10 + s + 1)),
                    f"K (p={p}, l={l}, s={s}, ps={ps}) dense readback off-formula",
                )
                self.assertTrue(
                    torch.all(dv[l][d] == float(p * 100 + l * 10 + s + 2)),
                    f"V (p={p}, l={l}, s={s}, ps={ps}) dense readback off-formula",
                )
            # dense-write -> strided-read
            for p, l, s in probes:
                t = p * ps + s
                d = _dense(t, ps)
                dk[l][d] = float(p * 100 + l * 10 + s + 3)
                dv[l][d] = float(p * 100 + l * 10 + s + 4)
                self.assertTrue(
                    torch.all(sk[l][p, s] == float(p * 100 + l * 10 + s + 3))
                )
                self.assertTrue(
                    torch.all(sv[l][p, s] == float(p * 100 + l * 10 + s + 4))
                )

    def test_k_and_v_share_one_dense_id_without_aliasing(self):
        """One dense id, 2L distinct cells (K and V of every layer): writes
        through all 2L views at the SAME id must not clobber each other."""
        ps, num_pages = 4, 4
        dk, dv = _build_dense(_make_raw(ps, num_pages), ps, num_pages)
        t = 2 * ps + 1  # page 2, slot 1
        d = _dense(t, ps)
        for l in range(_L):
            dk[l][d] = float(2 * l + 1)
            dv[l][d] = float(2 * l + 2)
        for l in range(_L):
            self.assertTrue(torch.all(dk[l][d] == float(2 * l + 1)))
            self.assertTrue(torch.all(dv[l][d] == float(2 * l + 2)))

    def test_missing_tail_pad_fails_loud(self):
        ps, num_pages = 2, 4
        raw = _make_raw(ps, num_pages, pad_pages=0)
        with self.assertRaises(AssertionError):
            _build_dense(raw, ps, num_pages)

    def test_asymmetric_dims_rejected(self):
        ps, num_pages = 2, 4
        raw = _make_raw(ps, num_pages)
        with self.assertRaises(AssertionError):
            _build_dense(raw, ps, num_pages, head_dim=6, v_head_dim=4)


if __name__ == "__main__":
    unittest.main()
