"""CPU correctness tests for the page-major layer-major envelope layout.

Covers the standalone view builders (no allocator / shared pool):

  - ``build_page_major_mha_views``: 4-D K/V views with correct addressing at
    page_size 1 (token-granularity envelope) and > 1 (layer-major within a page),
    and no aliasing across layers / slots.
  - ``build_page_major_mamba_views``: conv / temporal state views.
  - ``move_kv_cache_native`` 4-D branch: relocating token rows preserves data.

Runs on CPU — pure-torch advanced indexing, no Triton.

    python -m pytest test/registered/unit/mem_cache/test_page_major_layout.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
    mamba_entry_bytes,
    mha_entry_bytes,
)
from sglang.srt.mem_cache.memory_pool import move_kv_cache_native

_DEV = "cpu"
_DT = torch.float32


def _make_mha_views(layer_num, head_num, head_dim, v_head_dim, page_size, num_pages):
    entry = mha_entry_bytes(
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        itemsize=_DT.itemsize,
    )
    raw = torch.zeros(num_pages * page_size * entry, dtype=torch.uint8, device=_DEV)
    k, v = build_page_major_mha_views(
        raw,
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DT,
        page_size=page_size,
        num_pages=num_pages,
    )
    return raw, k, v


class TestPageMajorMHAViews(unittest.TestCase):
    def test_view_shapes(self):
        _, k, v = _make_mha_views(3, 2, 4, 4, page_size=2, num_pages=4)
        self.assertEqual(len(k), 3)
        for t in k:
            self.assertEqual(tuple(t.shape), (4, 2, 2, 4))
        for t in v:
            self.assertEqual(tuple(t.shape), (4, 2, 2, 4))

    def test_no_aliasing_ps1(self):
        # Every (layer, slot) cell must be independently addressable.
        layer_num, slots = 3, 5
        _, k, v = _make_mha_views(layer_num, 2, 4, 4, page_size=1, num_pages=slots)
        for L in range(layer_num):
            for s in range(slots):
                k[L][s, 0] = float(100 + L * 10 + s)
                v[L][s, 0] = float(200 + L * 10 + s)
        for L in range(layer_num):
            for s in range(slots):
                self.assertTrue(torch.all(k[L][s, 0] == float(100 + L * 10 + s)))
                self.assertTrue(torch.all(v[L][s, 0] == float(200 + L * 10 + s)))

    def test_page_slot_addressing_ps_gt1(self):
        # token id t -> page t // ps, slot t % ps; no aliasing across tokens.
        ps, pages = 2, 4
        total = ps * pages
        _, k, _ = _make_mha_views(2, 1, 2, 2, page_size=ps, num_pages=pages)
        for L in range(2):
            for t in range(total):
                k[L][t // ps, t % ps, 0] = float(1000 + L * 100 + t)
        for L in range(2):
            for t in range(total):
                self.assertEqual(
                    float(k[L][t // ps, t % ps, 0, 0].item()), 1000 + L * 100 + t
                )

    def test_asymmetric_v_head_dim(self):
        _, k, v = _make_mha_views(2, 2, 6, 4, page_size=1, num_pages=3)
        self.assertEqual(tuple(k[0].shape), (3, 1, 2, 6))
        self.assertEqual(tuple(v[0].shape), (3, 1, 2, 4))


class TestPageMajorMove(unittest.TestCase):
    def test_move_ps1(self):
        slots = 6
        _, k, v = _make_mha_views(2, 1, 4, 4, page_size=1, num_pages=slots)
        for L in range(2):
            for s in range(slots):
                k[L][s, 0] = float(s + 1)
                v[L][s, 0] = float(-(s + 1))
        tgt = torch.tensor([0, 1], dtype=torch.int64)
        src = torch.tensor([4, 5], dtype=torch.int64)
        move_kv_cache_native(k, v, tgt, src, page_size=1)
        for L in range(2):
            self.assertTrue(torch.all(k[L][0, 0] == 5.0))
            self.assertTrue(torch.all(k[L][1, 0] == 6.0))
            self.assertTrue(torch.all(v[L][0, 0] == -5.0))

    def test_move_ps_gt1(self):
        ps, pages = 2, 4
        total = ps * pages
        _, k, v = _make_mha_views(1, 1, 2, 2, page_size=ps, num_pages=pages)
        for t in range(total):
            k[0][t // ps, t % ps, 0] = float(t + 1)
        tgt = torch.tensor([0, 3], dtype=torch.int64)  # page0 slot0, page1 slot1
        src = torch.tensor([6, 7], dtype=torch.int64)  # page3 slot0, page3 slot1
        move_kv_cache_native(k, v, tgt, src, page_size=ps)
        self.assertEqual(float(k[0][0, 0, 0, 0].item()), 7.0)
        self.assertEqual(float(k[0][1, 1, 0, 0].item()), 8.0)


class TestMambaEnvelopeViews(unittest.TestCase):
    def test_conv_temporal_shapes_no_alias(self):
        layers, slots = 2, 4
        conv_shapes = [(2, 3)]
        temp_shape = (2, 2)
        conv_dt, temp_dt = torch.bfloat16, torch.float32
        entry = mamba_entry_bytes(
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dt,
            temporal_state_shape=temp_shape,
            temporal_dtype=temp_dt,
        )
        raw = torch.zeros(slots * entry, dtype=torch.uint8, device=_DEV)
        conv_views, temporal = build_page_major_mamba_views(
            raw,
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dt,
            temporal_state_shape=temp_shape,
            temporal_dtype=temp_dt,
            max_slots=slots,
        )
        self.assertEqual(tuple(conv_views[0].shape), (layers, slots, 2, 3))
        self.assertEqual(tuple(temporal.shape), (layers, slots, 2, 2))
        for L in range(layers):
            for s in range(slots):
                temporal[L, s] = float(s + L * 10 + 1)
        for L in range(layers):
            for s in range(slots):
                self.assertTrue(torch.all(temporal[L, s] == float(s + L * 10 + 1)))


if __name__ == "__main__":
    unittest.main()
