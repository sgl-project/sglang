"""Dense fused-draft geometry: the draft model's KV as two more parts of every
host slot's entry.

Derived-property pins for the fused entry layout

    [ K_0 | V_0 | ... | K_{Lh-1} | V_{Lh-1} | dK_0 | dV_0 | ... | pad ]

the draft parts start at the 16-B-aligned end of the host parts and the entry
rounds up to the 32-B entry alignment (never to an lcm of row widths), so the
draft's row width is free to differ from the host's; host and draft views
share the slot stride and are indexed by the same physical token id; and
writes through either family's views land inside their own part of their own
slot (compaction moves whole page envelopes, so confinement IS the correctness
of the fused move).

    python -m pytest test/registered/unit/mem_cache/test_dense_fused_draft_geometry.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.layout.fused_draft import DenseDraftRegion
from sglang.srt.mem_cache.layout.page_major import (
    ENTRY_ALIGN_BYTES,
    ROW_ALIGN_BYTES,
    align_entry_bytes,
    align_part_offset,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"
_DTYPE = torch.bfloat16
_ITEM = _DTYPE.itemsize


def _host_spec(draft_region=None, layer_num=2, head_num=2, head_dim=4):
    return MHASubPoolSpec(
        name="full",
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=_DTYPE,
        grow_direction="down",
        draft_region=draft_region,
    )


def _draft_region():
    # Deliberately a different row width from the host's (16 B): K rows of
    # 48 B, V rows of 16 B -- asymmetric draft rows are ordinary parts.
    return DenseDraftRegion(
        lane_num=1, head_num=1, head_dim=24, v_head_dim=8, store_dtype=_DTYPE
    )


class TestFusedSpecMath(unittest.TestCase):
    def test_unfused_spec_is_byte_identical_to_before(self):
        s = _host_spec()
        self.assertEqual(s.entry_bytes(), 2 * (16 + 16))
        self.assertEqual(s.host_entry_bytes(), s.entry_bytes())
        self.assertEqual([p.name for p in s.layout().parts], ["k", "v"])

    def test_fused_entry_appends_the_draft_parts(self):
        f = _host_spec(_draft_region())
        r = f.draft_region
        self.assertEqual(r.k_row_bytes(), 48)
        self.assertEqual(r.v_row_bytes(), 16)
        self.assertEqual(r.entry_bytes(), 64)
        self.assertEqual(
            f.draft_offset_in_entry(), align_part_offset(f.host_entry_bytes())
        )
        self.assertEqual(f.draft_offset_in_entry(), 64)
        self.assertEqual(f.entry_bytes(), align_entry_bytes(64 + 64))
        self.assertEqual(f.entry_bytes() % ENTRY_ALIGN_BYTES, 0)
        layout = f.layout()
        self.assertEqual(
            [p.name for p in layout.parts], ["k", "v", "draft_k", "draft_v"]
        )
        self.assertEqual(layout.part("draft_k").offset_bytes, 64)
        self.assertEqual(layout.part("draft_v").offset_bytes, 64 + 48)
        self.assertEqual(layout.part("draft_k").layer_stride_bytes, 64)

    def test_entry_alignment_pads_only_to_the_entry_quantum(self):
        # host 32 B + draft 48 B = 80 B of rows -> one 96 B entry (16 B pad),
        # not an lcm of the two row widths.
        f = _host_spec(
            DenseDraftRegion(
                lane_num=1, head_num=1, head_dim=8, v_head_dim=16, store_dtype=_DTYPE
            ),
            layer_num=1,
            head_num=1,
            head_dim=8,
        )
        self.assertEqual(f.host_entry_bytes(), 32)
        self.assertEqual(f.draft_region.entry_bytes(), 48)
        self.assertEqual(f.entry_bytes(), 96)
        self.assertEqual(f.draft_offset_in_entry() % ROW_ALIGN_BYTES, 0)
        f.layout()  # parts fit and do not overlap

    def test_draft_rows_must_be_row_aligned(self):
        with self.assertRaises(AssertionError):
            _host_spec(
                DenseDraftRegion(lane_num=1, head_num=1, head_dim=3, store_dtype=_DTYPE)
            ).layout()


class TestFusedRegionConfinement(unittest.TestCase):
    """Writes through host/draft views must land in their own part of their
    own slot."""

    PS = 2
    PAGES = 3

    def _build(self):
        spec = _host_spec(_draft_region())
        swa = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=4,
            store_dtype=_DTYPE,
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
        hk, hv = pool.mha_views_for("full")
        dk, dv = pool.build_dense_draft_views("full")
        return spec, pool, (hk, hv), (dk, dv)

    def test_host_and_draft_writes_confine_to_their_parts(self):
        spec, pool, (hk, hv), (dk, dv) = self._build()
        layout = spec.layout()
        raw = pool._raw
        entry = spec.entry_bytes()
        families = (
            ("k", hk),
            ("v", hv),
            ("draft_k", dk),
            ("draft_v", dv),
        )
        for t in range(self.PAGES * self.PS):
            for name, views in families:
                part = layout.part(name)
                for layer, view in enumerate(views):
                    raw.zero_()
                    view[t] = 1.0
                    nz = raw.nonzero().flatten()
                    lo = t * entry + part.layer_offset_bytes(layer)
                    hi = lo + part.row_bytes()
                    self.assertEqual(int(nz.min()), lo, f"{name}[{layer}] t={t}")
                    self.assertEqual(int(nz.max()), hi - 1, f"{name}[{layer}] t={t}")
                    # ... and inside the slot's page envelope.
                    page = t // self.PS
                    self.assertGreaterEqual(lo, page * self.PS * entry)
                    self.assertLessEqual(hi, (page + 1) * self.PS * entry)

    def test_host_and_draft_views_share_the_slot_stride(self):
        spec, _, (hk, hv), (dk, dv) = self._build()
        stride = spec.entry_bytes()
        for view in (*hk, *hv, *dk, *dv):
            self.assertEqual(view.stride(0) * view.element_size(), stride)
            self.assertEqual(view.shape[0], hk[0].shape[0])
        self.assertEqual(tuple(dk[0].shape[1:]), (1, 24))
        self.assertEqual(tuple(dv[0].shape[1:]), (1, 8))

    def test_pool_helper_refuses_an_unfused_sub_pool(self):
        _, pool, _, _ = self._build()
        with self.assertRaises(AssertionError):
            pool.build_dense_draft_views("swa")  # no fused region there


if __name__ == "__main__":
    unittest.main()
