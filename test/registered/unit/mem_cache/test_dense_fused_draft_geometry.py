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
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.layout.page_major import (
    ENTRY_ALIGN_BYTES,
    ROW_ALIGN_BYTES,
    align_entry_bytes,
    align_part_offset,
)
from sglang.srt.mem_cache.unified_draft_pool import UnifiedDraftKVPool
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
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


def _placement(region):
    return FusedDraftPlacement(region=region, runner_lane_counts=(region.lane_num,))


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

    def test_only_page_envelope_kinds_accept_a_draft_region(self):
        """`draft_region` is a universal spec field (None = unfused), but a
        kind without the fused entry layout must refuse one at construction;
        a silently-carried region would never reach the layout. MHA and MLA
        entries carry the draft parts; mamba state pages carry none yet."""
        with self.assertRaises(AssertionError):
            MambaSubPoolSpec(
                name="mamba",
                layer_num=1,
                conv_state_shapes=((2, 2),),
                conv_dtype=torch.float32,
                temporal_state_shape=(2,),
                temporal_dtype=torch.float32,
                grow_direction="up",
                draft_region=_draft_region(),
            )


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
            fused_draft=_placement(spec.draft_region),
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
            fused_draft=_placement(spec.draft_region),
        )

    def _draft_pool(self, pool):
        sentinel_allocator = object()
        dp = UnifiedDraftKVPool(
            unified_buffer=pool,
            host_sub_pool_name="full",
            host_allocator=sentinel_allocator,
            layer_lanes={0: 0},
            page_size=self.PS,
        )
        return dp, sentinel_allocator

    def test_probe_surface_and_view_binding(self):
        pool = self._pool()
        dp, alloc = self._draft_pool(pool)
        spec = pool.mha_spec("full")
        self.assertIs(dp.host_allocator, alloc)
        self.assertTrue(dp.requires_translated_write_loc)
        self.assertEqual(len(dp.k_buffer), 1)
        self.assertEqual(dp.k_buffer[0].shape[1:], (1, 24))
        self.assertEqual(dp.v_buffer[0].shape[1:], (1, 8))
        # Same slot space as the host: one row per physical token, the slot
        # stride being the whole fused entry.
        n_rows = pool.max_slots("full") // self.PS * self.PS
        self.assertEqual(dp.k_buffer[0].shape[0], n_rows)
        self.assertEqual(dp.size, n_rows - self.PS)
        self.assertEqual(
            dp.k_buffer[0].stride(0) * dp.k_buffer[0].element_size(), spec.entry_bytes()
        )

    def test_host_page_move_carries_the_draft_bytes(self):
        # THE fused-layout property: compaction relocates whole page envelopes
        # on the HOST pool; a draft marker written in page A must arrive at
        # page B after host.move_kv_cache(B, A), with zero draft-side moves.
        pool = self._pool()
        dp, _ = self._draft_pool(pool)
        host = UnifiedMHATokenToKVPool(
            unified_buffer=pool, sub_pool_name="full", page_size=self.PS
        )
        src_page, dst_page = 3, 5
        src_t, dst_t = src_page * self.PS, dst_page * self.PS
        dp.k_buffer[0][src_t] = 7.0
        self.assertEqual(float(dp.k_buffer[0][dst_t].sum()), 0.0)

        ps = self.PS
        to_tokens = lambda p: torch.arange(p * ps, (p + 1) * ps, dtype=torch.int64)
        host.move_kv_cache(to_tokens(dst_page), to_tokens(src_page))
        self.assertEqual(float(dp.k_buffer[0][dst_t].sum()), 7.0 * 24)

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


def _mla_host_spec(draft_region=None):
    # 32 B latent rows, two layers: a 64 B host entry before the draft parts.
    return MLASubPoolSpec(
        name="full",
        layer_num=2,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        store_dtype=torch.bfloat16,
        grow_direction="down",
        draft_region=draft_region,
    )


def _mamba_spec():
    return MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        conv_state_shapes=((2, 2),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2,),
        temporal_dtype=torch.float32,
        grow_direction="up",
    )


class TestFusedMLAHost(unittest.TestCase):
    """MLA entries carry the fused draft parts exactly like MHA entries: the
    same aligned entry, one slot stride for both families, and byte-disjoint
    host/draft parts within every slot. The unfused spec stays byte-identical;
    that identity guards every existing MLA deploy."""

    def setUp(self):
        # `KVIndexTranslator.__init__` reads `attn_dcp_size`, a derived
        # parallel width that only exists once a config is published.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    PS = 2
    PAGES = 8

    def test_unfused_spec_is_byte_identical_to_before(self):
        s = _mla_host_spec()
        self.assertEqual(s.row_bytes(), 32)
        self.assertEqual(s.entry_bytes(), 64)
        self.assertEqual(s.host_entry_bytes(), s.entry_bytes())
        self.assertEqual([p.name for p in s.layout().parts], ["kv"])

    def test_fused_entry_appends_the_draft_parts(self):
        f = _mla_host_spec(_draft_region())
        self.assertEqual(f.host_entry_bytes(), 64)
        self.assertEqual(f.draft_offset_in_entry(), 64)
        self.assertEqual(
            f.entry_bytes(), align_entry_bytes(64 + f.draft_region.entry_bytes())
        )
        self.assertEqual(f.entry_bytes(), 128)
        layout = f.layout()
        self.assertEqual([p.name for p in layout.parts], ["kv", "draft_k", "draft_v"])
        self.assertEqual(layout.part("draft_k").offset_bytes, 64)
        self.assertEqual(layout.part("draft_v").offset_bytes, 64 + 48)

    def _pool(self):
        full = _mla_host_spec(_draft_region())
        mamba = _mamba_spec()
        total = self.PAGES * self.PS * full.entry_bytes() + 4 * mamba.entry_bytes()
        return UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
            fused_draft=_placement(full.draft_region),
        )

    def test_host_and_draft_writes_stay_byte_disjoint_within_a_slot(self):
        pool = self._pool()
        full = pool.mla_spec("full")
        host_views = pool.mla_views_for("full")
        dk, dv = pool.build_dense_draft_views("full")
        raw = pool._raw
        entry = full.entry_bytes()
        t = 2 * self.PS + 1  # page 2, slot 1
        slot_lo = t * entry
        split = slot_lo + full.draft_offset_in_entry()
        slot_hi = slot_lo + entry
        for layer_view in host_views:
            raw.zero_()
            layer_view[t] = 1.0
            nz = raw.nonzero()
            self.assertGreater(nz.numel(), 0)
            self.assertTrue(bool((nz >= slot_lo).all() and (nz < split).all()))
        for family in (dk, dv):
            for layer_view in family:
                raw.zero_()
                layer_view[t] = 1.0
                nz = raw.nonzero()
                self.assertGreater(nz.numel(), 0)
                self.assertTrue(bool((nz >= split).all() and (nz < slot_hi).all()))

    def test_translator_takes_the_fused_disposition_on_the_mla_host(self):
        """A draft runner bound over the MLA host's entries must translate
        through the mamba allocator's full-side v2p, exactly as on the SWA
        host. A kind-specific probe regression here silently reverts the
        draft to passthrough (raw virtual ids into the views)."""
        from sglang.srt.mem_cache.allocator.unified_mamba import (
            UnifiedMambaTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator

        pool = self._pool()

        class _FakeKV:
            def attach_allocator(self, allocator):
                self.allocator = allocator

        kvcache = SimpleNamespace(full_kv_pool=_FakeKV(), mamba_pool=_FakeKV())
        alloc = UnifiedMambaTokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            page_size=self.PS,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=False,
        )
        dp = UnifiedDraftKVPool(
            unified_buffer=pool,
            host_sub_pool_name="full",
            host_allocator=alloc,
            layer_lanes={0: 0},
            page_size=self.PS,
        )
        translator = KVIndexTranslator(
            req_to_token=torch.zeros((2, 8), dtype=torch.int32),
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=dp,
            page_size=self.PS,
            device=_DEV,
        )
        self.assertTrue(translator.is_translating)
        self.assertIs(
            translator.full_flat_v2p(), alloc.full_attn_allocator.virtual_to_physical
        )

    def test_draft_pool_binds_over_the_mla_host(self):
        pool = self._pool()
        dp = UnifiedDraftKVPool(
            unified_buffer=pool,
            host_sub_pool_name="full",
            host_allocator=object(),
            layer_lanes={0: 0},
            page_size=self.PS,
        )
        entry = pool.mla_spec("full").entry_bytes()
        self.assertEqual(dp.k_buffer[0].shape[1:], (1, 24))
        self.assertEqual(dp.v_buffer[0].shape[1:], (1, 8))
        self.assertEqual(
            dp.k_buffer[0].stride(0) * dp.k_buffer[0].element_size(), entry
        )


if __name__ == "__main__":
    unittest.main()
