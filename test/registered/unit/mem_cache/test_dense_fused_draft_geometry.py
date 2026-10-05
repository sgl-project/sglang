"""Dense fused-draft geometry: the draft model's KV as two more parts of every
host slot's entry,

    [ K_0 | V_0 | ... | K_{Lh-1} | V_{Lh-1} | dK_0 | dV_0 | ... | pad ]

the draft parts starting at the aligned end of the host parts. Host and draft
views share the slot stride and are indexed by the same physical token id;
writes through either family's views land inside their own part of their own
slot, and a host page move carries the draft bytes with it.

    python -m pytest test/registered/unit/mem_cache/test_dense_fused_draft_geometry.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import memory_pool as mem_pool
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.layout.token_major import (
    ENTRY_ALIGN_BYTES,
    align_entry_bytes,
    align_part_offset,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.unified_draft_pool import (
    UnifiedDraftKVPool,
    UnifiedDraftMambaPool,
    UnifiedDraftSWAKVPool,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMHATokenToKVPool,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

_DEV = "cpu"
_DTYPE = torch.bfloat16


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


def _state_region(lane_num=1):
    return DraftStateRegion(
        lane_num=lane_num,
        state=DraftStateGeometry(
            conv_state_shapes=((2, 4), (2, 4)),
            conv_dtype=torch.float32,
            temporal_state_shape=(2,),
            temporal_dtype=torch.float32,
        ),
    )


def _mamba_spec(draft_region=None, layer_num=2):
    return MambaSubPoolSpec(
        name="mamba",
        layer_num=layer_num,
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2,),
        temporal_dtype=torch.float32,
        grow_direction="up",
        draft_region=draft_region,
    )


def _placement(region, num_runners=1):
    return FusedDraftPlacement.from_counts(
        counts={"full": [region.lane_num] * num_runners}, regions={"full": region}
    )


class TestFusedSpecMath(unittest.TestCase):
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
        self.assertTrue(dp.requires_physical_write_loc)
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
        # Compaction relocates whole page envelopes on the HOST pool: a draft
        # marker written in page A arrives at page B after
        # host.move_kv_cache(B, A), with no draft-side move.
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

    def test_an_fp8_region_holds_fp8(self):
        """Under an fp8 KV cache the fused rows are STORED as uint8 but hold
        fp8: the pool's `dtype` is what `set_kv_buffer` casts to before viewing
        the result as `store_dtype`, and what a read views the bytes back as.
        A pool built on the storage dtype turns every write into an integer
        cast -- silent garbage draft KV."""
        fp8 = torch.float8_e4m3fn
        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=16, store_dtype=torch.uint8, kv_dtype=fp8
        )
        host = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=8,
            store_dtype=torch.uint8,
            grow_direction="down",
            draft_region=region,
        )
        swa = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=8,
            store_dtype=torch.uint8,
            grow_direction="up",
        )
        pool = UnifiedKVPool(
            total_bytes=self.PAGES * self.PS * (host.entry_bytes() + swa.entry_bytes()),
            sub_pool_specs=[host, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
            fused_draft=_placement(region),
        )
        dp = UnifiedDraftKVPool(
            unified_buffer=pool,
            host_sub_pool_name="full",
            host_allocator=object(),
            layer_lanes={0: 0},
            page_size=self.PS,
        )
        self.assertEqual((dp.dtype, dp.store_dtype), (fp8, torch.uint8))
        # The fp8 bytes a write stores read back as the same fp8 values.
        loc = torch.tensor([3, 4], dtype=torch.int64)
        k = torch.tensor([0.5, -1.0, 2.0, 0.25] * 8).view(2, 1, 16)
        dp.k_buffer[0][loc] = k.to(fp8).view(torch.uint8)
        self.assertEqual(dp.get_key_buffer(0).dtype, fp8)
        torch.testing.assert_close(dp.get_key_buffer(0)[loc].float(), k)


class TestFusedStateHost(unittest.TestCase):
    """A state (mamba) entry carries the draft's state block after the host's
    streams: the same aligned entry for both, byte-disjoint blocks within a
    slot, and the host's whole-entry copy/clear carry the draft block, so a
    radix checkpoint restores the draft's state with the target's. Unfused,
    the entry stays its raw size; that identity guards every state deploy."""

    SLOTS = 8

    def test_unfused_spec_is_byte_identical_to_before(self):
        s = _mamba_spec()
        self.assertEqual(s.host_entry_bytes(), 2 * (3 * 8 * 2 + 2 * 4))
        self.assertEqual(s.entry_bytes(), s.host_entry_bytes())

    def test_fused_entry_appends_the_state_block(self):
        f = _mamba_spec(_state_region())
        host = f.host_entry_bytes()
        self.assertEqual(f.draft_offset_in_entry(), align_part_offset(host))
        self.assertEqual(
            f.entry_bytes(),
            align_entry_bytes(f.draft_offset_in_entry() + f.draft_region.entry_bytes()),
        )
        self.assertEqual(f.draft_region.entry_bytes(), 2 * (2 * 4 * 4) + 2 * 4)

    def _pool(self):
        region = _state_region()
        mamba = _mamba_spec(region)
        full = _host_spec()
        total = 4 * full.entry_bytes() + self.SLOTS * mamba.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=1,
            fused_draft=FusedDraftPlacement.from_counts(
                counts={"mamba": [1]}, regions={"mamba": region}
            ),
        )
        return mamba, pool

    def test_host_and_draft_blocks_stay_byte_disjoint_within_a_slot(self):
        spec, pool = self._pool()
        host_conv, host_temporal = pool.mamba_views_for("mamba")
        draft_conv, draft_temporal = pool.build_draft_state_views("mamba")
        raw = pool._raw
        entry = spec.entry_bytes()
        slot = 3
        lo = slot * entry
        split = lo + spec.draft_offset_in_entry()
        hi = (slot + 1) * entry
        for view in (*host_conv, host_temporal):
            for layer in range(view.shape[0]):
                raw.zero_()
                view[layer][slot] = 1.0
                nz = raw.nonzero()
                self.assertGreater(nz.numel(), 0)
                self.assertTrue(bool((nz >= lo).all() and (nz < split).all()))
        for view in (*draft_conv, draft_temporal):
            for layer in range(view.shape[0]):
                raw.zero_()
                view[layer][slot] = 1.0
                nz = raw.nonzero()
                self.assertGreater(nz.numel(), 0)
                self.assertTrue(bool((nz >= split).all() and (nz < hi).all()))

    def test_host_whole_entry_ops_carry_the_draft_block(self):
        _, pool = self._pool()
        host = UnifiedMambaPool(
            unified_buffer=pool,
            sub_pool_name="mamba",
            spec_state_size=2,
            mamba_layer_ids=[0, 1],
        )
        draft_conv, _ = pool.build_draft_state_views("mamba")
        host.mamba_cache.conv[0][1][3] = 2.0
        draft_conv[1][0][3] = 7.0
        host.copy_from(torch.tensor([3]), torch.tensor([5]))
        self.assertEqual(float(host.mamba_cache.conv[0][1][5].sum()), 2.0 * 3 * 8)
        self.assertEqual(float(draft_conv[1][0][5].sum()), 7.0 * 2 * 4)
        host.clear_slots(torch.tensor([5]))
        self.assertEqual(float(draft_conv[1][0][5].sum()), 0.0)
        self.assertEqual(float(host.mamba_cache.conv[0][1][5].sum()), 0.0)
        host.move_kv_cache(torch.tensor([6]), torch.tensor([3]))
        self.assertEqual(float(host.mamba_cache.conv[0][1][6].sum()), 2.0 * 3 * 8)
        self.assertEqual(float(draft_conv[1][0][6].sum()), 7.0 * 2 * 4)

    def test_draft_views_alias_the_fused_block(self):
        _, pool = self._pool()
        draft = UnifiedDraftMambaPool(
            unified_buffer=pool, sub_pool_name="mamba", layer_lanes={4: 0}
        )
        draft_conv, _ = pool.build_draft_state_views("mamba")
        self.assertEqual(
            draft.mamba_cache.conv[1].shape, (1, pool.max_slots("mamba"), 2, 4)
        )
        self.assertEqual(
            draft.mamba2_layer_cache(0).conv[1].data_ptr(),
            draft_conv[1][0].data_ptr(),
        )

    def test_slot_ops_clear_the_draft_block_and_spare_the_host(self):
        """BUG REGRESSION. These ops were no-ops, on the theory that the host's
        whole-entry clear covers the draft block. It does, but ModelRunner gates
        that to a plain EXTEND forward and returns on decode, target-verify and
        draft-extend -- exactly the modes a draft needs. A slot handed to a new
        request therefore kept its previous occupant's state, and a slot never
        written kept whatever the buffer held; a poisoned pool turned that into
        NaN and the accept length decayed over a long run (an Inkling MTP head).
        Equally, they must NOT clear the whole entry: the host's own streams
        share it and stay live while the draft's block is recycled."""
        spec, pool = self._pool()
        draft = UnifiedDraftMambaPool(
            unified_buffer=pool, sub_pool_name="mamba", layer_lanes={4: 0}
        )
        host = UnifiedMambaPool(
            unified_buffer=pool,
            sub_pool_name="mamba",
            spec_state_size=2,
            mamba_layer_ids=[0, 1],
        )
        draft_conv, draft_temporal = pool.build_draft_state_views("mamba")
        host.mamba_cache.conv[0][1][2] = 5.0  # host state, same slot
        draft_conv[0][0][2] = 9.0
        draft_temporal[0][2] = 7.0

        draft.clear_slots(torch.tensor([2]))
        self.assertEqual(float(draft_conv[0][0][2].sum()), 0.0)
        self.assertEqual(float(draft_temporal[0][2].sum()), 0.0)
        # The host's stream in the SAME slot must survive.
        self.assertEqual(float(host.mamba_cache.conv[0][1][2].sum()), 5.0 * 3 * 8)

        draft_conv[0][0][2] = 4.0
        draft.copy_from(torch.tensor([2]), torch.tensor([6]))
        self.assertEqual(float(draft_conv[0][0][6].sum()), 4.0 * 2 * 4)
        self.assertEqual(float(host.mamba_cache.conv[0][1][6].sum()), 0.0)


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


def _mamba_filler_spec():
    """A minimal second sub-pool so the MLA host is not alone in the pool.

    Deliberately not `_mamba_spec`: this one is a bystander with its own tiny
    geometry, while `_mamba_spec` is the subject of the fused-state tests. Two
    module-level defs of one name silently leave only the last, which is how
    every `_mamba_spec(region)` call started raising TypeError.
    """
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
    host/draft parts within every slot."""

    def setUp(self):
        # `KVIndexTranslator.__init__` reads `attn_dcp_size`, a derived
        # parallel width that only exists once a config is published.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    PS = 2
    PAGES = 8

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
        mamba = _mamba_filler_spec()
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
        """A draft runner bound over the MLA host's entries translates through
        the mamba allocator's full-side v2p, as on the SWA host; a passthrough
        here would write raw virtual ids into the views."""
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
            translator.full_v2p_table, alloc.full_attn_allocator.virtual_to_physical
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


class TestUnifiedDraftSWAKVPool(unittest.TestCase):
    """A draft with window layers binds one dense side per host sub-pool and
    routes per layer like the target's composite: a window layer's write
    needs the swa loc and lands in the swa entry's draft part, never in the
    full entry."""

    PS = 2
    PAGES = 8

    def _pool(self):
        full_region = _draft_region()
        swa_region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=_DTYPE
        )
        full = _host_spec(full_region)
        swa = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=4,
            store_dtype=_DTYPE,
            grow_direction="up",
            draft_region=swa_region,
        )
        total = self.PAGES * self.PS * (full.entry_bytes() + swa.entry_bytes())
        return UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
            fused_draft=FusedDraftPlacement.from_counts(
                counts={"full": [1], "swa": [1]},
                regions={"full": full_region, "swa": swa_region},
            ),
        )

    def _draft_pool(self, pool):
        return UnifiedDraftSWAKVPool(
            unified_buffer=pool,
            host_allocator=object(),
            page_size=self.PS,
            full_layer_lanes={0: 0},
            swa_layer_lanes={1: 0},
        )

    def test_routes_each_layer_to_its_side(self):
        pool = self._pool()
        dp = self._draft_pool(pool)
        self.assertEqual(dp.layers_mapping, {0: (0, False), 1: (0, True)})
        dk, _ = pool.build_dense_draft_views("swa")
        self.assertEqual(dp.get_key_buffer(1).data_ptr(), dk[0].data_ptr())
        self.assertEqual(dp.get_key_buffer(1).shape[1:], (1, 8))
        self.assertEqual(dp.get_key_buffer(0).shape[1:], (1, 24))
        self.assertEqual(dp.swa_layer_nums, 1)
        self.assertEqual(dp.full_layer_nums, 1)

    def test_a_window_only_draft_answers_its_own_v_width(self):
        # MiMoV2MTP: no full layer at all; the composite still works.
        pool = self._pool()
        swa_only = UnifiedDraftSWAKVPool(
            unified_buffer=pool,
            host_allocator=object(),
            page_size=self.PS,
            full_layer_lanes={},
            swa_layer_lanes={0: 0},
        )
        self.assertIsNone(swa_only.full_kv_pool)
        self.assertEqual(swa_only.get_v_head_dim(), 8)
        self.assertEqual(swa_only.layers_mapping, {0: (0, True)})

    def test_window_write_needs_the_swa_loc_and_stays_in_the_swa_entry(self):
        pool = self._pool()
        dp = self._draft_pool(pool)
        layer = SimpleNamespace(layer_id=1)
        k = torch.full((1, 1, 8), 3.0, dtype=_DTYPE)
        v = torch.full((1, 1, 8), 5.0, dtype=_DTYPE)
        loc = torch.tensor([6], dtype=torch.int64)
        with self.assertRaises(AssertionError):
            dp.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
        raw = pool._raw
        raw.zero_()
        # `_is_cuda` is a PLATFORM constant, not a per-tensor device check, so
        # on a CUDA box the store dispatches the CUDA-only `sglang::store_cache`
        # at these CPU tensors and raises NotImplementedError. This fixture is
        # CPU by contract (register_cpu_ci), and the subject here is WHERE the
        # bytes land, not which kernel puts them there -- take the naive path.
        with patch.object(mem_pool, "can_use_store_cache", return_value=False):
            dp.set_kv_buffer(layer, KVWriteLoc(loc, swa_loc=loc, physical=True), k, v)
        swa_entry = pool.spec("swa").entry_bytes()
        lo = 6 * swa_entry + pool.spec("swa").draft_offset_in_entry()
        nz = raw.nonzero()
        self.assertGreater(nz.numel(), 0)
        self.assertTrue(bool((nz >= lo).all() and (nz < 7 * swa_entry).all()))


if __name__ == "__main__":
    unittest.main()
