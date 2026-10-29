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
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import memory_pool as mem_pool
from sglang.srt.mem_cache.kv.layout.mha import DraftMHAView, MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import (
    DraftStateView,
    MambaSubPoolSpec,
    StateView,
)
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.layout.token_major import align_entry_bytes
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.unified_draft_pool import (
    routed_draft_pool,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    arena_of,
    plan_stand_in,
    stand_in_views,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
        pool = arena_of(
            total_bytes=total,
            sub_pool_specs=[spec, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
            fused_draft=_placement(spec.draft_region),
        )
        hk, hv = pool.spec("full").views(pool)
        dk, dv = pool.spec("full").draft_views(pool)
        return spec, pool, (hk, hv), (dk, dv)

    def test_host_and_draft_writes_confine_to_their_parts(self):
        spec, pool, (hk, hv), (dk, dv) = self._build()
        layout = spec.layout()
        raw = pool.raw
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


class TestDraftPartAlignment(unittest.TestCase):
    def test_mixed_dtype_state_parts_are_aligned_and_move_with_the_entry(self):
        region = DraftStateRegion(
            lane_num=3,
            state=DraftStateGeometry(
                conv_state_shapes=((1, 3), (1, 1)),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(3,),
                temporal_dtype=torch.float32,
            ),
        )
        state = _mamba_spec(region)
        full = _host_spec()
        arena = arena_of(
            total_bytes=16384,
            sub_pool_specs=[state, full],
            device="cpu",
            enable_memory_saver=False,
            page_size=1,
            fused_draft=FusedDraftPlacement.from_counts(
                counts={"mamba": [3]}, regions={"mamba": region}
            ),
        )
        arena.raw.zero_()
        conv, temporal = state.draft_views(arena)
        for tensor in (*conv, temporal):
            self.assertEqual(tensor.data_ptr() % tensor.element_size(), 0)
            tensor[:, 4].fill_(7)
        arena.move_units("mamba", torch.tensor([6]), torch.tensor([4]))
        for tensor in (*conv, temporal):
            self.assertTrue(torch.all(tensor[:, 6] == 7))
        host_conv, host_temporal = state.views(arena)
        for tensor in (*host_conv, host_temporal):
            self.assertTrue(torch.all(tensor[:, 6] == 0))


class TestDraftMHAView(unittest.TestCase):
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
        return arena_of(
            total_bytes=total,
            sub_pool_specs=[spec, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
            fused_draft=_placement(spec.draft_region),
        )

    def _draft_pool(self, pool):
        return DraftMHAView(
            arena=pool,
            host_spec=pool.spec("full"),
            layer_lanes={0: 0},
            page_size=self.PS,
        )

    def test_probe_surface_and_view_binding(self):
        pool = self._pool()
        dp = self._draft_pool(pool)
        spec = pool.spec("full")
        self.assertTrue(dp.requires_physical_write_loc)
        self.assertEqual(len(dp.k_buffer), 1)
        self.assertEqual(dp.k_buffer[0].shape[1:], (1, 24))
        self.assertEqual(dp.v_buffer[0].shape[1:], (1, 8))
        # Same slot space as the host: one row per physical token, the slot
        # stride being the whole fused entry.
        n_rows = pool.num_entries("full") // self.PS * self.PS
        self.assertEqual(dp.k_buffer[0].shape[0], n_rows)
        self.assertEqual(dp.size, n_rows - self.PS)
        self.assertEqual(
            dp.k_buffer[0].stride(0) * dp.k_buffer[0].element_size(), spec.entry_bytes()
        )

    def test_host_page_move_carries_the_draft_bytes(self):
        # Compaction relocates whole page envelopes on the HOST pool: a draft
        # marker written in page A arrives at page B after
        # the arena moves host unit A to B, with no draft-side move.
        pool = self._pool()
        dp = self._draft_pool(pool)
        host = MHAView(
            arena=pool,
            spec=pool.spec("full"),
            page_size=self.PS,
            model_layer_ids=tuple(range(pool.spec("full").layer_num)),
        )
        src_page, dst_page = 3, 5
        src_t, dst_t = src_page * self.PS, dst_page * self.PS
        dp.k_buffer[0][src_t] = 7.0
        self.assertEqual(float(dp.k_buffer[0][dst_t].sum()), 0.0)

        pool.move_units("full", torch.tensor([dst_page]), torch.tensor([src_page]))
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
        pool = arena_of(
            total_bytes=self.PAGES * self.PS * (host.entry_bytes() + swa.entry_bytes()),
            sub_pool_specs=[host, swa],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
            fused_draft=_placement(region),
        )
        dp = DraftMHAView(
            arena=pool,
            host_spec=pool.spec("full"),
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

    def _pool(self):
        region = _state_region()
        mamba = _mamba_spec(region)
        full = _host_spec()
        total = 4 * full.entry_bytes() + self.SLOTS * mamba.entry_bytes()
        pool = arena_of(
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
        host_conv, host_temporal = pool.spec("mamba").views(pool)
        draft_conv, draft_temporal = pool.spec("mamba").draft_views(pool)
        raw = pool.raw
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
        host = StateView(
            arena=pool,
            spec=pool.spec("mamba"),
            mamba_layer_ids=[0, 1],
        )
        draft_conv, _ = pool.spec("mamba").draft_views(pool)
        host.mamba_cache.conv[0][1][3] = 2.0
        draft_conv[1][0][3] = 7.0
        host.copy_from(torch.tensor([3]), torch.tensor([5]))
        self.assertEqual(float(host.mamba_cache.conv[0][1][5].sum()), 2.0 * 3 * 8)
        self.assertEqual(float(draft_conv[1][0][5].sum()), 7.0 * 2 * 4)
        host.clear_slots(torch.tensor([5]))
        self.assertEqual(float(draft_conv[1][0][5].sum()), 0.0)
        self.assertEqual(float(host.mamba_cache.conv[0][1][5].sum()), 0.0)
        pool.move_units("mamba", torch.tensor([6]), torch.tensor([3]))
        self.assertEqual(float(host.mamba_cache.conv[0][1][6].sum()), 2.0 * 3 * 8)
        self.assertEqual(float(draft_conv[1][0][6].sum()), 7.0 * 2 * 4)

    def test_draft_views_alias_the_fused_block(self):
        _, pool = self._pool()
        draft = DraftStateView(arena=pool, spec=pool.spec("mamba"), layer_lanes={4: 0})
        draft_conv, _ = pool.spec("mamba").draft_views(pool)
        self.assertEqual(
            draft.mamba_cache.conv[1].shape, (1, pool.num_entries("mamba"), 2, 4)
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
        draft = DraftStateView(arena=pool, spec=pool.spec("mamba"), layer_lanes={4: 0})
        host = StateView(
            arena=pool,
            spec=pool.spec("mamba"),
            mamba_layer_ids=[0, 1],
        )
        draft_conv, draft_temporal = pool.spec("mamba").draft_views(pool)
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
        # `RunnerIds.__init__` reads `attn_dcp_size`, a derived
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
        return arena_of(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
            page_size=self.PS,
            fused_draft=_placement(full.draft_region),
        )

    def test_host_and_draft_writes_stay_byte_disjoint_within_a_slot(self):
        pool = self._pool()
        full = pool.spec("full")
        host_views = pool.spec("full").views(pool)
        dk, dv = pool.spec("full").draft_views(pool)
        raw = pool.raw
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
        from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds

        pool = self._pool()

        class _FakeKV:
            def attach_allocator(self, allocator):
                self.allocator = allocator

        kvcache = SimpleNamespace(full_kv_pool=_FakeKV(), mamba_pool=_FakeKV())
        alloc = ChainAllocator(
            roles=ChainRoles(owner="full", state="mamba"),
            arena=pool,
            kvcache=kvcache,
            views=stand_in_views(
                ChainRoles(owner="full", state="mamba"), kvcache, kvcache.mamba_pool
            ),
            device=_DEV,
            page_size=self.PS,
            need_sort=False,
            forward_stream=None,
            lazy_compaction=False,
        )
        dp = DraftMHAView(
            arena=pool,
            host_spec=pool.spec("full"),
            layer_lanes={0: 0},
            page_size=self.PS,
        )
        runner_ids = RunnerIds(
            req_to_token=torch.zeros((2, 8), dtype=torch.int32),
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=dp,
            page_size=self.PS,
            device=_DEV,
        )
        self.assertTrue(runner_ids.is_translating)
        self.assertIs(
            runner_ids.full_v2p_table, alloc.full_attn_allocator.ids.virtual_to_physical
        )

    def test_draft_pool_binds_over_the_mla_host(self):
        pool = self._pool()
        dp = DraftMHAView(
            arena=pool,
            host_spec=pool.spec("full"),
            layer_lanes={0: 0},
            page_size=self.PS,
        )
        entry = pool.spec("full").entry_bytes()
        self.assertEqual(dp.k_buffer[0].shape[1:], (1, 24))
        self.assertEqual(dp.v_buffer[0].shape[1:], (1, 8))
        self.assertEqual(
            dp.k_buffer[0].stride(0) * dp.k_buffer[0].element_size(), entry
        )


class TestRoutedDraftPool(CustomTestCase):
    """A draft with window layers binds one dense view per host sub-pool and
    routes per layer like the target's pool: a window layer's write needs the
    swa loc and lands in the swa entry's draft part, never in the full entry."""

    PS = 2
    PAGES = 8

    def _pool(self, *, symmetric=False):
        full_region = (
            DenseDraftRegion(lane_num=1, head_num=1, head_dim=8, store_dtype=_DTYPE)
            if symmetric
            else _draft_region()
        )
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
        return arena_of(
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
        return routed_draft_pool(
            arena=pool,
            plan=plan_stand_in(pool),
            owner="full",
            page_size=self.PS,
            lanes_by_host={"full": {0: 0}, "swa": {1: 0}},
        )

    def test_routes_each_layer_to_its_side(self):
        pool = self._pool()
        dp = self._draft_pool(pool)
        self.assertEqual(dp.layer_router.route(0).sub_pool, "full")
        self.assertEqual(dp.layer_router.route(1).sub_pool, "swa")
        self.assertEqual(dp.layer_router.route(1).index, 0)
        dk, _ = pool.spec("swa").draft_views(pool)
        self.assertEqual(dp.get_key_buffer(1).data_ptr(), dk[0].data_ptr())
        self.assertEqual(dp.get_key_buffer(1).shape[1:], (1, 8))
        self.assertEqual(dp.get_key_buffer(0).shape[1:], (1, 24))

    def test_a_window_only_draft_answers_its_own_v_width(self):
        # MiMoV2MTP: no full layer at all; the swa view answers for the pool.
        pool = self._pool()
        swa_only = routed_draft_pool(
            arena=pool,
            plan=plan_stand_in(pool),
            owner="full",
            page_size=self.PS,
            lanes_by_host={"swa": {0: 0}},
        )
        self.assertIsNone(swa_only.layer_router.sub_pool("full"))
        self.assertEqual(swa_only.get_v_head_dim(), 8)
        self.assertEqual(swa_only.layer_router.route(0).sub_pool, "swa")

    def test_window_write_needs_the_swa_loc_and_stays_in_the_swa_entry(self):
        pool = self._pool()
        dp = self._draft_pool(pool)
        layer = SimpleNamespace(layer_id=1)
        k = torch.full((1, 1, 8), 3.0, dtype=_DTYPE)
        v = torch.full((1, 1, 8), 5.0, dtype=_DTYPE)
        loc = torch.tensor([6], dtype=torch.int64)
        with self.assertRaises(AssertionError):
            dp.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
        raw = pool.raw
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

    def _write_fixture(self, *, symmetric=False):
        arena = self._pool(symmetric=symmetric)
        pool = self._draft_pool(arena)
        full = torch.tensor([[14, 15, 16], [17, 18, 19]], dtype=torch.int64)
        swa = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.int64)
        values = []
        for layer_id, name in enumerate(("full", "swa")):
            region = arena.spec(name).draft_region
            k = (
                torch.arange(6 * region.head_dim, dtype=_DTYPE).view(
                    6, 1, region.head_dim
                )
                + 1
                + layer_id * 64
            )
            v = (
                -torch.arange(6 * region.resolved_v_head_dim(), dtype=_DTYPE).view(
                    6, 1, region.resolved_v_head_dim()
                )
                - 1
                - layer_id * 64
            )
            values.append((k, v))
        return arena, pool, full, swa, values

    def _expected_writes(self, arena, full, swa, values, commit_lens):
        arena.raw.fill_(0x5A)
        expected = arena.raw.clone()
        rows = (
            range(6)
            if commit_lens is None
            else (
                row * 3 + col
                for row, length in enumerate(commit_lens.tolist())
                for col in range(length)
            )
        )
        rows = tuple(rows)
        for name, loc, (k, v) in zip(("full", "swa"), (full, swa), values):
            key_views, value_views = arena.spec(name).draft_views(arena)
            for source, buffer in ((k, key_views[0]), (v, value_views[0])):
                base = buffer.data_ptr() - arena.raw.data_ptr()
                stride = buffer.stride(0) * buffer.element_size()
                for row in rows:
                    value = source[row].contiguous().view(torch.uint8).flatten()
                    offset = base + int(loc.flatten()[row]) * stride
                    expected[offset : offset + value.numel()] = value
        return expected

    def test_layer_stores_keep_the_full_and_window_id_spaces(self):
        """Routed draft writes must use each host's physical IDs and leave rejected rows intact."""
        arena, pool, full, swa, values = self._write_fixture()
        for lengths in (None, (2, 0), (0, 0)):
            with self.subTest(commit_lens=lengths):
                commit_lens = (
                    torch.tensor(lengths, dtype=torch.int32)
                    if lengths is not None
                    else None
                )
                expected = self._expected_writes(arena, full, swa, values, commit_lens)
                loc = KVWriteLoc(
                    full if commit_lens is not None else full.flatten(),
                    swa_loc=swa if commit_lens is not None else swa.flatten(),
                    physical=True,
                )
                with patch.multiple(
                    mem_pool, _is_cuda=False, _is_hip=False, _cpu_has_amx_support=False
                ):
                    for layer_id, (k, v) in enumerate(values):
                        layer = SimpleNamespace(layer_id=layer_id)
                        access = pool.layer_router.layer(layer_id)
                        if commit_lens is None:
                            access.store(layer, loc, k, v)
                        else:
                            access.store_prefix(layer, loc, commit_lens, k, v)
                self.assertTrue(torch.equal(arena.raw, expected))

    def _draft_layers(self, values):
        return [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    attn=SimpleNamespace(layer_id=i, k_scale=None, v_scale=None),
                    num_kv_heads=1,
                    head_dim=8,
                    kv_proj_only=lambda hidden, k=k, v=v: (k, v),
                    apply_k_norm=lambda k: k,
                    apply_k_rope=lambda positions, k: k,
                )
            )
            for i, (k, v) in enumerate(values)
        ]

    def test_dflash_context_append_routes_normal_and_prefix_writes(self):
        """Both DFLASH materializers must retain SWA IDs through accepted-prefix commits."""
        from sglang.srt.speculative import dflash_worker_v2
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        arena, pool, full, swa, values = self._write_fixture(symmetric=True)
        worker = object.__new__(DFlashWorkerV2)
        worker.model_runner = SimpleNamespace(device=torch.device("cpu"))
        worker.draft_model_runner = SimpleNamespace(token_to_kv_pool=pool)
        worker.draft_parallel_layout = None
        worker.lilicorr = None
        worker.draft_model = SimpleNamespace(
            layers=self._draft_layers(values),
            project_target_hidden=lambda hidden: hidden,
            prepare_context_hidden_for_kv=lambda layer, hidden: hidden,
        )

        class Materializer:
            def materialize(self, *, write_layer_kv, **kwargs):
                for layer_id, (k, v) in enumerate(values):
                    write_layer_kv(layer_id, k, v)

        for fused in (False, True):
            for lengths in (None, (2, 0), (0, 0)):
                with self.subTest(fused=fused, commit_lens=lengths):
                    worker._use_fused_kv_materialize = fused
                    worker._fused_kv_helper = Materializer() if fused else None
                    commit_lens = (
                        torch.tensor(lengths, dtype=torch.int32)
                        if lengths is not None
                        else None
                    )
                    expected = self._expected_writes(
                        arena, full, swa, values, commit_lens
                    )
                    with (
                        patch.multiple(
                            mem_pool,
                            _is_cuda=False,
                            _is_hip=False,
                            _cpu_has_amx_support=False,
                        ),
                        patch.object(dflash_worker_v2, "_is_npu", False),
                        patch.object(
                            dflash_worker_v2,
                            "draft_tp_context",
                            return_value=nullcontext(),
                        ),
                    ):
                        worker._append_target_hidden_to_draft_kv_by_loc(
                            target_hidden=torch.ones((6, 8), dtype=_DTYPE),
                            cache_loc=full.flatten(),
                            swa_cache_loc=swa.flatten(),
                            positions=torch.arange(6),
                            cache_loc_2d=full if commit_lens is not None else None,
                            commit_lens=commit_lens,
                        )
                    self.assertTrue(torch.equal(arena.raw, expected))
                    self.assertEqual(worker._use_fused_kv_materialize, fused)

    def test_dspark_context_injector_uses_both_plan_id_spaces(self):
        """DSpark's projected context must commit into both routed hosts without touching rejected rows."""
        from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
        from sglang.srt.mem_cache.kv.alloc.iteration_ids import IterationIds
        from sglang.srt.models.dspark import DSparkDraftMixin
        from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
            TargetHiddenKvInjector,
        )

        arena, pool, full, swa, values = self._write_fixture(symmetric=True)
        spaces = {
            kind: IdSpace(key=kind, write=lambda ids, loc=loc: loc.flatten()[ids])
            for kind, loc in (
                (IdSpaceKind.FULL, full),
                (IdSpaceKind.SLIDING_WINDOW, swa),
            )
        }
        runner_ids = SimpleNamespace(space=spaces.get)
        plan = IterationIds(
            source=runner_ids,
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([3, 3]),
            seq_lens_cpu=torch.tensor([3, 3]),
            write_virtual=torch.arange(6),
        )
        model = SimpleNamespace(
            layers=self._draft_layers(values),
            project_target_hidden=lambda hidden: hidden,
            _fused_kv_write_bundle=lambda pool: None,
            _stacked_ctx_kv_params=lambda: None,
        )
        model.write_target_hidden_kv = MethodType(
            DSparkDraftMixin.write_target_hidden_kv, model
        )
        injector = TargetHiddenKvInjector(
            draft_model=model,
            draft_model_runner=SimpleNamespace(
                token_to_kv_pool=pool, runner_ids=runner_ids
            ),
            model_runner=SimpleNamespace(device=torch.device("cpu")),
            device=torch.device("cpu"),
            verify_num_draft_tokens=3,
            block_pos_offsets=torch.arange(3),
        )
        for lengths in (None, (2, 0), (0, 0)):
            with self.subTest(commit_lens=lengths):
                commit_lens = (
                    torch.tensor(lengths, dtype=torch.int32)
                    if lengths is not None
                    else None
                )
                expected = self._expected_writes(arena, full, swa, values, commit_lens)
                with patch.multiple(
                    mem_pool, _is_cuda=False, _is_hip=False, _cpu_has_amx_support=False
                ):
                    injector.inject_target_hidden(
                        target_hidden=torch.ones((6, 8), dtype=_DTYPE),
                        cache_loc=plan.write_ids(runner_ids),
                        positions=torch.arange(6),
                        cache_loc_2d=full if commit_lens is not None else None,
                        commit_lens=commit_lens,
                        iteration_ids=plan,
                    )
                self.assertTrue(torch.equal(arena.raw, expected))

    def test_dspark_single_location_fusion_declines_window_layers(self):
        """A shared-location kernel cannot address full and SWA hosts with different IDs."""
        from sglang.srt.models.dspark import DSparkDraftMixin

        arena, pool, _, _, values = self._write_fixture(symmetric=True)
        layers = self._draft_layers(values)
        rotary = type("RotaryEmbedding", (), {})()
        rotary.position_start = 0
        rotary.rotary_dim = 8
        rotary.is_neox_style = True
        rotary.cos_sin_cache = torch.ones((16, 8), dtype=_DTYPE)
        for layer in layers:
            layer.self_attn.kv_size = 8
            layer.self_attn.rotary_emb = rotary
            layer.self_attn.k_norm = SimpleNamespace(
                weight=torch.ones(8, dtype=_DTYPE), variance_epsilon=1e-6
            )
        model = SimpleNamespace(layers=layers)
        model._fused_k_norm_weights = MethodType(
            DSparkDraftMixin._fused_k_norm_weights, model
        )
        self.assertIsNone(
            DSparkDraftMixin._fused_kv_write_layout(model, pool, project=lambda x: x)
        )
        model.layers = layers[:1]
        self.assertIsNotNone(
            DSparkDraftMixin._fused_kv_write_layout(model, pool, project=lambda x: x)
        )


if __name__ == "__main__":
    unittest.main()
