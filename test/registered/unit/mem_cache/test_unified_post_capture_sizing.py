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
"""Post-capture KV sizing on the unified hybrid-SWA byte pool.

With post-capture sizing the shared arena is reserved as CUDA virtual memory
before graph capture: only the slot-0 sink is physically backed, the graphs
capture the final tensor address, and ``finalize_backing`` backs the measured
byte budget in place. ``ChainAllocator.resize`` then
re-derives every capacity the sub-allocators and sub-pools expose.

The oracle is the eagerly-allocated pool of the same final size: a resized pool
must hand out exactly the capacity it would have had if it had been built at
that size, without moving the tensor the graphs captured.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.layout.mha import DraftMHAView
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftBand,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
    setup_allocator_context,
)

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")

_RESERVED_BYTES = 64 << 20
_FINAL_BYTES = 16 << 20
_PAGE_SIZE = 4


def _bundle(
    *,
    total_bytes: int,
    post_capture_active: bool,
    chain="swa",
    fused_draft=None,
    head_dim=8,
):
    has_state = chain in ("state", "tri")
    has_window = chain in ("swa", "tri")
    return build_test_unified_pools(
        device="cuda",
        kv_cache_dtype=torch.float16,
        head_num=2,
        head_dim=head_dim,
        v_head_dim=head_dim,
        swa_head_num=2,
        swa_head_dim=head_dim,
        swa_v_head_dim=head_dim,
        page_size=_PAGE_SIZE,
        start_layer=0,
        end_layer=4,
        swa_attention_layer_ids=[1, 3] if has_window else [],
        mamba_layer_ids=[1, 3] if has_state else [],
        mamba2_cache_params=SimpleNamespace(
            layers=[1, 3],
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(1, 4, 8)),
            dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
        )
        if has_state
        else None,
        max_mamba_cache_size=16 if has_state else 0,
        speculative_num_draft_tokens=2 if has_state else None,
        full_attention_layer_ids=[0, 2],
        total_bytes=total_bytes,
        fused_draft=fused_draft,
        enable_memory_saver=False,
        post_capture_active=post_capture_active,
        need_sort=False,
        model_context_len=256,
        sliding_window_size=64,
    )


def _final_config(bundle):
    storage = bundle.token_to_kv_pool_allocator.storage
    sized = size_plan(
        storage.plan,
        KVSizingInputs(
            budget_bytes=_FINAL_BYTES,
            context_len=256,
            floor_window=64,
            max_running_requests=1,
            state_slots=16,
            alignment=1,
        ),
        frozen=storage.sized,
    )
    return SimpleNamespace(
        unified_memory_pool_bytes=sized.total_bytes,
        unified_full_cap_tokens=sized.full_cap_tokens,
        full_max_total_num_tokens=sized.full_tokens,
        swa_max_total_num_tokens=sized.window_tokens,
    )


def _capacities(bundle) -> dict:
    """Every capacity-derived value a resize must keep consistent."""
    shared = bundle.arena
    kvcache = bundle.token_to_kv_pool
    allocator = bundle.token_to_kv_pool_allocator
    out = {
        "total_bytes": shared.total_bytes,
        "active_allocation_bytes": shared.accessible_bytes,
        "buf_lens": kvcache.get_contiguous_buf_infos()[1],
        "kvcache.size": kvcache.size,
        "allocator.size": allocator.size,
        "allocator.available_size": allocator.available_size(),
        "allocator.full_available_size": allocator.full_available_size(),
        "allocator.swa_available_size": allocator.swa_available_size(),
    }
    for name, sub, pool in (
        ("full", allocator.full_attn_allocator, kvcache.layer_router.view("full")),
        ("swa", allocator.swa_attn_allocator, kvcache.layer_router.view("swa")),
    ):
        out[f"{name}.max_slots"] = shared.num_entries(name)
        out[f"{name}.sub.max_slots"] = sub.max_slots
        out[f"{name}.sub.num_pages"] = sub.num_pages
        out[f"{name}.sub.ids.num_virtual_ids"] = sub.ids.num_virtual_ids
        out[f"{name}.sub.available_size"] = sub.available_size()
        out[f"{name}.pool.size"] = pool.size
        out[f"{name}.pool.host_capacity_tokens"] = pool.host_capacity_tokens
        out[f"{name}.pool.host_capacity_bytes"] = pool.host_capacity_bytes
    return out


def _mla_draft_bundle(
    dcp_size, *, latent_dim=8, rope_dim=8, total_bytes=1 << 20, draft_layers=1
):
    from transformers import PretrainedConfig

    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.mem_cache.layout.fused_draft import (
        PlacementContext,
        draft_kv_profile,
        place_fused_draft,
    )

    mc = SimpleNamespace(
        hf_config=PretrainedConfig(architectures=["BailingMoeForCausalLMNextN"]),
        is_hybrid_swa=False,
        num_nextn_predict_layers=None,
        attention_arch=AttentionArch.MLA,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        linear_attn_registry_result=None,
    )
    placement = place_fused_draft(
        profile=draft_kv_profile(
            mc, num_layers=draft_layers, attn_tp_size=1
        ),
        num_runners=1,
        ctx=PlacementContext(host_names=("full", "mamba"), dcp_size=dcp_size),
        store_dtype=torch.float16,
    ).placement
    return build_test_unified_pools(
        device="cuda",
        kv_cache_dtype=torch.float16,
        page_size=64,
        start_layer=0,
        end_layer=2,
        full_attention_layer_ids=[0],
        head_num=1,
        head_dim=8,
        use_mla_backend=True,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        mamba_layer_ids=[1],
        max_mamba_cache_size=16,
        mamba2_cache_params=SimpleNamespace(
            layers=[1],
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(1, 4, 8)),
            dtype=SimpleNamespace(conv=torch.float16, temporal=torch.float32),
        ),
        total_bytes=total_bytes,
        max_num_reqs=2,
        model_context_len=512,
        fused_draft=placement,
        lazy_compaction=False,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestUnifiedPostCaptureSizing(CustomTestCase):
    def test_mla_window_context_writes_use_the_window_physical_rows(self):
        """Full and window MLA lanes use independent physical locations on replay."""
        from sglang.srt.mem_cache.layout.fused_draft import DraftBand, MLADraftRegion
        from sglang.srt.mem_cache.unified_draft_pool import routed_draft_pool
        from sglang.srt.models.dflash import DFlashMLAAttention

        region = MLADraftRegion(
            lane_num=1, kv_lora_rank=8, qk_rope_head_dim=8, store_dtype=torch.float16
        )
        bundle = _bundle(
            total_bytes=1 << 20,
            post_capture_active=False,
            fused_draft=FusedDraftPlacement.from_counts(
                counts={"full": [1], "draft_swa": [1]},
                regions={"full": region, "draft_swa": region},
                new_bands={"draft_swa": DraftBand(kind="window", window=32)},
            ),
        )
        a = bundle.token_to_kv_pool_allocator
        virtual = a.alloc(16)
        self.assertIsNotNone(virtual)
        full_ids = a.full_attn_allocator.ids.translate_write(virtual).long()
        window_ids = (
            a.window_allocators["draft_swa"].ids.translate_write(virtual).long()
        )
        self.assertFalse(torch.equal(full_ids, window_ids))
        pool = routed_draft_pool(
            arena=bundle.arena,
            plan=bundle.storage.plan,
            owner="full",
            page_size=4,
            lanes_by_host={"full": {0: 0}, "draft_swa": {1: 0}},
        )
        bundle.arena.raw.zero_()
        full_entries = bundle.arena.entries("full")
        full_entries[full_ids] = 23
        full_before = full_entries[full_ids].clone()
        window = pool.get_key_buffer(1)
        loc = full_ids[:6].reshape(2, 3).clone()
        swa_loc = window_ids[:6].reshape(2, 3).clone()
        lens = torch.tensor([2, 1], device="cuda")
        nope = torch.arange(48, device="cuda", dtype=torch.float16).reshape(6, 1, 8)
        rope = nope + 64
        attn = SimpleNamespace(
            attn_mqa=SimpleNamespace(layer_id=1),
            project_context_kv=lambda *args: (nope, rope),
        )
        for prefix in (False, True):

            def write():
                DFlashMLAAttention.write_context_kv(
                    attn,
                    pool=pool,
                    positions=loc.flatten(),
                    hidden_states=nope[:, 0],
                    cache_loc=loc.flatten(),
                    swa_cache_loc=swa_loc.flatten(),
                    cache_loc_2d=loc if prefix else None,
                    commit_lens=lens if prefix else None,
                )

            window[window_ids] = -7
            write()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                write()
            for replay in (False, True):
                if replay:
                    loc.copy_(full_ids[4:10].reshape(2, 3))
                    swa_loc.copy_(window_ids[4:10].reshape(2, 3))
                    lens.copy_(torch.tensor([0, 3], device="cuda"))
                    nope.add_(1)
                    window[window_ids] = -7
                    graph.replay()
                expected = torch.full_like(window[window_ids], -7)
                mask = torch.ones((2, 3), dtype=torch.bool, device="cuda")
                if prefix:
                    mask = torch.arange(3, device="cuda") < lens[:, None]
                positions = torch.searchsorted(window_ids, swa_loc[mask])
                expected[positions] = torch.cat((nope, rope), dim=-1)[mask.flatten()]
                torch.testing.assert_close(window[window_ids], expected)
                torch.testing.assert_close(full_entries[full_ids], full_before)
            loc.copy_(full_ids[:6].reshape(2, 3))
            swa_loc.copy_(window_ids[:6].reshape(2, 3))
            lens.copy_(torch.tensor([2, 1], device="cuda"))
        a.free(virtual)
        self.assertEqual(a.verify_byte_accounting(), [])

    def test_dflash_mla_prefix_store_preserves_masked_rows_on_graph_replay(self):
        """Accepted prefixes reach their draft lane without touching rejected rows."""
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.mem_cache.unified_draft_pool import routed_draft_pool
        from sglang.srt.models.dflash import DFlashMLAAttention

        bundle = _mla_draft_bundle(1, draft_layers=2)
        routed = routed_draft_pool(
            arena=bundle.arena,
            plan=bundle.storage.plan,
            owner="full",
            page_size=64,
            lanes_by_host={"full": {0: 1, 1: 0}},
        )
        plain = MLATokenToKVPool(
            size=256,
            page_size=64,
            dtype=torch.float16,
            kv_lora_rank=8,
            qk_rope_head_dim=8,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        target = bundle.token_to_kv_pool_allocator.get_kvcache().get_key_buffer(0)
        target.fill_(23)
        for pool in (plain, routed):
            with self.subTest(pool=type(pool).__name__):
                loc = torch.tensor([[64, 65, 66], [128, 129, 130]], device="cuda")
                lens = torch.tensor([2, 1], device="cuda")
                nope = torch.arange(48, device="cuda", dtype=torch.float16).reshape(
                    6, 1, 8
                )
                rope = nope + 64
                attn = SimpleNamespace(
                    attn_mqa=SimpleNamespace(layer_id=1),
                    project_context_kv=lambda *args: (nope, rope),
                )

                def write():
                    DFlashMLAAttention.write_context_kv(
                        attn,
                        pool=pool,
                        positions=loc.flatten(),
                        hidden_states=nope[:, 0],
                        cache_loc=loc.flatten(),
                        cache_loc_2d=loc,
                        commit_lens=lens,
                    )

                buffers = [pool.get_key_buffer(layer) for layer in range(2)]
                for buf in buffers:
                    buf.fill_(-7)
                write()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    write()
                for replay in (False, True):
                    if replay:
                        loc.add_(4)
                        lens.copy_(torch.tensor([0, 3], device="cuda"))
                        nope.add_(1)
                        for buf in buffers:
                            buf.fill_(-7)
                        graph.replay()
                    expected = torch.full_like(buffers[1], -7)
                    mask = torch.arange(3, device="cuda") < lens[:, None]
                    expected[loc[mask]] = torch.cat((nope, rope), dim=-1)[
                        mask.flatten()
                    ]
                    torch.testing.assert_close(buffers[1], expected)
                    self.assertTrue(torch.all(buffers[0] == -7))
                    self.assertTrue(torch.all(target == 23))

    def setUp(self):
        setup_allocator_context()

    def test_host_retraction_restores_every_retained_window(self):
        """Host retraction must preserve both window payloads at new device slots."""
        from unittest.mock import patch

        from sglang.srt.managers.schedule_batch import ReqKvInfo
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.runtime_context import get_memory
        from sglang.srt.server_args import ServerArgs

        for draft_window in (32, 96):
            with self.subTest(draft_window=draft_window):
                placement = FusedDraftPlacement.from_counts(
                    counts={"draft_swa": [1]},
                    regions={
                        "draft_swa": DenseDraftRegion(
                            lane_num=1,
                            head_num=1,
                            head_dim=64,
                            store_dtype=torch.float16,
                        )
                    },
                    new_bands={
                        "draft_swa": DraftBand(kind="window", window=draft_window)
                    },
                )
                bundle = _bundle(
                    total_bytes=1 << 20,
                    post_capture_active=False,
                    chain="swa",
                    fused_draft=placement,
                    head_dim=64,
                )
                a = bundle.token_to_kv_pool_allocator
                params = CacheInitParams(
                    disable=False,
                    req_to_token_pool=bundle.storage.requests,
                    token_to_kv_pool_allocator=a,
                    page_size=4,
                    sliding_window_size=64,
                    tree_slots=bundle.storage.tree_slots(),
                    tree_core_backend="python",
                )
                cache = UnifiedRadixCache(params)
                with (
                    get_memory().override(
                        hicache_ratio=1.0,
                        hicache_size=0,
                        hicache_mem_layout="layer_first",
                    ),
                    patch("torch.distributed.get_world_size", return_value=1),
                ):
                    cache.init_hicache(ServerArgs(model_path="dummy"), params)
                self.addCleanup(cache.release_host_resources)
                req = SimpleNamespace(rid="window-backup", seqlen=130, kv=ReqKvInfo())
                req.kv.req_pool_idx = 0
                source = a.alloc(132)
                self.assertIsNotNone(source)
                bundle.storage.requests.req_to_token[0, :132] = source.int()
                # Give each band recognizable bytes, including the padded last page.
                expected = {}
                for i, band in enumerate(a.members):
                    rows = band.ids.translate(source)
                    a.arena.entries(band.sub_pool_name)[rows] = i + 11
                    expected[band.sub_pool_name] = i + 11
                backup = cache.backup_kv_cache(req)
                self.assertIsNotNone(backup)
                self.assertEqual(
                    {t.name for t in backup.pool_transfers},
                    {PoolName.SWA, PoolName.AUXILIARY_SWA},
                )
                a.free(source)
                blocker = a.alloc(132)
                destination = a.alloc(132)
                self.assertIsNotNone(destination)
                bundle.storage.requests.req_to_token[0, :132] = destination.int()
                for band in a.members:
                    rows = band.ids.translate(destination)
                    a.arena.entries(band.sub_pool_name)[rows] = 0
                cache.restore_kv_cache(req, backup)
                cache._finish_restore(req.kv)
                for band in a.members:
                    window = {"full": 132, "swa": 64, "draft_swa": draft_window}[
                        band.sub_pool_name
                    ]
                    start = max(0, 129 - window) // 4 * 4
                    rows = band.ids.translate(destination[start:132])
                    self.assertTrue(
                        torch.all(
                            a.arena.entries(band.sub_pool_name)[rows]
                            == expected[band.sub_pool_name]
                        )
                    )
                    if start:
                        discarded = band.ids.translate(destination[:start])
                        self.assertTrue(
                            torch.all(
                                a.arena.entries(band.sub_pool_name)[discarded] == 0
                            )
                        )
                a.free(blocker)
                a.free(destination)
                self.assertEqual(bundle.storage.accounting_violations(), [])

    def test_auxiliary_host_pool_attaches_to_its_own_cache_component(self):
        """A separately retained draft window must survive final HiCache attachment."""
        from unittest.mock import patch

        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as host
        from sglang.srt.mem_cache.unified_cache.components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.runtime_context import get_memory

        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=torch.float16
        )
        placement = FusedDraftPlacement.from_counts(
            counts={"draft_swa": [1]},
            regions={"draft_swa": region},
            new_bands={"draft_swa": DraftBand(kind="window", window=32)},
        )
        bundle = _bundle(
            total_bytes=1 << 20,
            post_capture_active=False,
            chain="swa",
            fused_draft=placement,
        )
        a = bundle.token_to_kv_pool_allocator
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=bundle.storage.requests,
            token_to_kv_pool_allocator=a,
            page_size=4,
            sliding_window_size=64,
            tree_slots=bundle.storage.tree_slots(),
            tree_core_backend="python",
        )
        cache = UnifiedRadixCache(params)
        # The IO controller starts background transfer workers; keep the real
        # device/host layouts, component tree and final attachment in this test.
        controller = SimpleNamespace(layer_done_counter=None, transfer_layer_id_max=4)
        with (
            get_memory().override(
                hicache_ratio=1.0, hicache_size=0, hicache_mem_layout="layer_first"
            ),
            patch.object(host, "HybridCacheController", return_value=controller),
        ):
            host.attach_hybrid_pool_to_unified_cache(
                cache, params, None, load_cache_event=None
            )
        pools = (
            cache.full_kv_pool_host,
            cache.swa_kv_pool_host,
            cache.auxiliary_swa_kv_pool_host,
        )
        for pool in pools:
            self.addCleanup(pool.destroy)
        self.assertIsNot(pools[1], pools[2])
        self.assertIs(cache.components[ComponentType.SWA]._swa_kv_pool_host, pools[1])
        self.assertIs(
            cache.components[ComponentType.AUXILIARY_SWA]._swa_kv_pool_host, pools[2]
        )
        ids = pools[2].alloc(4)
        self.assertIsNotNone(ids)
        pools[2].free(ids)

    def _post_capture_bundle(self, total_bytes=_RESERVED_BYTES):
        return _bundle(total_bytes=total_bytes, post_capture_active=True)

    def test_auto_state_shrink_preserves_captured_buffers(self):
        from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
        from sglang.srt.mem_cache.kv.decl import (
            MHAGeometry,
            StateGeometry,
            kv_layer_decls,
        )
        from sglang.srt.mem_cache.kv.plan import plan_kv_storage

        setup_allocator_context()
        plan = plan_kv_storage(
            kv_layer_decls(
                full_layer_ids=(0,),
                full_geometry=MHAGeometry(
                    head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.float16
                ),
                state_layer_ids=(1,),
                state_geometry=StateGeometry(
                    conv_state_shapes=((8, 8),),
                    conv_dtype=torch.float16,
                    temporal_state_shape=(1, 4, 8),
                    temporal_dtype=torch.float32,
                ),
            ),
            page_size=4,
            speculative_num_draft_tokens=3,
        )
        captured = size_plan(
            plan,
            KVSizingInputs(
                budget_bytes=1 << 20,
                max_running_requests=10,
                radix_cache=False,
                context_len=16,
                alignment=1,
            ),
        )
        storage = build_kv_storage(
            plan,
            captured,
            UnifiedBuildContext(
                device="cuda",
                start_layer=0,
                end_layer=2,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=10,
                model_context_len=16,
                state_cache_params=SimpleNamespace(layers=(1,)),
                speculative_num_draft_tokens=3,
                post_capture_active=True,
            ),
        )
        a = storage.allocator
        buffers = [
            a.arena.raw,
            storage.requests.req_to_token,
            *storage.scratch.tensors.values(),
        ]
        buffers.extend(
            t
            for b in a.members
            for t in (b.ids.virtual_to_physical, b.ids.physical_to_virtual)
        )
        identities = [(t.data_ptr(), tuple(t.shape)) for t in buffers]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            a.arena.raw[:1].fill_(7)
        # Only five state slots plus their minimum token demand fit now.
        state_bytes = plan.sub_pool("mamba").spec.entry_bytes()
        budget = (
            captured.fixed_bytes
            + plan.sink_bytes()
            + 5 * state_bytes
            + 16 * plan.owner().spec.entry_bytes()
        )
        final = size_plan(
            plan,
            KVSizingInputs(
                budget_bytes=budget,
                max_running_requests=10,
                radix_cache=False,
                context_len=16,
                alignment=1,
            ),
            frozen=captured,
        )
        self.assertLess(final.max_running_requests, captured.max_running_requests)
        self.assertGreater(final.max_running_requests, 0)
        self.assertEqual(final.units("mamba"), final.max_running_requests)
        self.assertEqual(final.fixed, captured.fixed)
        config = SimpleNamespace(
            unified_memory_pool_bytes=final.total_bytes,
            unified_full_cap_tokens=final.full_cap_tokens,
            full_max_total_num_tokens=final.full_tokens,
            swa_max_total_num_tokens=final.window_tokens,
        )
        a.get_kvcache().finalize_backing(config)
        a.resize(config)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(a.arena.raw[0].item(), 7)
        self.assertEqual([(t.data_ptr(), tuple(t.shape)) for t in buffers], identities)
        slots = a.alloc_state_slots(final.max_running_requests)
        self.assertIsNotNone(slots)
        tokens = a.alloc(
            -(-2 * final.max_running_requests // a.page_size) * a.page_size
        )
        self.assertIsNotNone(tokens)
        a.free(tokens)
        a.state_slots.free(slots)
        self.assertEqual(storage.accounting_violations(), [])

    def test_reservation_backs_only_the_sink_before_capture(self):
        """Before sizing, the arena is address space: the sink is resident and
        only it may be handed to transfer engines, not the reserved bound."""
        bundle = self._post_capture_bundle()
        shared = bundle.arena
        self.assertTrue(bundle.token_to_kv_pool.post_capture_active)
        self.assertEqual(
            shared.total_bytes
            + bundle.token_to_kv_pool_allocator.storage.sized.fixed_bytes,
            _RESERVED_BYTES,
        )
        self.assertGreater(shared.accessible_bytes, 0)
        self.assertLessEqual(shared.accessible_bytes, shared.backed_bytes)
        self.assertLess(shared.backed_bytes, _RESERVED_BYTES)
        _, buf_lens, _ = bundle.token_to_kv_pool.get_contiguous_buf_infos()
        self.assertEqual(buf_lens, [shared.accessible_bytes])

    def test_finalize_and_resize_match_the_eager_pool_of_the_final_size(self):
        bundle = self._post_capture_bundle()
        shared = bundle.arena
        allocator = bundle.token_to_kv_pool_allocator
        captured_ptr = shared.raw.data_ptr()
        tables = [
            table
            for sub in (allocator.full_attn_allocator, allocator.swa_attn_allocator)
            for table in (sub.ids.virtual_to_physical, sub.ids.physical_to_virtual)
        ]
        table_ptrs = [table.data_ptr() for table in tables]

        config = _final_config(bundle)
        bundle.token_to_kv_pool.finalize_backing(config)
        allocator.resize(config)

        # Graphs captured these addresses; sizing may not move them.
        self.assertEqual(shared.raw.data_ptr(), captured_ptr)
        self.assertEqual([table.data_ptr() for table in tables], table_ptrs)
        self.assertGreaterEqual(shared.backed_bytes, shared.accessible_bytes)
        # The whole advertised span is resident, not just the sink.
        last = shared.raw[shared.accessible_bytes - 1]
        last.fill_(7)
        torch.cuda.synchronize()
        self.assertEqual(int(last.item()), 7)

        eager = _bundle(total_bytes=_FINAL_BYTES, post_capture_active=False)
        self.assertEqual(_capacities(bundle), _capacities(eager))
        self.assertEqual(allocator.verify_byte_accounting(), [])

        # The resized capacity is fully allocatable and accounted.
        n = allocator.available_size()
        self.assertGreater(n, 0)
        self.assertIsNotNone(allocator.alloc(n))
        self.assertEqual(allocator.available_size(), 0)

    def test_retraction_restores_complete_entries_at_new_dcp_coordinates(self):
        placement = FusedDraftPlacement.from_counts(
            counts={"full": [2]},
            regions={
                "full": DenseDraftRegion(
                    lane_num=2,
                    head_num=1,
                    head_dim=3,
                    v_head_dim=5,
                    store_dtype=torch.bfloat16,
                )
            },
        )
        for dcp in (1, 2, 4):
            for rank in range(dcp):
                with (
                    self.subTest(dcp=dcp, rank=rank),
                    get_parallel().override(attn_dcp_size=dcp, attn_dcp_rank=rank),
                ):
                    bundle = _bundle(
                        total_bytes=_FINAL_BYTES,
                        post_capture_active=False,
                        chain="state",
                        fused_draft=placement,
                    )
                    allocator = bundle.token_to_kv_pool_allocator
                    source = allocator.alloc(16 * dcp)
                    old_state = allocator.alloc_state_slots(1)
                    target = allocator.alloc(16 * dcp)
                    new_state = allocator.alloc_state_slots(1)
                    for band in allocator.members:
                        virtual = old_state if band.sub_pool_name == "mamba" else source
                        physical = band.ids.translate_write(virtual.to(torch.int64))
                        physical = physical[physical > 0]
                        allocator.arena.entries(band.sub_pool_name)[physical] = 79
                    saved = allocator.get_cpu_copy(source, mamba_indices=old_state)
                    allocator.load_cpu_copy(saved, target, mamba_indices=new_state)
                    for band in allocator.members:
                        virtual = new_state if band.sub_pool_name == "mamba" else target
                        physical = band.ids.translate_write(virtual.to(torch.int64))
                        physical = physical[physical > 0]
                        self.assertTrue(
                            torch.all(
                                allocator.arena.entries(band.sub_pool_name)[physical]
                                == 79
                            )
                        )

    def test_every_chain_retains_captured_and_fixed_addresses(self):
        for chain, dcp in (
            ("full", 1),
            ("swa", 1),
            ("state", 1),
            ("tri", 1),
            ("state", 2),
            ("state", 4),
        ):
            with (
                self.subTest(chain=chain, dcp=dcp),
                get_parallel().override(attn_dcp_size=dcp),
            ):
                bundle = _bundle(
                    total_bytes=_RESERVED_BYTES, post_capture_active=True, chain=chain
                )
                allocator = bundle.token_to_kv_pool_allocator
                storage = allocator.storage
                tensors = [bundle.arena.raw, storage.requests.req_to_token]
                tensors.extend(storage.scratch.tensors.values())
                for band in allocator.members:
                    tensors.extend(
                        (band.ids.virtual_to_physical, band.ids.physical_to_virtual)
                    )
                pointers = [t.data_ptr() for t in tensors]
                shapes = [tuple(t.shape) for t in tensors]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    bundle.arena.raw[:1].fill_(9)
                config = _final_config(bundle)
                bundle.token_to_kv_pool.finalize_backing(config)
                allocator.resize(config)
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(bundle.arena.raw[0].item(), 9)
                self.assertEqual([t.data_ptr() for t in tensors], pointers)
                self.assertEqual([tuple(t.shape) for t in tensors], shapes)
                eager = _bundle(
                    total_bytes=_FINAL_BYTES, post_capture_active=False, chain=chain
                )
                self.assertEqual(
                    allocator.available_size(),
                    eager.token_to_kv_pool_allocator.available_size(),
                )
                self.assertEqual(allocator.verify_byte_accounting(), [])
                for band in allocator.members:
                    self.assertEqual(
                        band.max_slots, bundle.arena.num_entries(band.sub_pool_name)
                    )
                    view = allocator.views[band.sub_pool_name]
                    self.assertEqual(view.size, band.max_slots - band.pool_page_size)
                if allocator.state is not None:
                    state = allocator.state.pool
                    self.assertEqual(allocator.state_slots.size, state.size)
                    self.assertEqual(allocator.state_slots.available_size(), state.size)
                    self.assertEqual(state.host_capacity_tokens, state.size)
                    self.assertEqual(
                        state.host_capacity_bytes,
                        state.size * allocator.mamba_allocator.entry_bytes,
                    )


    def test_short_draft_rows_store_and_replay_without_touching_host(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                region = DenseDraftRegion(
                    lane_num=2, head_num=1, head_dim=3, v_head_dim=5, store_dtype=dtype
                )
                bundle = _bundle(
                    total_bytes=1 << 20,
                    post_capture_active=False,
                    fused_draft=FusedDraftPlacement.from_counts(
                        counts={"full": [2]}, regions={"full": region}
                    ),
                )
                arena = bundle.arena
                spec = bundle.storage.plan.owner().spec
                view = DraftMHAView(
                    arena=arena,
                    host_spec=spec,
                    layer_lanes={0: 0, 1: 1},
                    page_size=_PAGE_SIZE,
                )
                arena.raw.fill_(11)
                before = arena.raw.clone()
                loc = torch.tensor([8, 13], device="cuda", dtype=torch.int64)
                k = (
                    torch.arange(6, device="cuda", dtype=torch.float32)
                    .view(2, 1, 3)
                    .to(dtype)
                )
                v = (
                    torch.arange(10, device="cuda", dtype=torch.float32)
                    .view(2, 1, 5)
                    .to(dtype)
                )
                layer = SimpleNamespace(layer_id=1)
                view.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    view.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
                k.add_(3)
                v.add_(5)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(view.k_buffer[1][loc], k)
                torch.testing.assert_close(view.v_buffer[1][loc], v)
                changed = torch.zeros_like(arena.raw, dtype=torch.bool)
                for part in region.parts(spec.draft_offset_in_entry()):
                    for row in (8, 13):
                        lo = row * spec.entry_bytes() + part.layer_offset_bytes(1)
                        changed[lo : lo + part.row_bytes()] = True
                self.assertTrue(torch.equal(arena.raw[~changed], before[~changed]))

    def test_replicated_draft_graph_refreshes_after_owner_compaction(self):
        from sglang.srt.mem_cache.kv.alloc.id_space import ReplicatedTokenIds

        for factor in (2, 4):
            with (
                self.subTest(dcp=factor),
                get_parallel().override(attn_dcp_size=factor),
            ):
                region = DenseDraftRegion(
                    lane_num=1,
                    head_num=1,
                    head_dim=3,
                    v_head_dim=5,
                    store_dtype=torch.float16,
                    token_replicas=factor,
                )
                bundle = _bundle(
                    total_bytes=1 << 20,
                    post_capture_active=False,
                    chain="state",
                    fused_draft=FusedDraftPlacement.from_counts(
                        counts={"full": [1]}, regions={"full": region}
                    ),
                )
                alloc = bundle.token_to_kv_pool_allocator
                first = alloc.alloc(_PAGE_SIZE * factor)
                live = alloc.alloc(_PAGE_SIZE * factor)
                coords = ReplicatedTokenIds(alloc.full_attn_allocator.ids, factor)
                space = coords.space("full")
                view = DraftMHAView(
                    arena=bundle.arena,
                    host_spec=bundle.storage.plan.owner().spec,
                    layer_lanes={0: 0},
                    page_size=_PAGE_SIZE,
                )
                k = torch.arange(
                    live.numel() * 3, device="cuda", dtype=torch.float16
                ).reshape(-1, 1, 3)
                v = torch.arange(
                    live.numel() * 5, device="cuda", dtype=torch.float16
                ).reshape(-1, 1, 5)
                layer = SimpleNamespace(layer_id=0)
                # Compile both kernels before capture.
                coords.refresh()
                view.set_kv_buffer(
                    layer, KVWriteLoc(coords.write(live), physical=True), k, v
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    coords.refresh()
                    loc = coords.write(live)
                    view.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
                pointer = space.read_v2p.data_ptr()
                before = coords.write(live).clone()
                alloc.free(first)
                k.add_(7)
                v.add_(11)
                graph.replay()
                torch.cuda.synchronize()
                after = coords.write(live)
                self.assertFalse(torch.equal(before, after))
                torch.testing.assert_close(view.k_buffer[0][after.long()], k)
                torch.testing.assert_close(view.v_buffer[0][after.long()], v)
                virtual_pages = live[::_PAGE_SIZE] // _PAGE_SIZE
                torch.testing.assert_close(
                    space.read_v2p[virtual_pages.long()],
                    after[::_PAGE_SIZE] // _PAGE_SIZE,
                )
                self.assertEqual(space.read_v2p.data_ptr(), pointer)
                alloc.free(live)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_dcp_host_pages_preserve_owner_and_draft_window_bytes(self):
        from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions

        for factor in (1, 2, 4):
            with (
                self.subTest(dcp=factor),
                get_parallel().override(attn_dcp_size=factor),
            ):
                region = DenseDraftRegion(
                    lane_num=1,
                    head_num=1,
                    head_dim=3,
                    v_head_dim=5,
                    store_dtype=torch.float16,
                    token_replicas=factor,
                )
                placement = FusedDraftPlacement.from_counts(
                    counts={"full": [1], "swa": [1]},
                    regions={"full": region, "swa": region},
                    new_bands={"swa": DraftBand(kind="window", window=32)},
                )
                bundle = _bundle(
                    total_bytes=1 << 20,
                    post_capture_active=False,
                    chain="state",
                    fused_draft=placement,
                )
                alloc = bundle.token_to_kv_pool_allocator
                storage = alloc.storage
                ids = alloc.alloc(8 * factor)
                for name in ("full", "swa"):
                    layout = storage.host_layout(name)
                    host = layout.build(
                        HostPoolOptions(
                            ratio=1,
                            size=0,
                            page_size=alloc.page_size,
                            layout="layer_first",
                            allocator_type="default",
                            pool_label=name,
                        )
                    )
                    host_ids = host.alloc(ids.numel())
                    translate = storage.host_ids(name)
                    device_ids = (
                        translate(ids)
                        if translate is not None
                        else storage.window_ids(name, ids)
                    )
                    pages = device_ids[:: alloc.page_size] // alloc.page_size
                    buffer = layout.get_page_envelope_buffer()
                    buffer[pages] = 29
                    expected = buffer[pages].clone()
                    host.backup_from_device_all_layer(
                        layout, host_ids, device_ids, "kernel"
                    )
                    torch.cuda.synchronize()
                    buffer[pages] = 0
                    host.load_to_device_per_layer(
                        layout, host_ids, device_ids, 0, "kernel"
                    )
                    torch.cuda.synchronize()
                    torch.testing.assert_close(buffer[pages], expected)
                    host.free(host_ids)
                    host.destroy()
                alloc.free(ids)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_two_window_chain_resizes_after_capture(self):
        region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=3, v_head_dim=5, store_dtype=torch.float16
        )
        placement = FusedDraftPlacement.from_counts(
            counts={"draft_swa": [1]},
            regions={"draft_swa": region},
            new_bands={"draft_swa": DraftBand(kind="window", window=32)},
        )
        captured = _bundle(
            total_bytes=_RESERVED_BYTES,
            post_capture_active=True,
            chain="tri",
            fused_draft=placement,
        )
        alloc = captured.token_to_kv_pool_allocator
        graph = torch.cuda.CUDAGraph()
        pointer = captured.arena.raw.data_ptr()
        with torch.cuda.graph(graph):
            captured.arena.raw[:1].fill_(7)
        config = _final_config(captured)
        captured.token_to_kv_pool.finalize_backing(config)
        alloc.resize(config)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(captured.arena.raw.data_ptr(), pointer)
        self.assertEqual(captured.arena.raw[0].item(), 7)
        eager = _bundle(
            total_bytes=_FINAL_BYTES,
            post_capture_active=False,
            chain="tri",
            fused_draft=placement,
        )
        self.assertEqual(
            alloc.available_size(), eager.token_to_kv_pool_allocator.available_size()
        )
        ids = alloc.alloc(8)
        self.assertIsNotNone(ids)
        alloc.free(ids)
        self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_sizing_contract_is_enforced(self):
        bundle = self._post_capture_bundle()
        kvcache = bundle.token_to_kv_pool
        allocator = bundle.token_to_kv_pool_allocator
        with self.subTest("resize before finalize_backing"):
            with self.assertRaisesRegex(
                RuntimeError, "must be backed before allocator.resize"
            ):
                allocator.resize(_final_config(bundle))
        with self.subTest("budget without a byte count"):
            with self.assertRaises(ValueError):
                kvcache.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=None)
                )
        with self.subTest("budget above the reservation"):
            with self.assertRaises(ValueError):
                kvcache.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=2 * _RESERVED_BYTES)
                )
        with self.subTest("budget below the bs=1 floor"):
            with self.assertRaisesRegex(RuntimeError, "bs=1 floor"):
                kvcache.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=4096)
                )
        with self.subTest("resize with live allocations"):
            kvcache.finalize_backing(_final_config(bundle))
            # Move the grow-down allocator's frontier into the backed range
            # before allocating; it still reflects the larger reservation.
            allocator.resize(_final_config(bundle))
            handed = allocator.alloc(_PAGE_SIZE)
            self.assertIsNotNone(handed)
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                allocator.resize(_final_config(bundle))
            allocator.free(handed)

        eager = _bundle(total_bytes=_FINAL_BYTES, post_capture_active=False)
        with self.subTest("eager pool cannot be resized"):
            self.assertFalse(eager.token_to_kv_pool.post_capture_active)
            with self.assertRaises(RuntimeError):
                eager.token_to_kv_pool.finalize_backing(
                    SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
                )
            with self.assertRaises(RuntimeError):
                eager.token_to_kv_pool_allocator.resize(
                    SimpleNamespace(unified_memory_pool_bytes=_FINAL_BYTES)
                )


if __name__ == "__main__":
    unittest.main()
