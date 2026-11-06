"""HiCache page format and TP identity follow different storage properties."""

import ctypes
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MLAGeometry, StateGeometry, kv_layer_decls
from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftBand,
    FusedDraftPlacement,
    MLADraftRegion,
)
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ClientConfig:
    @classmethod
    def from_environment(cls):
        return SimpleNamespace(
            role=None,
            dram=SimpleNamespace(capacity_bytes=0),
            ssd=SimpleNamespace(
                enabled=False,
                storage_dir="/tmp",
                capacity_bytes=0,
                ssd_backend="file",
                spdk_proxy_tenant_id=0,
                spdk_proxy_tenant_quota_bytes=0,
            ),
            distributed=None,
            standalone_process=None,
        )


def _storage_and_host(band, *, mla_draft=False):
    geometry = MLAGeometry(
        kv_lora_rank=8, qk_rope_head_dim=8, kv_cache_dtype=torch.bfloat16
    )
    placement = (
        None
        if band is None
        else FusedDraftPlacement.from_counts(
            counts={band: [1]},
            regions={
                band: (
                    MLADraftRegion(
                        lane_num=1,
                        kv_lora_rank=8,
                        qk_rope_head_dim=8,
                        store_dtype=torch.bfloat16,
                    )
                    if mla_draft
                    else DenseDraftRegion(
                        lane_num=1,
                        head_num=1,
                        head_dim=8,
                        store_dtype=torch.bfloat16,
                    )
                )
            },
            new_bands={}
            if band == "full"
            else {band: DraftBand(kind="window", window=8)},
        )
    )
    plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=geometry,
            window_layer_ids=(2,) if band == "draft_swa" else (),
            window_geometry=geometry,
            window=16,
            state_layer_ids=(1,),
            state_geometry=StateGeometry(
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(1, 4, 8),
                temporal_dtype=torch.float32,
            ),
        ),
        page_size=4,
        fused_draft=placement,
    )
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=64,
            window_tokens=32,
            state_slots=4,
            max_running_requests=1,
            context_len=64,
            alignment=1,
        ),
    )
    storage = build_kv_storage(
        plan,
        sized,
        UnifiedBuildContext(
            device="cpu",
            start_layer=0,
            end_layer=3,
            enable_memory_saver=False,
            need_sort=False,
            max_num_reqs=1,
            model_context_len=64,
            state_cache_params=SimpleNamespace(layers=(1,)),
        ),
    )
    with host_memory_budget_scope(1 << 30):
        host = storage.host_layout("full").build(
            HostPoolOptions(
                ratio=1,
                size=0,
                page_size=4,
                layout="page_first",
                allocator_type="default",
                pool_label="full",
                pin_memory=False,
            )
        )
    composite = SimpleNamespace(
        layout="page_first", anchor_entry=SimpleNamespace(host_pool=host)
    )
    return storage, host, composite


class TestStorageIdentity(unittest.TestCase):
    def test_mla_anchor_with_full_or_window_draft_keeps_rank_identity(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
        from sglang.srt.mem_cache.storage.umbp import umbp_store

        setup_allocator_context()
        for band in (None, "full", "swa", "draft_swa"):
            with self.subTest(band=band), tempfile.TemporaryDirectory() as directory:
                storage, host, composite = _storage_and_host(band)
                records = []
                for rank in (0, 1):
                    with patch("sglang.srt.managers.cache_controller.LayerDoneCounter"):
                        controller = HybridCacheController(
                            token_to_kv_pool_allocator=storage.allocator,
                            mem_pool_host=composite,
                            page_size=4,
                            tp_group=None,
                            load_cache_event=threading.Event(),
                        )
                    with (
                        get_parallel().override(
                            tp_rank=rank,
                            tp_size=2,
                            attn_tp_rank=rank,
                            attn_tp_size=2,
                            moe_tp_rank=rank,
                            moe_tp_size=2,
                            pp_rank=0,
                        ),
                        patch.object(
                            controller, "_build_unified_plan", return_value=None
                        ),
                    ):
                        config = controller._generate_storage_config("mla-draft")
                    self.assertTrue(config.is_mla_model)
                    self.assertEqual(config.is_rank_replicated, band is None)
                    self.assertEqual(
                        config.is_storage_writer, band is not None or rank == 0
                    )
                    client = MagicMock()
                    written = {}

                    def put(keys, ptrs, sizes):
                        written.update(
                            (k, ctypes.string_at(p, n))
                            for k, p, n in zip(keys, ptrs, sizes)
                        )
                        return [True] * len(keys)

                    client.batch_put_from_ptr.side_effect = put
                    imported = (
                        lambda cfg: client,
                        _ClientConfig,
                        SimpleNamespace(
                            Standalone="standalone",
                            SharedSSDLeader="leader",
                            SharedSSDFollower="follower",
                        ),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    with patch.object(
                        umbp_store, "_import_umbp_client", return_value=imported
                    ):
                        store = umbp_store.UMBPStore(config, host)
                    indices = host.alloc(4)
                    self.assertIsNotNone(indices)
                    self.assertEqual(store.batch_set_v1(["page"], indices), [True])
                    self.assertEqual(bool(written), band is not None or rank == 0)
                    self.assertEqual(len(store._anchor_key_suffixes()), 1)
                    keys, ptrs, _ = store._batch_preprocess(["page"], indices)
                    self.assertEqual(len(keys), len(ptrs))
                    backend = HiCacheFile(config, file_path=directory)
                    payload = torch.full((64,), 100 + rank, dtype=torch.uint8)
                    if config.is_storage_writer:
                        self.assertTrue(backend.set("window", payload))
                    records.append((store, keys, backend, payload))
                if band is not None:
                    self.assertNotEqual(records[0][1], records[1][1])
                    for _, _, backend, expected in records:
                        actual = torch.empty_like(expected)
                        self.assertIsNotNone(backend.get("window", actual))
                        torch.testing.assert_close(actual, expected)
                else:
                    self.assertEqual(records[0][1], records[1][1])
                for store, *_ in records:
                    store.close()

    def test_replicated_mla_envelope_writer_and_readers_share_keys(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
        from sglang.srt.mem_cache.storage.umbp import umbp_store

        setup_allocator_context()
        stage_keys = {}
        for pp_size, pp_rank in ((1, 0), (2, 0), (2, 1)):
            with self.subTest(pp_size=pp_size, pp_rank=pp_rank):
                storage, host, composite = _storage_and_host("full", mla_draft=True)
                self.assertTrue(host.stores_page_envelope)
                written = {}
                records = []
                for rank in (0, 1):
                    with patch("sglang.srt.managers.cache_controller.LayerDoneCounter"):
                        controller = HybridCacheController(
                            token_to_kv_pool_allocator=storage.allocator,
                            mem_pool_host=composite,
                            page_size=4,
                            tp_group=None,
                            load_cache_event=threading.Event(),
                        )
                    with (
                        get_parallel().override(
                            tp_rank=rank,
                            tp_size=2,
                            attn_tp_rank=rank,
                            attn_tp_size=2,
                            moe_tp_rank=rank,
                            moe_tp_size=2,
                            pp_rank=pp_rank,
                            pp_size=pp_size,
                        ),
                        patch.object(
                            controller, "_build_unified_plan", return_value=None
                        ),
                    ):
                        config = controller._generate_storage_config("mla-draft")
                    self.assertTrue(config.is_rank_replicated)
                    self.assertEqual(config.is_storage_writer, rank == 0)
                    client = MagicMock()

                    def put(keys, ptrs, sizes):
                        written.update(
                            (key, ctypes.string_at(ptr, size))
                            for key, ptr, size in zip(keys, ptrs, sizes)
                        )
                        return [True] * len(keys)

                    def get(keys, ptrs, sizes):
                        for key, ptr, size in zip(keys, ptrs, sizes):
                            if key in written:
                                self.assertEqual(len(written[key]), size)
                                ctypes.memmove(ptr, written[key], size)
                        return [key in written for key in keys]

                    def exists(keys):
                        for i, key in enumerate(keys):
                            if key not in written:
                                return i
                        return len(keys)

                    client.batch_put_from_ptr.side_effect = put
                    client.batch_get_into_ptr.side_effect = get
                    client.batch_exists_consecutive.side_effect = exists
                    imported = (
                        lambda cfg: client,
                        _ClientConfig,
                        SimpleNamespace(
                            Standalone="standalone",
                            SharedSSDLeader="leader",
                            SharedSSDFollower="follower",
                        ),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    with patch.object(
                        umbp_store, "_import_umbp_client", return_value=imported
                    ):
                        store = umbp_store.UMBPStore(config, host)
                    self.addCleanup(store.close)
                    indices = host.alloc(4)
                    keys, ptrs, sizes = store._batch_preprocess(["page"], indices)
                    for ptr, size in zip(ptrs, sizes):
                        ctypes.memset(ptr, 101 + rank, size)
                    self.assertEqual(store.batch_set_v1(["page"], indices), [True])
                    self.assertEqual(
                        client.batch_put_from_ptr.call_count, 1 if rank == 0 else 0
                    )
                    records.append((store, indices, keys, ptrs, sizes))
                self.assertEqual(records[0][2], records[1][2])
                stage_keys[(pp_size, pp_rank)] = records[0][2]
                for store, indices, _, ptrs, sizes in records:
                    self.assertEqual(store.batch_exists(["page"]), 1)
                    for ptr, size in zip(ptrs, sizes):
                        ctypes.memset(ptr, 0, size)
                    self.assertEqual(store.batch_get_v1(["page"], indices), [True])
                    for ptr, size in zip(ptrs, sizes):
                        self.assertEqual(
                            ctypes.string_at(ptr, size), bytes([101]) * size
                        )
        self.assertNotEqual(stage_keys[(2, 0)], stage_keys[(2, 1)])


if __name__ == "__main__":
    unittest.main()
