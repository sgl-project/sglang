"""Dynamic HiCache storage under DCP: the factory admits a plugin by its
declared capabilities and validates the real host pools before the plugin
constructor runs, and a late sidecar is published only after the backend
accepted it."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo, MLABufferInfo
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolName,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

DCP = 8
PHYSICAL_PAGE = 16
PLUGIN_MODULE = "toy_dynamic_storage"
ALL_CAPABILITIES = frozenset(
    {
        "mla_sharded",
        "logical_page_indices",
        "mamba_tp_sharded",
        "exact_restorable_prefix_pages",
        "draft_mla_sharded",
    }
)


def _mla_pool(dcp=DCP, rank=0, layout="page_first_direct", physical_page=PHYSICAL_PAGE):
    buffers = tuple(torch.empty((256, 1, 12), dtype=torch.float16) for _ in range(2))
    info = MLABufferInfo(
        buffers=buffers,
        data_ptrs=torch.tensor([b.data_ptr() for b in buffers], dtype=torch.uint64),
        layer_indices=(0, 1),
        device_layer_num=2,
        page_size=physical_page,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        kv_cache_dim=12,
        dtype=torch.float16,
        device=torch.device("cpu"),
        scales=None,
    )
    return MLATokenToKVPoolHost(
        info,
        host_to_device_ratio=1.0,
        host_size=0,
        page_size=physical_page * dcp,
        layout=layout,
        pin_memory=False,
        device="cpu",
        dcp_size=dcp,
        dcp_rank=rank,
        device_capacity_tokens=256,
        start_layer=0,
        end_layer=2,
        target_layer_num=2,
    )


def _mamba_pool():
    return MambaPoolHost(
        MambaStateBufferInfo(
            temporal=torch.zeros((2, 16, 2, 4), dtype=torch.float32),
            conv=(torch.zeros((2, 16, 4, 2), dtype=torch.float16),),
        ),
        host_to_device_ratio=1.0,
        host_size=0,
        device_capacity=16,
        pin_memory=False,
        device="cpu",
        layout="page_first_direct",
    )


def _config(dcp=DCP):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=8,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="toy",
        dcp_size=dcp,
        dcp_rank=0,
        logical_page_size=PHYSICAL_PAGE * dcp,
        extra_config={
            "backend_name": "toy",
            "module_path": PLUGIN_MODULE,
            "class_name": "ToyStorage",
        },
    )


class TestDynamicStorageDCPContract(CustomTestCase):
    def setUp(self):
        events = self.events = []

        class ToyStorage(HiCacheStorage):
            dcp_storage_capabilities = ALL_CAPABILITIES

            @classmethod
            def validate_dcp_config(cls, cfg, pools):
                events.append(("validate", tuple(pools)))
                if cfg.extra_config.get("reject_validation"):
                    raise ValueError("toy format rejected")

            def __init__(self, cfg, extra_config=None):
                events.append(("construct",))

            get = set = batch_get = batch_set = exists = None

        self.plugin = ToyStorage
        module = types.ModuleType(PLUGIN_MODULE)
        module.ToyStorage = ToyStorage
        sys.modules[PLUGIN_MODULE] = module
        self.addCleanup(sys.modules.pop, PLUGIN_MODULE)

    def create(self, cfg=None, pools=None):
        cfg = cfg or _config()
        pools = pools or {PoolName.KV: _mla_pool()}
        return StorageBackendFactory.create_backend(
            "dynamic", cfg, pools[PoolName.KV], storage_host_pools=pools
        )

    def test_validation_sees_every_pool_and_precedes_constructor(self):
        pools = {
            PoolName.KV: _mla_pool(),
            PoolName.MAMBA: _mamba_pool(),
            PoolName.DRAFT: _mla_pool(),
        }
        self.assertIsInstance(self.create(pools=pools), self.plugin)
        self.assertEqual(self.events, [("validate", tuple(pools)), ("construct",)])

        self.events.clear()
        cfg = _config()
        cfg.extra_config["reject_validation"] = True
        with self.assertRaises(ValueError):
            self.create(cfg, pools)
        self.assertEqual(self.events, [("validate", tuple(pools))])

    def test_legacy_plugin_is_admitted_at_dcp1_only(self):
        self.plugin.dcp_storage_capabilities = frozenset()
        self.plugin.validate_dcp_config = HiCacheStorage.validate_dcp_config
        self.create(_config(dcp=1), {PoolName.KV: _mla_pool(dcp=1)})
        self.assertEqual(self.events, [("construct",)])

        self.events.clear()
        cfg = _config()
        # Configuration flags never stand in for a class-level declaration.
        cfg.extra_config["dcp_storage_capabilities"] = sorted(ALL_CAPABILITIES)
        with self.assertRaises(NotImplementedError):
            self.create(cfg)
        self.assertEqual(self.events, [])

    def test_opted_in_plugin_is_validated_at_dcp1(self):
        cfg = _config(dcp=1)
        cfg.extra_config["reject_validation"] = True
        with self.assertRaises(ValueError):
            self.create(cfg, {PoolName.KV: _mla_pool(dcp=1)})
        self.assertNotIn(("construct",), self.events)

    def test_each_sidecar_requires_its_own_capability(self):
        for name, pool, capability in (
            (PoolName.MAMBA, _mamba_pool(), "mamba_tp_sharded"),
            (PoolName.MAMBA, _mamba_pool(), "exact_restorable_prefix_pages"),
            (PoolName.DRAFT, _mla_pool(), "draft_mla_sharded"),
        ):
            with self.subTest(capability=capability):
                self.plugin.dcp_storage_capabilities = ALL_CAPABILITIES - {capability}
                with self.assertRaises(NotImplementedError):
                    self.create(pools={PoolName.KV: _mla_pool(), name: pool})
        self.assertEqual(self.events, [])

    def test_pool_geometry_and_type_mismatches_are_rejected(self):
        unmaterialized = _mla_pool()
        unmaterialized.kv_buffer = None
        split = _mla_pool()
        split.layout = "page_first_kv_split"
        cases = {
            "other dcp rank": {PoolName.KV: _mla_pool(rank=1)},
            "other dcp size": {PoolName.KV: _mla_pool(dcp=4)},
            "other logical page": {PoolName.KV: _mla_pool(physical_page=32)},
            "split kv layout": {PoolName.KV: split},
            "no kv buffer": {PoolName.KV: unmaterialized},
            "non-mla kv object": {PoolName.KV: SimpleNamespace(**_mla_pool().__dict__)},
            "draft layout differs": {
                PoolName.KV: _mla_pool(),
                PoolName.DRAFT: _mla_pool(layout="page_first"),
            },
            "mla pool named mamba": {
                PoolName.KV: _mla_pool(),
                PoolName.MAMBA: _mla_pool(),
            },
            "swa sidecar": {PoolName.KV: _mla_pool(), PoolName.SWA: _mla_pool()},
        }
        for label, pools in cases.items():
            with self.subTest(label):
                with self.assertRaises(NotImplementedError):
                    self.create(pools=pools)
        self.assertEqual(self.events, [])

    def test_late_sidecar_is_published_only_after_the_backend_accepts_it(self):
        cfg = _config()
        anchor = _mla_pool()
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=anchor,
                    device_pool=None,
                    layer_mapper=lambda layer: layer,
                    is_primary_index_anchor=True,
                )
            ]
        )
        backend = self.plugin(cfg)
        backend.register_mem_host_pool_v2 = mock.Mock()
        backend.on_host_pools_registered = mock.Mock()
        controller = SimpleNamespace(
            mem_pool_host=group,
            storage_host_pool=anchor,
            enable_storage=True,
            storage_backend=backend,
            storage_backend_type="dynamic",
            storage_config=cfg,
            extra_host_mem_release_queues={},
        )
        controller._storage_host_pools = lambda: HiCacheController._storage_host_pools(
            controller
        )
        entry = PoolEntry(
            name=PoolName.DRAFT,
            host_pool=_mla_pool(),
            device_pool=None,
            layer_mapper=lambda layer: layer,
        )

        def register():
            HybridCacheController.register_host_pool_entry(controller, entry)

        cfg.extra_config["reject_validation"] = True
        with self.assertRaises(ValueError):
            register()
        backend.register_mem_host_pool_v2.assert_not_called()

        cfg.extra_config.pop("reject_validation")
        backend.register_mem_host_pool_v2.side_effect = RuntimeError("native")
        with self.assertRaises(RuntimeError):
            register()
        self.assertNotIn(PoolName.DRAFT, group.entry_map)
        self.assertEqual(controller.extra_host_mem_release_queues, {})

        backend.register_mem_host_pool_v2.side_effect = None
        register()
        self.assertIs(group.entry_map[PoolName.DRAFT], entry)
        self.assertIn(PoolName.DRAFT, controller.extra_host_mem_release_queues)
        backend.on_host_pools_registered.assert_not_called()

        # An entry the group would reject must fail before native registration:
        # add_entry() rejecting it afterwards would leave the backend modified.
        backend.register_mem_host_pool_v2.reset_mock()
        with self.assertRaises(ValueError):
            register()
        backend.register_mem_host_pool_v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
