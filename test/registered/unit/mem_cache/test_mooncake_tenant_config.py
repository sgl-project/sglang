import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import cache_controller
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    DEFAULT_TENANT_ID,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _fake_mooncake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls

    class ReplicateConfig:
        pass

    mooncake_store.ReplicateConfig = ReplicateConfig
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
    }


class FakeMooncakeDistributedStore:
    instances = []

    def __init__(self):
        self.setup_calls = []
        self.objects = {}
        type(self).instances.append(self)

    def setup(self, *args, **kwargs):
        self.setup_calls.append((args, kwargs))
        return 0

    def register_buffer(self, *args, **kwargs):
        return 0

    def put(self, key, value, *args):
        self.objects[key] = value
        return 0

    def is_exist(self, key):
        return 1 if key in self.objects else 0

    def get(self, key):
        return self.objects.get(key)

    def batch_is_exist(self, keys):
        return [self.is_exist(key) for key in keys]


class OldMooncakeDistributedStore(FakeMooncakeDistributedStore):
    instances = []

    def setup(self, *args, **kwargs):
        if "tenant_id" in kwargs:
            raise TypeError("tenant_id is an invalid keyword argument")
        return super().setup(*args, **kwargs)


def _make_storage_config(tenant_id=DEFAULT_TENANT_ID):
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
    }
    if tenant_id is not None:
        extra_config["tenant_id"] = tenant_id

    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        dcp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        tp_lcm_size=None,
        should_split_heads=False,
        extra_config=extra_config,
    )


class TestMooncakeTenantConfig(CustomTestCase):
    def test_dtype_keys_isolate_objects_without_changing_tenants(self):
        """A default/old master sharing one namespace must not mix KV encodings."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        parallel = SimpleNamespace(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_dcp_size=1, attn_dcp_rank=0
        )
        for tenant, client in (
            (DEFAULT_TENANT_ID, OldMooncakeDistributedStore),
            ("tenant-a", FakeMooncakeDistributedStore),
        ):
            with (
                self.subTest(tenant=tenant),
                patch.object(cache_controller, "get_parallel", return_value=parallel),
                patch.object(
                    cache_controller, "is_dp_attention_enabled", return_value=False
                ),
            ):
                objects = {}
                stores = []
                for dtype in (
                    torch.bfloat16,
                    torch.float16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ):
                    host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
                    host.dtype = torch.uint8 if dtype.itemsize == 1 else dtype
                    host._value_dtype = dtype
                    controller = SimpleNamespace(
                        enable_storage_metrics=False,
                        mem_pool_host=SimpleNamespace(layout="page_first"),
                        mem_pool_device=SimpleNamespace(),
                        storage_host_pool=host,
                        get_attn_cp_rank_and_size=lambda: (0, 1),
                        _build_unified_plan=lambda *_: None,
                    )
                    extra = _make_storage_config(tenant).extra_config
                    extra["extra_backend_tag"] = "prod"
                    cfg = cache_controller.HiCacheController._generate_storage_config(
                        controller, "test", extra
                    )
                    with patch.dict("sys.modules", _fake_mooncake_modules(client)):
                        store = MooncakeStore(cfg)
                    self.assertEqual(store.config.tenant_id, tenant)
                    # The fake master deliberately ignores tenants, as a default
                    # Mooncake master does. Only object keys can isolate values.
                    store.store.objects = objects
                    self.assertEqual(store.batch_exists(["page0"]), 0)
                    for key in store._tag_keys(["page0_0_k", "page0_0_v"]):
                        store.store.put(key, b"payload")
                    self.assertEqual(store.batch_exists(["page0"]), 1)
                    stores.append(store)
                self.assertEqual(len(objects), 8)
                for store in stores:
                    self.assertEqual(store.batch_exists(["page0"]), 1)

    def test_mooncake_store_forwards_non_default_tenant_id(self):
        FakeMooncakeDistributedStore.instances = []
        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(FakeMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            MooncakeStore(_make_storage_config("tenant-a"))

        fake_store = FakeMooncakeDistributedStore.instances[-1]
        self.assertEqual(fake_store.setup_calls[0][1]["tenant_id"], "tenant-a")

    def test_mooncake_store_keeps_default_tenant_backward_compatible(self):
        FakeMooncakeDistributedStore.instances = []
        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(FakeMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            MooncakeStore(_make_storage_config())

        fake_store = FakeMooncakeDistributedStore.instances[-1]
        self.assertNotIn("tenant_id", fake_store.setup_calls[0][1])

    def test_non_default_tenant_requires_new_mooncake(self):
        OldMooncakeDistributedStore.instances = []
        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(OldMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            with self.assertRaisesRegex(RuntimeError, "tenant_id"):
                MooncakeStore(_make_storage_config("tenant-a"))

    def test_embedding_store_forwards_tenant_id(self):
        FakeMooncakeDistributedStore.instances = []
        storage_config = SimpleNamespace(
            extra_config={
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": "tenant-embedding",
            }
        )

        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(FakeMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
                MooncakeEmbeddingStore,
            )

            MooncakeEmbeddingStore(storage_config)

        fake_store = FakeMooncakeDistributedStore.instances[-1]
        self.assertEqual(fake_store.setup_calls[0][1]["tenant_id"], "tenant-embedding")


if __name__ == "__main__":
    unittest.main(verbosity=2)
