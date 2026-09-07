"""Unit tests for Mooncake KV-cache dtype isolation via extra_config/tenant."""

import json
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    format_kv_cache_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def _fake_metrics_module():
    metrics = types.ModuleType("sglang.srt.observability.metrics_collector")

    class StorageMetrics:
        def __init__(self):
            self.prefetch_pgs = []
            self.backup_pgs = []
            self.prefetch_bandwidth = []
            self.backup_bandwidth = []

    metrics.StorageMetrics = StorageMetrics
    return {"sglang.srt.observability.metrics_collector": metrics}


def _fake_host_pool_modules():
    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")

    class HostKVCache:
        pass

    class HostTensorAllocator:
        pass

    pool_host.HostKVCache = HostKVCache
    pool_host.HostTensorAllocator = HostTensorAllocator

    pool_host_mla = types.ModuleType("sglang.srt.mem_cache.pool_host.mla")

    class MLATokenToKVPoolHost:
        pass

    pool_host_mla.MLATokenToKVPoolHost = MLATokenToKVPoolHost
    return {
        "sglang.srt.mem_cache.pool_host": pool_host,
        "sglang.srt.mem_cache.pool_host.mla": pool_host_mla,
    }


def _import_stubs(fake_store_cls):
    return {
        **_fake_mooncake_modules(fake_store_cls),
        **_fake_host_pool_modules(),
        **_fake_metrics_module(),
    }


def _fake_store_class():
    class FakeMooncakeDistributedStore:
        instances = []

        def __init__(self):
            self.setup_calls = []
            self.batch_put_calls = []
            self.existing_keys = set()
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
            return 1 if key in self.objects or key in self.existing_keys else 0

        def get(self, key):
            return self.objects.get(key)

        def batch_is_exist(self, keys):
            return [1 if key in self.existing_keys else 0 for key in keys]

        def batch_put_from(self, keys, ptrs, sizes, *args):
            self.batch_put_calls.append({"keys": list(keys), "args": args})
            self.existing_keys.update(keys)
            return [0] * len(keys)

    return FakeMooncakeDistributedStore


class OldMooncakeDistributedStore(_fake_store_class()):
    instances = []

    def setup(self, *args, **kwargs):
        if "tenant_id" in kwargs:
            raise TypeError("tenant_id is an invalid keyword argument")
        return super().setup(*args, **kwargs)


class FakeHostKVCache:
    def __init__(self):
        self.kv_buffer = torch.empty((1024,), dtype=torch.uint8)
        self.layout = "page_first"
        self.page_size = 1

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        page_count = len(indices) // self.page_size
        return (
            [1000 + i for i in range(page_count * 2)],
            [8] * (page_count * 2),
        )


def _make_config(
    *,
    extra_backend_tag=None,
    kv_cache_dtype=None,
    tenant_id=None,
):
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
    }
    if extra_backend_tag is not None:
        extra_config["extra_backend_tag"] = extra_backend_tag
    if tenant_id is not None:
        extra_config["tenant_id"] = tenant_id

    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        extra_config=extra_config,
        kv_cache_dtype=kv_cache_dtype,
    )


def _make_store(**kwargs):
    fake_store_cls = _fake_store_class()
    cfg = _make_config(**kwargs)
    with patch.dict(
        "sys.modules",
        _import_stubs(fake_store_cls),
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(cfg)
    store.register_mem_pool_host(FakeHostKVCache())
    return store, fake_store_cls.instances[-1]


class TestFormatKvCacheDtype(CustomTestCase):
    def test_formats_torch_dtype(self):
        self.assertEqual(format_kv_cache_dtype(torch.bfloat16), "bfloat16")
        self.assertEqual(format_kv_cache_dtype(torch.float8_e4m3fn), "float8_e4m3fn")

    def test_formats_string_and_none(self):
        self.assertEqual(format_kv_cache_dtype("torch.bfloat16"), "bfloat16")
        self.assertEqual(format_kv_cache_dtype("  fp8_e4m3  "), "fp8_e4m3")
        self.assertIsNone(format_kv_cache_dtype(None))
        self.assertIsNone(format_kv_cache_dtype("   "))


class TestMooncakeKvCacheDtypeIsolation(CustomTestCase):
    def test_missing_dtype_keeps_legacy_keys(self):
        store, fake_store = _make_store()
        result = store.batch_set_v1(["page0"], torch.tensor([0]))
        self.assertEqual(result, [True])
        self.assertEqual(
            fake_store.batch_put_calls[0]["keys"],
            ["page0_0_k", "page0_0_v"],
        )

    def test_dtype_is_isolated_by_tenant_id(self):
        store_bf16, fake_bf16 = _make_store(kv_cache_dtype="bfloat16")
        store_fp8, fake_fp8 = _make_store(kv_cache_dtype="fp8_e4m3")

        self.assertEqual(fake_bf16.setup_calls[0][1]["tenant_id"], "dtype_bfloat16")
        self.assertEqual(fake_fp8.setup_calls[0][1]["tenant_id"], "dtype_fp8_e4m3")

        store_bf16.batch_set_v1(["page0"], torch.tensor([0]))
        store_fp8.batch_set_v1(["page0"], torch.tensor([0]))

        self.assertEqual(
            fake_bf16.batch_put_calls[0]["keys"],
            ["page0_0_k", "page0_0_v"],
        )
        self.assertEqual(
            fake_fp8.batch_put_calls[0]["keys"],
            ["page0_0_k", "page0_0_v"],
        )

    def test_user_tag_is_independent_of_dtype_tenant(self):
        store, fake_store = _make_store(
            extra_backend_tag="prod", kv_cache_dtype="bfloat16"
        )
        self.assertEqual(fake_store.setup_calls[0][1]["tenant_id"], "dtype_bfloat16")
        store.batch_set_v1(["page0"], torch.tensor([0]))
        self.assertEqual(
            fake_store.batch_put_calls[0]["keys"],
            ["prod_page0_0_k", "prod_page0_0_v"],
        )

    def test_batch_exists_uses_unprefixed_keys_under_tenant(self):
        store, fake_store = _make_store(kv_cache_dtype="bfloat16")
        fake_store.existing_keys.update(["page0_0_k", "page0_0_v"])
        self.assertEqual(store.batch_exists(["page0"]), 1)

    def test_explicit_tenant_appends_dtype(self):
        store, fake_store = _make_store(
            tenant_id="tenant-a",
            kv_cache_dtype="bfloat16",
        )
        self.assertEqual(
            fake_store.setup_calls[0][1]["tenant_id"],
            "tenant-a_dtype_bfloat16",
        )
        store.batch_set_v1(["page0"], torch.tensor([0]))
        self.assertEqual(
            fake_store.batch_put_calls[0]["keys"],
            ["page0_0_k", "page0_0_v"],
        )

    def test_dtype_isolation_requires_mooncake_tenant_id(self):
        OldMooncakeDistributedStore.instances = []
        cfg = _make_config(kv_cache_dtype="bfloat16")
        with patch.dict(
            "sys.modules",
            _import_stubs(OldMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            with self.assertRaisesRegex(
                RuntimeError, "mooncake-transfer-engine>=0.3.12"
            ):
                MooncakeStore(cfg)


class TestMooncakeTenantConfig(CustomTestCase):
    def test_load_from_extra_config_normalizes_tenant(self):
        with patch.dict("sys.modules", _import_stubs(_fake_store_class())):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                DEFAULT_TENANT_ID,
                MooncakeStoreConfig,
            )

            cfg = MooncakeStoreConfig.load_from_extra_config(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "tenant_id": " tenant-extra ",
                }
            )
            self.assertEqual(cfg.tenant_id, "tenant-extra")

            cfg = MooncakeStoreConfig.load_from_extra_config(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "tenant_id": " ",
                }
            )
            self.assertEqual(cfg.tenant_id, DEFAULT_TENANT_ID)

    def test_load_from_env_reads_mooncake_tenant_id(self):
        with patch.dict("sys.modules", _import_stubs(_fake_store_class())):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStoreConfig,
            )

            with (
                envs.MOONCAKE_MASTER.override("127.0.0.1:50051"),
                envs.MOONCAKE_TENANT_ID.override("tenant-env"),
            ):
                cfg = MooncakeStoreConfig.load_from_env()
        self.assertEqual(cfg.tenant_id, "tenant-env")

    def test_load_from_file_reads_tenant_id(self):
        with patch.dict("sys.modules", _import_stubs(_fake_store_class())):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStoreConfig,
            )

            with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
                json.dump(
                    {
                        "master_server_address": "127.0.0.1:50051",
                        "tenant_id": "tenant-file",
                    },
                    config_file,
                )
                config_file.flush()
                with envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(
                    config_file.name
                ):
                    cfg = MooncakeStoreConfig.from_file()
        self.assertEqual(cfg.tenant_id, "tenant-file")

    def test_forwards_non_default_tenant_id(self):
        store, fake_store = _make_store(tenant_id="tenant-a")
        self.assertEqual(fake_store.setup_calls[0][1]["tenant_id"], "tenant-a")

    def test_default_tenant_is_not_passed_to_setup(self):
        store, fake_store = _make_store()
        self.assertNotIn("tenant_id", fake_store.setup_calls[0][1])

    def test_non_default_tenant_requires_new_mooncake(self):
        OldMooncakeDistributedStore.instances = []
        cfg = _make_config(tenant_id="tenant-a")
        with patch.dict(
            "sys.modules",
            _import_stubs(OldMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            with self.assertRaisesRegex(RuntimeError, "tenant_id"):
                MooncakeStore(cfg)

    def test_embedding_store_forwards_tenant_id(self):
        fake_store_cls = _fake_store_class()
        storage_config = types.SimpleNamespace(
            extra_config={
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": "tenant-embedding",
            }
        )
        with patch.dict("sys.modules", _import_stubs(fake_store_cls)):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
                MooncakeEmbeddingStore,
            )

            MooncakeEmbeddingStore(storage_config)
        fake_store = fake_store_cls.instances[-1]
        self.assertEqual(fake_store.setup_calls[0][1]["tenant_id"], "tenant-embedding")


class TestHiCacheStorageConfigDtype(CustomTestCase):
    def test_storage_config_carries_kv_cache_dtype(self):
        cfg = _make_config(kv_cache_dtype="bfloat16")
        self.assertEqual(cfg.kv_cache_dtype, "bfloat16")


if __name__ == "__main__":
    unittest.main()
