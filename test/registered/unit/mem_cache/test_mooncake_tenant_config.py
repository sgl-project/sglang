import json
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    DEFAULT_TENANT_ID,
    MooncakeStoreConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


class OldMooncakeDistributedStore(FakeMooncakeDistributedStore):
    instances = []

    def setup(self, *args, **kwargs):
        if "tenant_id" in kwargs:
            raise TypeError("tenant_id is an invalid keyword argument")
        return super().setup(*args, **kwargs)


def _make_storage_config(tenant_id=DEFAULT_TENANT_ID, kv_cache_dtype=None):
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
        kv_cache_dtype=kv_cache_dtype,
    )


class TestMooncakeTenantConfig(unittest.TestCase):
    def test_load_from_extra_config_normalizes_tenant(self):
        cfg = MooncakeStoreConfig.load_from_extra_config(
            {
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": "  tenant-extra  ",
            }
        )
        self.assertEqual(cfg.tenant_id, "tenant-extra")

        cfg = MooncakeStoreConfig.load_from_extra_config(
            {
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": "   ",
            }
        )
        self.assertEqual(cfg.tenant_id, DEFAULT_TENANT_ID)

        cfg = MooncakeStoreConfig.load_from_extra_config(
            {
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": None,
            }
        )
        self.assertEqual(cfg.tenant_id, DEFAULT_TENANT_ID)

    def test_load_from_env_reads_mooncake_tenant_id(self):
        with (
            envs.MOONCAKE_MASTER.override("127.0.0.1:50051"),
            envs.MOONCAKE_TENANT_ID.override("tenant-env"),
        ):
            cfg = MooncakeStoreConfig.load_from_env()

        self.assertEqual(cfg.tenant_id, "tenant-env")

    def test_load_from_file_reads_tenant_id(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "tenant_id": "tenant-file",
                },
                config_file,
            )
            config_file.flush()

            with envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(config_file.name):
                cfg = MooncakeStoreConfig.from_file()

        self.assertEqual(cfg.tenant_id, "tenant-file")

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

            for tenant, dtype in (("tenant-a", None), ("default", "bfloat16")):
                with (
                    self.subTest(tenant=tenant, dtype=dtype),
                    self.assertRaisesRegex(
                        RuntimeError, r"mooncake-transfer-engine>=0\.3\.12"
                    ),
                ):
                    MooncakeStore(_make_storage_config(tenant, dtype))

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

    def test_dtype_namespaces_preserve_key_prefixes(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        cases = (
            ("default", "bfloat16", "dtype_bfloat16"),
            ("default", "float16", "dtype_float16"),
            ("default", "float8_e4m3fn", "dtype_float8_e4m3fn"),
            ("default", "float8_e5m2", "dtype_float8_e5m2"),
            ("tenant-a", "bfloat16", "tenant-a_dtype_bfloat16"),
        )
        for tenant, dtype, expected in cases:
            with (
                self.subTest(tenant=tenant, dtype=dtype),
                patch.dict(
                    "sys.modules", _fake_mooncake_modules(FakeMooncakeDistributedStore)
                ),
            ):
                cfg = _make_storage_config(tenant, dtype)
                cfg.extra_config["extra_backend_tag"] = "prod"
                store = MooncakeStore(cfg)
                self.assertEqual(store.store.setup_calls[0][1]["tenant_id"], expected)
                with patch.object(
                    store.store, "batch_is_exist", return_value=[1, 1], create=True
                ) as exists:
                    self.assertEqual(store.batch_exists(["page0"]), 1)
                    exists.assert_called_once_with(
                        ["prod_test_page0_0_k", "prod_test_page0_0_v"]
                    )

    def test_ssd_fallback_preserves_dtype_tenant(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        cfg = _make_storage_config(kv_cache_dtype="bfloat16")
        cfg.extra_config["enable_ssd_offload"] = True
        with (
            patch.dict(
                "sys.modules", _fake_mooncake_modules(FakeMooncakeDistributedStore)
            ),
            patch.object(
                FakeMooncakeDistributedStore,
                "setup",
                side_effect=[
                    TypeError("unexpected keyword argument 'enable_ssd_offload'"),
                    0,
                ],
            ) as setup,
        ):
            MooncakeStore(cfg)
        self.assertEqual(setup.call_count, 2)
        self.assertEqual(setup.call_args.kwargs, {"tenant_id": "dtype_bfloat16"})

    def test_controller_uses_host_storage_dtype(self):
        from sglang.srt.managers import cache_controller
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        controller = SimpleNamespace(
            enable_storage_metrics=False,
            # The allocation facade does not need a dtype; its storage anchor
            # owns the persisted representation.
            mem_pool_host=SimpleNamespace(layout="page_first"),
            mem_pool_device=SimpleNamespace(dtype=torch.float16),
            get_attn_cp_rank_and_size=lambda: (0, 1),
        )
        parallel = SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
        cases = (
            # A typed host representation wins over the device value dtype.
            (torch.bfloat16, torch.float8_e4m3fn, "bfloat16"),
            (torch.float8_e4m3fn, torch.bfloat16, "float8_e4m3fn"),
            (torch.uint8, torch.bfloat16, "uint8"),
            # Raw FP8 bytes still require different interpretation.
            (torch.uint8, torch.float8_e4m3fn, "float8_e4m3fn"),
            (torch.uint8, torch.float8_e5m2, "float8_e5m2"),
            (torch.uint8, torch.float8_e4m3fnuz, "float8_e4m3fnuz"),
        )
        with (
            patch.object(cache_controller, "get_parallel", return_value=parallel),
            patch.object(
                cache_controller, "is_dp_attention_enabled", return_value=False
            ),
        ):
            for pool_type in (MHATokenToKVPoolHost, MLATokenToKVPoolHost):
                for buffer_dtype, value_dtype, expected in cases:
                    with self.subTest(pool_type=pool_type, expected=expected):
                        # Exercise the real host dtype contract without allocating
                        # GPU memory or host transfer buffers.
                        host_pool = pool_type.__new__(pool_type)
                        host_pool.dtype = buffer_dtype
                        host_pool.device_pool = SimpleNamespace(
                            dtype=value_dtype, store_dtype=buffer_dtype
                        )
                        controller.storage_host_pool = host_pool
                        config = (
                            cache_controller.HiCacheController._generate_storage_config(
                                controller
                            )
                        )
                        self.assertEqual(config.kv_cache_dtype, expected)

    def test_controller_accepts_logical_storage_anchor(self):
        from sglang.srt.managers import cache_controller
        from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool

        controller = SimpleNamespace(
            enable_storage_metrics=False,
            mem_pool_host=SimpleNamespace(layout="page_first"),
            mem_pool_device=SimpleNamespace(dtype=torch.float8_e4m3fn),
            storage_host_pool=LogicalHostPool(size=16, page_size=1),
            get_attn_cp_rank_and_size=lambda: (0, 1),
        )
        parallel = SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
        with (
            patch.object(cache_controller, "get_parallel", return_value=parallel),
            patch.object(
                cache_controller, "is_dp_attention_enabled", return_value=False
            ),
        ):
            config = cache_controller.HiCacheController._generate_storage_config(
                controller
            )
        self.assertIsNone(config.kv_cache_dtype)

    def test_page_envelope_reports_persisted_value_dtype(self):
        from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool

        for store_dtype, value_dtype, expected in (
            (torch.bfloat16, torch.float16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.bfloat16, torch.float8_e4m3fn),
            (torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fn),
            (torch.uint8, torch.float8_e5m2, torch.float8_e5m2),
        ):
            with self.subTest(store_dtype=store_dtype, value_dtype=value_dtype):
                host_pool = UnifiedPageEnvelopeHostPool.__new__(
                    UnifiedPageEnvelopeHostPool
                )
                host_pool.dtype = torch.uint8
                host_pool.device_pool = SimpleNamespace(
                    dtype=value_dtype, store_dtype=store_dtype
                )
                self.assertEqual(host_pool.get_storage_dtype(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
