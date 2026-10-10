"""CPU regressions for dtype namespaces and Mooncake setup compatibility."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import cache_controller
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeBaseStore,
    MooncakeStore,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeKvCacheDtype(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.setup.return_value = 0
        self.client.batch_is_exist.return_value = [1, 1]
        self.config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name=None,
            extra_config={
                "master_server_address": "127.0.0.1:50051",
                "check_server": False,
                "global_segment_size": 1024 * 1024,
            },
        )
        for patcher in (
            patch.object(
                MooncakeBaseStore,
                "_import_mooncake_store",
                return_value=lambda: self.client,
            ),
            patch.object(
                MooncakeBaseStore,
                "_import_mooncake_group_semantics",
                return_value=(None, False),
            ),
            patch.object(MooncakeStore, "warmup"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_dtype_namespaces_preserve_object_keys(self):
        for tenant in ("default", "tenant-a"):
            for dtype in (None, "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"):
                with self.subTest(tenant=tenant, dtype=dtype):
                    self.config.kv_cache_dtype = dtype
                    self.config.extra_config["tenant_id"] = tenant
                    store = MooncakeStore(self.config)
                    expected = tenant
                    if dtype:
                        expected = (
                            f"dtype_{dtype}"
                            if tenant == "default"
                            else f"{tenant}_dtype_{dtype}"
                        )
                    self.assertEqual(
                        self.client.setup.call_args.kwargs.get("tenant_id", "default"),
                        expected,
                    )
                    self.assertEqual(store.batch_exists(["page0"]), 1)
                    self.client.batch_is_exist.assert_called_with(
                        ["page0_0_k", "page0_0_v"]
                    )

    def test_model_and_backend_tag_still_prefix_keys(self):
        self.config.model_name = "org/model"
        self.config.kv_cache_dtype = "bfloat16"
        self.config.extra_config["extra_backend_tag"] = "prod"
        store = MooncakeStore(self.config)
        self.assertEqual(store.batch_exists(["page0"]), 1)
        self.client.batch_is_exist.assert_called_once_with(
            ["prod_org-model_page0_0_k", "prod_org-model_page0_0_v"]
        )
        self.assertEqual(
            self.client.setup.call_args.kwargs["tenant_id"], "dtype_bfloat16"
        )

    def test_unsupported_tenant_fails_without_retry(self):
        self.config.kv_cache_dtype = "bfloat16"
        self.client.setup.side_effect = TypeError(
            "tenant_id is an invalid keyword argument"
        )
        with self.assertRaisesRegex(
            RuntimeError, r"mooncake-transfer-engine>=0\.3\.12"
        ):
            MooncakeStore(self.config)
        self.client.setup.assert_called_once()

    def test_ssd_fallback_preserves_tenant(self):
        self.config.kv_cache_dtype = "bfloat16"
        self.config.extra_config["enable_ssd_offload"] = True
        self.client.setup.side_effect = [
            TypeError("unexpected keyword argument 'enable_ssd_offload'"),
            0,
        ]
        MooncakeStore(self.config)
        self.assertEqual(self.client.setup.call_count, 2)
        self.assertEqual(
            self.client.setup.call_args.kwargs, {"tenant_id": "dtype_bfloat16"}
        )

    def test_unrelated_setup_type_error_is_preserved(self):
        self.client.setup.side_effect = TypeError("invalid buffer size")
        with self.assertRaisesRegex(TypeError, "invalid buffer size"):
            MooncakeStore(self.config)
        self.client.setup.assert_called_once()

    def test_controller_uses_logical_dtype_for_byte_backed_pools(self):
        controller = cache_controller.HiCacheController.__new__(
            cache_controller.HiCacheController
        )
        # Covers both a byte-backed host pool and a hybrid facade with no dtype.
        controller.enable_storage_metrics = False
        controller.get_attn_cp_rank_and_size = lambda: (0, 1)
        parallel = SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
        with (
            patch.object(cache_controller, "get_parallel", return_value=parallel),
            patch.object(
                cache_controller, "is_dp_attention_enabled", return_value=False
            ),
        ):
            for host_pool in (
                SimpleNamespace(layout="page_first", dtype=torch.uint8),
                SimpleNamespace(layout="page_first"),
            ):
                controller.mem_pool_host = host_pool
                for dtype, expected in (
                    (torch.bfloat16, "bfloat16"),
                    (torch.float8_e4m3fn, "float8_e4m3fn"),
                    (torch.float8_e5m2, "float8_e5m2"),
                ):
                    with self.subTest(host_pool=host_pool, dtype=dtype):
                        controller.mem_pool_device = SimpleNamespace(
                            dtype=dtype, store_dtype=torch.uint8
                        )
                        config = controller._generate_storage_config()
                        self.assertEqual(config.kv_cache_dtype, expected)


if __name__ == "__main__":
    unittest.main()
