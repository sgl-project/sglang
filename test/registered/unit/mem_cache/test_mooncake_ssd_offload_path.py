import os
import tempfile
import types
import unittest
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fake_store_class():
    class FakeMooncakeDistributedStore:
        instances = []

        def __init__(self):
            self.setup_kwargs = None
            self.objects = {}
            type(self).instances.append(self)

        def setup(self, *args, **kwargs):
            self.setup_kwargs = kwargs
            return 0

        def register_buffer(self, *args, **kwargs):
            return 0

        def put(self, key, value, *args):
            self.objects[key] = value
            return 0

        def get(self, key):
            return self.objects.get(key)

        def is_exist(self, key):
            return 1 if key in self.objects else 0

        def batch_is_exist(self, keys):
            return [1 if key in self.objects else 0 for key in keys]

        def remove(self, key):
            self.objects.pop(key, None)
            return 0

    return FakeMooncakeDistributedStore


def _fake_mooncake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
    }


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


def _make_config(*, tp_rank, pp_rank, ssd_offload_path, dp_rank=0):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=8,
        pp_rank=pp_rank,
        pp_size=1,
        dp_rank=dp_rank,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "global_segment_size": 1024 * 1024,
            "enable_ssd_offload": True,
            "ssd_offload_path": ssd_offload_path,
        },
    )


def _make_store(*, tp_rank, pp_rank, ssd_offload_path, dp_rank=0):
    fake_store_cls = _fake_store_class()
    cfg = _make_config(
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        ssd_offload_path=ssd_offload_path,
        dp_rank=dp_rank,
    )
    with patch.dict(
        "sys.modules",
        {
            **_fake_mooncake_modules(fake_store_cls),
            **_fake_host_pool_modules(),
        },
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        MooncakeStore(cfg)
    return fake_store_cls.instances[-1]


class TestMooncakeSsdOffloadPath(CustomTestCase):
    def test_each_rank_gets_private_subdirectory(self):
        with tempfile.TemporaryDirectory() as base:
            for tp_rank in (0, 3):
                client = _make_store(tp_rank=tp_rank, pp_rank=0, ssd_offload_path=base)
                expected = os.path.join(base, f"rank_0_{tp_rank}_0")
                self.assertEqual(client.setup_kwargs.get("ssd_offload_path"), expected)
                self.assertTrue(os.path.isdir(expected))

    def test_rank_directories_are_distinct(self):
        with tempfile.TemporaryDirectory() as base:
            paths = {
                _make_store(
                    tp_rank=tp_rank, pp_rank=0, ssd_offload_path=base
                ).setup_kwargs["ssd_offload_path"]
                for tp_rank in range(4)
            }
            self.assertEqual(len(paths), 4)

    def test_dp_ranks_are_distinct_when_attn_tp_rank_is_zero(self):
        # dp-attention with attn_tp_size 1: every DP rank reports tp_rank 0
        with tempfile.TemporaryDirectory() as base:
            paths = {
                _make_store(
                    tp_rank=0, pp_rank=0, ssd_offload_path=base, dp_rank=dp_rank
                ).setup_kwargs["ssd_offload_path"]
                for dp_rank in range(8)
            }
            self.assertEqual(len(paths), 8)
            self.assertIn(os.path.join(base, "rank_5_0_0"), paths)


if __name__ == "__main__":
    unittest.main()
