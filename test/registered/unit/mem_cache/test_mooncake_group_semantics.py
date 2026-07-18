import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class ReplicateConfigWithGroupIds:
    def __init__(self):
        self.group_ids = None


class ReplicateConfigWithoutGroupIds:
    __slots__ = ()


class ReplicateConfigWithClassGroupIdsAndRequiredInit:
    group_ids = None

    def __init__(self, required):
        self.group_ids = required


def _fake_mooncake_modules(fake_store_cls, replicate_config_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    mooncake_store.ReplicateConfig = replicate_config_cls
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
    }


def _fake_pool_host_mla_module():
    pool_host_mla = types.ModuleType("sglang.srt.mem_cache.pool_host.mla")

    class MLATokenToKVPoolHost:
        pass

    pool_host_mla.MLATokenToKVPoolHost = MLATokenToKVPoolHost
    return pool_host_mla


def _fake_pool_host_module():
    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")

    class HostKVCache:
        pass

    class HostTensorAllocator:
        pass

    pool_host.HostKVCache = HostKVCache
    pool_host.HostTensorAllocator = HostTensorAllocator
    return pool_host


def _fake_host_pool_modules():
    return {
        "sglang.srt.mem_cache.pool_host": _fake_pool_host_module(),
        "sglang.srt.mem_cache.pool_host.mla": _fake_pool_host_mla_module(),
    }


def _fake_store_class():
    class FakeMooncakeDistributedStore:
        instances = []

        def __init__(self):
            self.batch_put_calls = []
            self.existing_keys = set()
            self.objects = {}
            type(self).instances.append(self)

        def setup(self, *args, **kwargs):
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
            self.batch_put_calls.append(
                {
                    "method": "batch_put_from",
                    "keys": list(keys),
                    "ptrs": list(ptrs),
                    "sizes": list(sizes),
                    "args": args,
                }
            )
            self.existing_keys.update(keys)
            return [0] * len(keys)

        def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
            self.batch_put_calls.append(
                {
                    "method": "batch_put_from_multi_buffers",
                    "keys": list(keys),
                    "ptrs": list(ptrs),
                    "sizes": list(sizes),
                    "args": args,
                }
            )
            self.existing_keys.update(keys)
            return [0] * len(keys)

    return FakeMooncakeDistributedStore


class FakeHostKVCache:
    def __init__(self, objects_per_page):
        self.objects_per_page = objects_per_page
        self.kv_buffer = torch.empty((1024,), dtype=torch.uint8)
        self.layout = "page_first"
        self.page_size = 1

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        page_count = len(indices) // self.page_size
        ptrs = []
        sizes = []
        for page_idx in range(page_count):
            for object_idx in range(self.objects_per_page):
                ptrs.append(1000 + page_idx * 100 + object_idx)
                sizes.append(8)
        return ptrs, sizes

    def get_split_heads_page_buffer_meta(self, indices, split_factor):
        page_count = len(indices) // self.page_size
        ptrs = []
        sizes = []
        for page_idx in range(page_count):
            for object_idx in range(2 * split_factor):
                ptrs.append(2000 + page_idx * 100 + object_idx)
                sizes.append(8)
        return ptrs, sizes


class FakeIndexerPool:
    page_size = 1

    def __init__(self):
        self.buffer = torch.empty((128,), dtype=torch.uint8)

    def get_hybrid_pool_buffer(self):
        return [self.buffer]

    def get_page_buffer_meta(self, indices):
        return [3000 + i for i in range(len(indices))], [8] * len(indices)


class FakeMultiBufferPool:
    page_size = 1

    def __init__(self):
        self.buffers = [
            torch.empty((128,), dtype=torch.uint8),
            torch.empty((128,), dtype=torch.uint8),
        ]

    def get_hybrid_pool_buffer(self):
        return self.buffers

    def get_page_buffer_meta(self, indices):
        ptrs = []
        sizes = []
        for i in range(len(indices)):
            ptrs.extend([4000 + i * 10, 4001 + i * 10])
            sizes.extend([8, 16])
        return ptrs, sizes


def _make_config(
    *,
    enable_group_semantics=True,
    extra_backend_tag=None,
    is_mla_model=False,
    should_split_heads=False,
    tp_rank=0,
    tp_size=1,
    tp_lcm_size=None,
):
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
        "enable_group_semantics": enable_group_semantics,
    }
    if extra_backend_tag is not None:
        extra_config["extra_backend_tag"] = extra_backend_tag

    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        tp_lcm_size=tp_lcm_size,
        should_split_heads=should_split_heads,
        extra_config=extra_config,
    )


def _make_store(
    *,
    enable_group_semantics=True,
    replicate_config_cls=ReplicateConfigWithGroupIds,
    extra_backend_tag=None,
    is_mla_model=False,
    should_split_heads=False,
    tp_rank=0,
    tp_size=1,
    tp_lcm_size=None,
):
    fake_store_cls = _fake_store_class()
    cfg = _make_config(
        enable_group_semantics=enable_group_semantics,
        extra_backend_tag=extra_backend_tag,
        is_mla_model=is_mla_model,
        should_split_heads=should_split_heads,
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_lcm_size=tp_lcm_size,
    )

    with patch.dict(
        "sys.modules",
        {
            **_fake_mooncake_modules(fake_store_cls, replicate_config_cls),
            **_fake_host_pool_modules(),
        },
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(cfg)

    return store, fake_store_cls.instances[-1]


class TestMooncakeGroupSemantics(CustomTestCase):
    def test_v2_rejects_partial_trailing_window(self):
        store, _ = _make_store(enable_group_semantics=False)
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))
        store.register_mem_host_pool_v2(
            FakeHostKVCache(objects_per_page=2), PoolName.SWA
        )
        store.batch_exists = lambda keys, extra_info=None: len(keys)
        store._get_hybrid_page_component_keys = lambda keys, transfer: (keys, 1)
        store._batch_exist = lambda keys: [1] * len(keys)

        result = store.batch_exists_v2(
            ["page0", "page1"],
            [
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=["window0", "window1", "window2"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ],
        )

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertNotIn(PoolName.SWA, result.extra_pool_hit_pages)

    def test_group_id_detection_uses_class_attribute_without_instantiating(self):
        fake_store_cls = _fake_store_class()
        with patch.dict(
            "sys.modules",
            {
                **_fake_mooncake_modules(
                    fake_store_cls,
                    ReplicateConfigWithClassGroupIdsAndRequiredInit,
                ),
                **_fake_host_pool_modules(),
            },
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeBaseStore,
            )

            replicate_config_cls, supports_group_ids = (
                MooncakeBaseStore()._import_mooncake_group_semantics()
            )

        self.assertIs(
            replicate_config_cls, ReplicateConfigWithClassGroupIdsAndRequiredInit
        )
        self.assertTrue(supports_group_ids)

    def test_flag_off_uses_three_arg_batch_put(self):
        store, fake_store = _make_store(enable_group_semantics=False)
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))

        result = store.batch_set_v1(["page0"], torch.tensor([0]))

        self.assertEqual(result, [True])
        self.assertEqual(len(fake_store.batch_put_calls), 1)
        self.assertEqual(
            fake_store.batch_put_calls[0]["keys"], ["page0_0_k", "page0_0_v"]
        )
        self.assertEqual(fake_store.batch_put_calls[0]["args"], ())

    def test_mha_group_ids_use_tagged_logical_page_key(self):
        store, fake_store = _make_store(extra_backend_tag="tag")
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))

        result = store.batch_set_v1(["page0", "page1"], torch.tensor([0, 1]))

        self.assertEqual(result, [True, True])
        call = fake_store.batch_put_calls[0]
        self.assertEqual(
            call["keys"],
            [
                "tag_page0_0_k",
                "tag_page0_0_v",
                "tag_page1_0_k",
                "tag_page1_0_v",
            ],
        )
        self.assertEqual(len(call["args"]), 1)
        self.assertEqual(
            call["args"][0].group_ids,
            [
                "sglang-hicache:tag_page0",
                "sglang-hicache:tag_page0",
                "sglang-hicache:tag_page1",
                "sglang-hicache:tag_page1",
            ],
        )

    def test_old_mooncake_falls_back_to_three_arg_batch_put(self):
        store, fake_store = _make_store(
            enable_group_semantics=True,
            replicate_config_cls=ReplicateConfigWithoutGroupIds,
        )
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))

        result = store.batch_set_v1(["page0"], torch.tensor([0]))

        self.assertEqual(result, [True])
        self.assertEqual(fake_store.batch_put_calls[0]["args"], ())

    def test_mla_group_ids(self):
        store, fake_store = _make_store(is_mla_model=True)
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=1))

        result = store.batch_set_v1(["page0"], torch.tensor([0]))

        self.assertEqual(result, [True])
        call = fake_store.batch_put_calls[0]
        self.assertEqual(call["keys"], ["page0__k"])
        self.assertEqual(call["args"][0].group_ids, ["sglang-hicache:page0"])

    def test_split_heads_group_ids(self):
        store, fake_store = _make_store(
            should_split_heads=True,
            tp_rank=1,
            tp_size=2,
            tp_lcm_size=4,
        )
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=4))

        result = store.batch_set_v1(["page0"], torch.tensor([0]))

        self.assertEqual(result, [True])
        call = fake_store.batch_put_calls[0]
        self.assertEqual(
            call["keys"],
            ["page0_2_k", "page0_2_v", "page0_3_k", "page0_3_v"],
        )
        self.assertEqual(
            call["args"][0].group_ids,
            ["sglang-hicache:page0"] * 4,
        )

    def test_existing_filter_keeps_group_ids_aligned_with_missing_keys(self):
        store, fake_store = _make_store()
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))
        fake_store.existing_keys.update({"page0_0_k", "page1_0_v"})

        result = store.batch_set_v1(["page0", "page1"], torch.tensor([0, 1]))

        self.assertEqual(result, [True, True])
        call = fake_store.batch_put_calls[0]
        self.assertEqual(call["keys"], ["page0_0_v", "page1_0_k"])
        self.assertEqual(
            call["args"][0].group_ids,
            ["sglang-hicache:page0", "sglang-hicache:page1"],
        )

    def test_v2_indexer_group_ids_use_logical_page_key(self):
        store, fake_store = _make_store(extra_backend_tag="tag", is_mla_model=True)
        indexer_pool = FakeIndexerPool()
        store.register_mem_host_pool_v2(indexer_pool, PoolName.INDEXER)

        result = store.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.INDEXER,
                    keys=["page0", "page1"],
                    host_indices=torch.tensor([0, 1]),
                )
            ]
        )

        self.assertEqual(result[PoolName.INDEXER], [True, True])
        call = fake_store.batch_put_calls[0]
        self.assertEqual(call["keys"], ["tag_page0__indexer", "tag_page1__indexer"])
        self.assertEqual(
            call["args"][0].group_ids,
            ["sglang-hicache:tag_page0", "sglang-hicache:tag_page1"],
        )

    def test_v2_multi_buffer_put_passes_group_ids(self):
        store, fake_store = _make_store(extra_backend_tag="tag", is_mla_model=True)
        multi_buffer_pool = FakeMultiBufferPool()
        store.register_mem_host_pool_v2(multi_buffer_pool, PoolName.DEEPSEEK_V4_C4)

        result = store.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C4,
                    keys=["page0", "page1"],
                    host_indices=torch.tensor([0, 1]),
                )
            ]
        )

        self.assertEqual(result[PoolName.DEEPSEEK_V4_C4], [True, True])
        call = fake_store.batch_put_calls[0]
        self.assertEqual(call["method"], "batch_put_from_multi_buffers")
        self.assertEqual(
            call["keys"],
            ["tag_page0__deepseek_v4_c4", "tag_page1__deepseek_v4_c4"],
        )
        self.assertEqual(call["ptrs"], [[4000, 4001], [4010, 4011]])
        self.assertEqual(call["sizes"], [[8, 16], [8, 16]])
        self.assertEqual(
            call["args"][0].group_ids,
            ["sglang-hicache:tag_page0", "sglang-hicache:tag_page1"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
