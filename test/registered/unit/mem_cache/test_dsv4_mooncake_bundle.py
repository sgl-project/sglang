"""Native persistent pools restore one complete producer generation."""

import ctypes
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NAMES = (
    PoolName.DEEPSEEK_V4_C1,
    PoolName.DEEPSEEK_V4_C1_INDEXER,
    PoolName.DEEPSEEK_V4_C2,
    PoolName.DEEPSEEK_V4_C2_INDEXER,
)
PLAN = SimpleNamespace(pool_names=NAMES, fingerprint="a" * 64)


class CpuPool:
    page_size = 2

    def __init__(self):
        self.buffers = [(ctypes.c_ubyte * 32)() for _ in range(2)]

    def fill(self, value):
        for buffer in self.buffers:
            ctypes.memset(ctypes.addressof(buffer), value, len(buffer))

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
        for first in indices.reshape(-1, self.page_size)[:, 0].tolist():
            for buffer in self.buffers:
                ptrs.append(ctypes.addressof(buffer) + first // self.page_size * 16)
                sizes.append(16)
        return ptrs, sizes

    def page(self):
        return b"".join(bytes(buffer)[:16] for buffer in self.buffers)


class ImmutableSdk:
    """Only the external SDK is replaced; real bytes cross its pointer API."""

    def __init__(self):
        self.objects = {}
        self.fail_put_at = None
        self.short_get = False

    def put(self, key, value, *args):
        self.objects.setdefault(key, bytes(value))
        return 0

    def get(self, key):
        return self.objects.get(key, b"")

    def get_size(self, key):
        return len(self.get(key))

    def remove(self, key, force=False):
        self.objects.pop(key, None)
        return 0

    def batch_is_exist(self, keys):
        return [int(key in self.objects) for key in keys]

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
        result = []
        for i, (key, pointers, lengths) in enumerate(
            zip(keys, ptrs, sizes, strict=True)
        ):
            if i == self.fail_put_at:
                result.append(-1)
                continue
            self.put(
                key,
                b"".join(
                    ctypes.string_at(p, n)
                    for p, n in zip(pointers, lengths, strict=True)
                ),
            )
            result.append(0)
        return result

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        result = []
        for key, pointers, lengths in zip(keys, ptrs, sizes, strict=True):
            value = self.get(key)
            if self.short_get:
                value = value[:-1]
            offset = 0
            for pointer, length in zip(pointers, lengths, strict=True):
                part = value[offset : offset + length]
                if part:
                    ctypes.memmove(pointer, part, len(part))
                offset += length
            result.append(len(value))
        return result


def backend(sdk=None, prefix="test"):
    cache = object.__new__(MooncakeStore)
    cache.store = sdk if sdk is not None else ImmutableSdk()
    cache.registered_pools = {name: CpuPool() for name in NAMES}
    cache.mem_pool_host = SimpleNamespace(kv_buffer=None)
    cache.config_prefix, cache.mla_suffix = prefix, "0"
    cache.is_mla_backend = True
    cache._use_group_semantics = cache._supports_session_ids = False
    cache._replicate_config_cls = SimpleNamespace
    cache.configure_persistent_pool_bundle(PLAN)
    return cache


def transfers():
    return [
        PoolTransfer(name=name, keys=["page0"], host_indices=torch.arange(2))
        for name in NAMES
    ]


def fill(cache, value):
    for pool in cache.registered_pools.values():
        pool.fill(value)


def all_ok(result):
    return set(result) == set(NAMES) and all(
        value == [True] for value in result.values()
    )


class TestDsv4MooncakeBundle(CustomTestCase):
    def test_partial_generation_repair_restores_all_pools_in_a_new_consumer(self):
        cache = backend()
        fill(cache, 11)
        self.assertTrue(all_ok(cache.batch_set_v2(transfers())))
        old = cache._load_persistent_manifest("page0")
        spec = cache._persistent_pool_bundle
        key = spec.object_key(
            "page0",
            cache._persistent_pool_bundle_namespace,
            old["generation"],
            str(NAMES[-1]),
        )
        cache.store.remove(key, force=True)
        self.assertEqual(cache.batch_exists_v2(["page0"], transfers()).kv_hit_pages, 0)
        fill(cache, 22)
        self.assertTrue(all_ok(cache.batch_set_v2(transfers())))
        self.assertNotEqual(
            old["generation"], cache._load_persistent_manifest("page0")["generation"]
        )
        reader = backend(cache.store)
        self.assertTrue(all_ok(reader.batch_get_v2(transfers())))
        self.assertEqual(
            [p.page() for p in reader.registered_pools.values()], [bytes([22]) * 32] * 4
        )

    def test_failed_generation_never_publishes(self):
        cache = backend()
        cache.store.fail_put_at = 2
        fill(cache, 11)
        result = cache.batch_set_v2(transfers())
        self.assertEqual(result, {name: [False] for name in NAMES})
        self.assertIsNone(cache._load_persistent_manifest("page0"))
        self.assertEqual(cache.batch_exists_v2(["page0"], transfers()).kv_hit_pages, 0)

    def test_short_read_refuses_every_pool(self):
        cache = backend()
        fill(cache, 11)
        self.assertTrue(all_ok(cache.batch_set_v2(transfers())))
        reader = backend(cache.store)
        cache.store.short_get = True
        self.assertEqual(
            reader.batch_get_v2(transfers()), {name: [False] for name in NAMES}
        )

    def test_namespaces_and_pool_members_are_not_interchangeable(self):
        cache = backend()
        fill(cache, 11)
        self.assertTrue(all_ok(cache.batch_set_v2(transfers())))
        other = backend(cache.store, prefix="other")
        self.assertEqual(other.batch_exists_v2(["page0"], transfers()).kv_hit_pages, 0)
        with self.assertRaises(ValueError):
            cache.batch_set_v2(transfers()[:-1])


if __name__ == "__main__":
    unittest.main()
