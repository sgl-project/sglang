"""radixshmem backend against a real in-process radix-server: SWA and Mamba
pools ride on the anchor path as radixshmem components."""

import hashlib
import unittest
import uuid

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

try:
    import radixshmem
    from radixshmem import RadixServer, ServerConfig

    HAVE_RADIXSHMEM = hasattr(radixshmem, "_data")
except ImportError:
    HAVE_RADIXSHMEM = False

PAGE = 4


class _HostPool:
    def __init__(self, num_pages, page_bytes, page_size=PAGE):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self.buf = torch.zeros(num_pages * page_bytes, dtype=torch.uint8)

    def get_page_buffer_meta(self, indices):
        starts = indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        ptrs = [self.buf.data_ptr() + int(p) * self.page_bytes for p in starts]
        return ptrs, [self.page_bytes] * len(ptrs)

    def page(self, p):
        return self.buf[p * self.page_bytes : (p + 1) * self.page_bytes]


def _keys(n, seed="k"):
    return [hashlib.sha256(f"{seed}{i}".encode()).hexdigest() for i in range(n)]


def _indices(pages, page_size=PAGE):
    return torch.arange(pages * page_size, dtype=torch.int64)


def _info(prefix):
    return HiCacheStorageExtraInfo(prefix_keys=list(prefix))


@unittest.skipUnless(HAVE_RADIXSHMEM, "radixshmem with a data plane is not installed")
class TestRadixShmemComponents(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.storage.radixshmem import RadixShmemStorage

        tag = uuid.uuid4().hex[:8]
        self.name, self.endpoint = f"/rsc_{tag}", f"unix:///tmp/rsc_{tag}.sock"
        self.server = RadixServer(
            ServerConfig(
                name=self.name,
                endpoint=self.endpoint,
                data_bytes=8 << 20,
                swa_ratio=0.25,
                mamba_ratio=0.25,
                slot_align=64,
                prefault=False,
                background_evict_ratio=0.0,
            )
        ).start()
        self.cls = RadixShmemStorage

    def tearDown(self):
        self.server.close()

    def _backend(self, window=2, **pools):
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="m",
            extra_config={
                "radixshmem_name": self.name,
                "radixshmem_endpoint": self.endpoint,
                "radixshmem_swa_window_pages": window,
            },
        )
        kv = _HostPool(16, 64)
        storage = self.cls(config, kv)
        storage.register_mem_pool_host(kv)
        for name, pool in pools.items():
            storage.register_mem_host_pool_v2(pool, PoolName(name))
        return storage, kv

    def test_geometry_per_component(self):
        swa, mamba = _HostPool(16, 32), _HostPool(4, 256, page_size=1)
        storage, _ = self._backend(swa=swa, mamba=mamba)
        storage.batch_exists(_keys(1), _info([]))
        pools = storage.client.geometry["pools"]
        self.assertEqual(pools["full"]["slot_bytes"], 64)
        self.assertEqual(pools["swa"]["slot_bytes"], 32)
        self.assertEqual(pools["swa"]["window_blocks"], 2)
        self.assertEqual(pools["mamba"]["slot_bytes"], 256)
        storage.close()

    def test_swa_window(self):
        swa = _HostPool(16, 32)
        storage, kv = self._backend(swa=swa)
        keys = _keys(5)
        self.assertEqual(storage.batch_set_v1(keys, _indices(5), _info([])), [True] * 5)
        swa.page(0).fill_(3)
        swa.page(1).fill_(4)
        window = PoolTransfer(
            name=PoolName.SWA,
            host_indices=_indices(2),
            keys=keys[3:5],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        self.assertEqual(
            storage.batch_set_v2([window], _info(keys[:3])), {PoolName.SWA: [True] * 2}
        )
        probe = PoolTransfer(
            name=PoolName.SWA, keys=["x"] * 2, hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        result = storage.batch_exists_v2(keys, [probe], _info([]))
        self.assertEqual(result.kv_hit_pages, 5)
        self.assertEqual(result.restorable_prefix_pages, [5])
        self.assertEqual(
            storage.batch_exists_v2(keys[:4], [probe], _info([])).kv_hit_pages, 0
        )

        swa.buf.zero_()
        got = storage.batch_get_v2([window], _info(keys[:3]))
        self.assertEqual(got, {PoolName.SWA: [True] * 2})
        self.assertTrue(torch.all(swa.page(0) == 3))
        self.assertTrue(torch.all(swa.page(1) == 4))
        storage.close()

    def test_mamba_checkpoint(self):
        mamba = _HostPool(4, 256, page_size=1)
        storage, _ = self._backend(mamba=mamba)
        keys = _keys(4)
        storage.batch_set_v1(keys, _indices(4), _info([]))
        mamba.page(0).fill_(9)
        state = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([0]),
            keys=keys[3:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        self.assertEqual(
            storage.batch_set_v2([state], _info(keys[:3])), {PoolName.MAMBA: [True]}
        )
        probe = PoolTransfer(
            name=PoolName.MAMBA, keys=["x"], hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        self.assertEqual(
            storage.batch_exists_v2(keys, [probe], _info([])).kv_hit_pages, 4
        )
        self.assertEqual(
            storage.batch_exists_v2(keys[:3], [probe], _info([])).kv_hit_pages, 0
        )
        mamba.buf.zero_()
        self.assertEqual(
            storage.batch_get_v2([state], _info(keys[:3])), {PoolName.MAMBA: [True]}
        )
        self.assertTrue(torch.all(mamba.page(0) == 9))
        state.keys = keys[2:3]
        self.assertEqual(
            storage.batch_get_v2([state], _info(keys[:2])), {PoolName.MAMBA: [False]}
        )
        storage.close()

    def test_component_before_kv_is_deferred(self):
        mamba = _HostPool(4, 256, page_size=1)
        storage, _ = self._backend(mamba=mamba)
        keys = _keys(3)
        mamba.page(0).fill_(6)
        state = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([0]),
            keys=keys[2:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        self.assertEqual(
            storage.batch_set_v2([state], _info(keys[:2])), {PoolName.MAMBA: [True]}
        )
        probe = PoolTransfer(
            name=PoolName.MAMBA, keys=["x"], hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        self.assertEqual(
            storage.batch_exists_v2(keys, [probe], _info([])).kv_hit_pages, 0
        )
        self.assertEqual(len(storage._deferred), 1)
        storage.batch_set_v1(keys, _indices(3), _info([]))
        self.assertEqual(storage._deferred, [])
        self.assertEqual(
            storage.batch_exists_v2(keys, [probe], _info([])).kv_hit_pages, 3
        )
        mamba.buf.zero_()
        self.assertEqual(
            storage.batch_get_v2([state], _info(keys[:2])), {PoolName.MAMBA: [True]}
        )
        self.assertTrue(torch.all(mamba.page(0) == 6))
        storage.close()

    def test_deferred_component_expires(self):
        mamba = _HostPool(4, 256, page_size=1)
        storage, _ = self._backend(mamba=mamba)
        storage.defer_s = 0.0
        keys = _keys(2, "orphan")
        state = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([0]),
            keys=keys[1:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        storage.batch_set_v2([state], _info(keys[:1]))
        self.assertEqual(len(storage._deferred), 1)
        storage.batch_set_v1(_keys(1, "other"), _indices(1), _info([]))
        self.assertEqual(storage._deferred, [])
        storage.close()

    def test_window_unknown_falls_back_to_pages(self):
        swa = _HostPool(16, 32)
        storage, _ = self._backend(window=0, swa=swa)
        storage.batch_exists(_keys(1), _info([]))
        self.assertNotIn("swa", storage.client.geometry["pools"])
        self.assertEqual(storage.pools[PoolName.SWA].kind, storage.full)
        storage.close()


if __name__ == "__main__":
    unittest.main()
