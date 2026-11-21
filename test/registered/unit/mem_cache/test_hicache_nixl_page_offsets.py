"""Unit tests for vectorized page-offset extraction in HiCacheNixl."""

import sys
import types
import unittest
from types import SimpleNamespace

import torch

# Allow running unit tests on CPU-only environments where the native `nixl`
# wheel is not installed, while using the real package when present.
try:
    import nixl._api  # noqa: F401
except ModuleNotFoundError:
    nixl_pkg = types.ModuleType("nixl")
    nixl_api = types.ModuleType("nixl._api")
    nixl_api.nixl_agent = object
    nixl_api.nixl_agent_config = object
    nixl_api.nixlBind = object
    nixl_pkg._api = nixl_api
    sys.modules.setdefault("nixl", nixl_pkg)
    sys.modules.setdefault("nixl._api", nixl_api)

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import (
    HiCacheNixl,
    _HybridPoolContext,
)


class _DummyHostPool:
    def __init__(self, page_size: int = 4, num_pages: int = 8, page_numel: int = 16):
        self.page_size = page_size
        self.page_numel = page_numel
        self.received_get_offsets = []
        self.received_set_offsets = []
        self.storage = {
            p * page_size: torch.full((page_numel,), float(p + 1), dtype=torch.float32)
            for p in range(num_pages)
        }

    def get_page_buffer_meta(self, host_indices: torch.Tensor):
        offsets = host_indices[:: self.page_size].tolist()
        ptrs = [1000 + off * 64 for off in offsets]
        sizes = [self.page_numel * 4 for _ in offsets]
        return ptrs, sizes

    def get_data_page(self, index, flat: bool = True):
        self.received_get_offsets.append(index)
        return self.storage[index]

    def set_from_flat_data_page(self, index, data_page: torch.Tensor):
        self.received_set_offsets.append(index)
        self.storage[index] = data_page.clone()


class TestHiCacheNixlPageOffsets(unittest.TestCase):
    def _make_stub_nixl(self, host_pool: _DummyHostPool, is_zero_copy: bool) -> HiCacheNixl:
        nixl = HiCacheNixl.__new__(HiCacheNixl)
        nixl.mem_pool_host = host_pool
        nixl.is_zero_copy = is_zero_copy
        nixl.is_mla_model = False
        nixl._logical_anchor = False
        nixl._bounce_page_bytes = host_pool.page_numel * 4
        nixl._bounce_set = torch.zeros(
            (16, host_pool.page_numel), dtype=torch.float32
        )
        nixl._bounce_get = torch.zeros(
            (16, host_pool.page_numel), dtype=torch.float32
        )
        nixl._hybrid_pool_ctx = {
            PoolName.MAMBA: _HybridPoolContext(
                host_pool=host_pool,
                is_zero_copy=is_zero_copy,
                bounce_set=nixl._bounce_set,
                bounce_get=nixl._bounce_get,
                bounce_page_bytes=nixl._bounce_page_bytes,
            )
        }
        nixl._get_suffixed_key = lambda key, pool_name=None: (
            f"{key}_{pool_name}" if pool_name else f"{key}_v1"
        )
        nixl._get_component_key = lambda key, pool_name=None: (
            f"{key}_{pool_name}" if pool_name else key
        )
        return nixl

    def test_prepare_pool_transfer_strided_page_offsets_zero_copy_and_bounce(self):
        page_size = 4
        host_pool = _DummyHostPool(page_size=page_size, num_pages=8)
        # Pages at base offsets 0, 12, 20 (each spanning page_size=4 contiguous indices)
        host_indices = torch.tensor(
            [0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23], dtype=torch.int64
        )
        keys = ["k0", "k1", "k2"]
        transfer = PoolTransfer(
            name=PoolName.MAMBA, keys=keys, host_indices=host_indices
        )

        nixl_zc = self._make_stub_nixl(host_pool, is_zero_copy=True)
        _, key_strs, host_buffers, page_offsets, mult = nixl_zc._prepare_pool_transfer(
            transfer, for_write=False
        )
        self.assertEqual(page_offsets, [0, 12, 20])
        self.assertTrue(all(type(off) is int for off in page_offsets))
        self.assertEqual(len(key_strs), 3)
        self.assertEqual(len(host_buffers), 3)
        self.assertEqual(mult, 1)

        nixl_bounce = self._make_stub_nixl(host_pool, is_zero_copy=False)
        _, key_strs_b, host_buffers_b, page_offsets_b, mult_b = (
            nixl_bounce._prepare_pool_transfer(transfer, for_write=True)
        )
        self.assertEqual(page_offsets_b, [0, 12, 20])
        self.assertTrue(all(type(off) is int for off in page_offsets_b))
        self.assertEqual(host_pool.received_get_offsets, [0, 12, 20])
        self.assertTrue(
            all(type(off) is int for off in host_pool.received_get_offsets)
        )
        self.assertEqual(len(key_strs_b), 3)
        self.assertEqual(len(host_buffers_b), 3)
        self.assertEqual(mult_b, 1)

    def test_v1_batch_preprocess_and_postprocess_pass_int_page_offsets(self):
        page_size = 4
        host_pool = _DummyHostPool(page_size=page_size, num_pages=8)
        nixl = self._make_stub_nixl(host_pool, is_zero_copy=False)

        host_indices = torch.tensor(
            [4, 5, 6, 7, 16, 17, 18, 19], dtype=torch.int64
        )
        keys = ["p1", "p4"]
        key_list, host_buffers = nixl._batch_preprocess(keys, host_indices, op="set")
        self.assertEqual(key_list, ["p1_v1", "p4_v1"])
        self.assertEqual(len(host_buffers), 2)
        self.assertEqual(host_pool.received_get_offsets, [4, 16])
        self.assertTrue(
            all(type(off) is int for off in host_pool.received_get_offsets)
        )

        nixl._bounce_get[0].fill_(42.0)
        nixl._bounce_get[1].fill_(99.0)
        results = nixl._batch_get_postprocess(host_indices, [True, True])
        self.assertEqual(results, [True, True])
        self.assertEqual(host_pool.received_set_offsets, [4, 16])
        self.assertTrue(
            all(type(off) is int for off in host_pool.received_set_offsets)
        )
        self.assertTrue(torch.all(host_pool.storage[4] == 42.0))
        self.assertTrue(torch.all(host_pool.storage[16] == 99.0))


if __name__ == "__main__":
    unittest.main()
