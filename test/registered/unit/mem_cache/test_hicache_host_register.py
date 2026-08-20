import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host import mha as mha_pool_host
from sglang.srt.mem_cache.pool_host import mla as mla_pool_host
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_register,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeBuffer:
    def __init__(self, base: int, size: int):
        self._base = base
        self._size = size

    def data_ptr(self) -> int:
        return self._base

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1


class _FakeCudart:
    def __init__(self):
        self.registrations = []

    def cudaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        self.registrations.append((ptr, size, flags))
        return 0


class TestHiCacheHostRegister(unittest.TestCase):
    def test_dsa_page_layouts_with_draft_use_page_registration_granularity(self):
        target_buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(3)]
        draft_buffer = torch.empty(1, dtype=torch.uint8)

        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
                host.device_pool = SimpleNamespace(
                    device="cpu", index_k_with_scale_buffer=target_buffers
                )
                host.mtp_draft_device_pools = [
                    SimpleNamespace(index_k_with_scale_buffer=[draft_buffer])
                ]
                host.layout = layout
                host.layer_num = 4
                host.indexer_page_num = 3
                host.indexer_page_stride_size = 512
                host.indexer_layout_dim = host.layer_num * host.indexer_page_stride_size
                host.indexer_dtype = torch.uint8
                host.device = "cpu"
                host.pin_memory = True
                host.allocator = mock.sentinel.allocator
                alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}):
                    host.init_kv_buffer()

                self.assertEqual(len(host.packed_device_index_buffers), 4)
                self.assertIs(host.packed_device_index_buffers[-1], draft_buffer)
                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    host.indexer_layout_dim,
                )

    def test_page_first_direct_mla_uses_page_registration_granularity(self):
        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.layout = "page_first_direct"
        pool.page_num = 4
        pool.layer_num = 3
        pool.page_size = 2
        pool.kv_cache_dim = 5
        pool.dtype = torch.float16
        pool.device_pool = SimpleNamespace(device="cuda")
        pool.device = "cpu"
        pool.pin_memory = True
        pool.allocator = object()
        alloc = mock.Mock(return_value=object())

        with mock.patch.dict(mla_pool_host.ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
            pool.init_kv_buffer()

        self.assertEqual(
            alloc.call_args.kwargs["registration_granularity_bytes"],
            pool.page_size * pool.layer_num * pool.kv_cache_dim * pool.dtype.itemsize,
        )

    def test_page_first_direct_mha_uses_page_registration_granularity(self):
        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.layout = "page_first_direct"
        pool.page_num = 4
        pool.layer_num = 3
        pool.page_size = 2
        pool.head_num = 2
        pool.head_dim = 4
        pool.dtype = torch.float16
        pool.device_pool = SimpleNamespace(device="cuda")
        pool.device = "cpu"
        pool.pin_memory = True
        pool.allocator = object()
        alloc = mock.Mock(return_value=object())

        with mock.patch.dict(mha_pool_host.ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
            pool.init_kv_buffer()

        self.assertEqual(
            alloc.call_args.kwargs["registration_granularity_bytes"],
            pool.page_size
            * pool.layer_num
            * pool.head_num
            * pool.head_dim
            * pool.dtype.itemsize,
        )

    def test_registration_boundaries_honor_page_copy_granularity(self):
        mib = 1024**2
        gib = 1024**3
        base = 0x10000000
        total = 2500 * mib
        page_copy_bytes = 300 * mib
        cudart = _FakeCudart()

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            _cuda_host_register(
                _FakeBuffer(base, total),
                registration_granularity_bytes=page_copy_bytes,
            )

        aligned_chunk = 900 * mib
        self.assertLessEqual(aligned_chunk, gib)
        self.assertEqual(
            cudart.registrations,
            [
                (base, aligned_chunk, 0),
                (base + aligned_chunk, aligned_chunk, 0),
                (base + 2 * aligned_chunk, 700 * mib, 0),
            ],
        )
        for ptr, _, _ in cudart.registrations:
            self.assertEqual((ptr - base) % page_copy_bytes, 0)


if __name__ == "__main__":
    unittest.main()
