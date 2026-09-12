import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.srt.mem_cache.pool_host import mha as mha_pool_host
from sglang.srt.mem_cache.pool_host import mla as mla_pool_host
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_register,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
    def __init__(self, fail_on_registration: int | None = None):
        self.registrations = []
        self.unregistrations = []
        self.fail_on_registration = fail_on_registration

    def cudaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        self.registrations.append((ptr, size, flags))
        if len(self.registrations) == self.fail_on_registration:
            return 1
        return 0

    def cudaHostUnregister(self, ptr: int) -> int:
        self.unregistrations.append(ptr)
        return 0

    def cudaGetErrorString(self, rc: int) -> str:
        return "injected error"


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

    def test_mamba_page_layouts_use_per_buffer_page_granularity(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = MambaPoolHost.__new__(MambaPoolHost)
                pool.layout = layout
                pool.size = 4
                pool.num_mamba_layers = 3
                pool.temporal_state_shape = (2, 5)
                pool.conv_state_shapes = [(7,), (2, 2)]
                pool.temporal_dtype = torch.float16
                pool.conv_dtype = torch.float32
                pool.device_pool = SimpleNamespace(device="cuda")
                pool.device = "cpu"
                pool.pin_memory = True
                pool.allocator = object()
                alloc = mock.Mock(
                    side_effect=lambda *args, **kwargs: torch.empty(
                        1, dtype=torch.uint8
                    )
                )

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
                    pool.init_kv_buffer()

                self.assertEqual(
                    [
                        call.kwargs["registration_granularity_bytes"]
                        for call in alloc.call_args_list
                    ],
                    [
                        3 * 2 * 5 * torch.float16.itemsize,
                        3 * 7 * torch.float32.itemsize,
                        3 * 2 * 2 * torch.float32.itemsize,
                    ],
                )

    def test_deepseek_v4_page_layouts_use_page_registration_granularity(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(pool="paged", layout=layout):
                alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))
                device_buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(3)]
                with (
                    mock.patch.object(
                        memory_pool_host,
                        "host_memory_budget_bytes",
                        return_value=1024**3,
                    ),
                    mock.patch.dict(ALLOC_MEMORY_FUNCS, {torch.device("cpu"): alloc}),
                ):
                    DeepSeekV4PagedHostPool(
                        pool_name="test",
                        device_buffers=device_buffers,
                        item_bytes=11,
                        num_host_pages=4,
                        slot_page_size=2,
                        layout=layout,
                    )

                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    3 * 11,
                )

            with self.subTest(pool="state", layout=layout):
                alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))
                state_pools = [
                    SimpleNamespace(
                        ring_size=2,
                        kv_score_buffer=SimpleNamespace(
                            kv_score=torch.empty((4, 3), dtype=torch.uint8)
                        ),
                    )
                    for _ in range(2)
                ]
                with (
                    mock.patch.object(
                        memory_pool_host,
                        "host_memory_budget_bytes",
                        return_value=1024**3,
                    ),
                    mock.patch.dict(ALLOC_MEMORY_FUNCS, {torch.device("cpu"): alloc}),
                ):
                    DeepSeekV4StateHostPool(
                        pool_name="test",
                        state_pools=state_pools,
                        num_host_pages=4,
                        swa_page_size=2,
                        layout=layout,
                    )

                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    2 * 2 * 3,
                )

    def test_k_only_mha_page_layouts_use_page_registration_granularity(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = MHATokenToKOnlyPoolHost.__new__(MHATokenToKOnlyPoolHost)
                pool.layout = layout
                pool.size = 8
                pool.page_num = 4
                pool.page_size = 2
                pool.layer_num = 3
                pool.head_num = 2
                pool.head_dim = 5
                pool.dtype = torch.float16
                pool.layout_dim = (
                    pool.layer_num * pool.head_num * pool.head_dim * pool.dtype.itemsize
                )
                pool.device_pool = SimpleNamespace(device="cuda")
                pool.device = "cpu"
                pool.pin_memory = True
                pool.allocator = object()
                alloc = mock.Mock(return_value=object())

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
                    pool.init_kv_buffer()

                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    pool.page_size * pool.layout_dim,
                )

    def test_asymmetric_mha_page_layouts_use_native_page_granularities(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = AsymmetricMHATokenToKVPoolHost.__new__(
                    AsymmetricMHATokenToKVPoolHost
                )
                pool.layout = layout
                pool.size = 8
                pool.page_num = 4
                pool.page_size = 2
                pool.layer_num = 3
                pool.head_num = 2
                pool.head_dim = 5
                pool.v_head_dim = 7
                pool.dtype = torch.float16
                pool.device_pool = SimpleNamespace(device="cuda")
                pool.device = "cpu"
                pool.pin_memory = True
                pool.allocator = object()
                alloc = mock.Mock(side_effect=[object(), object()])

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
                    pool.init_kv_buffer()

                self.assertEqual(
                    [
                        call.kwargs["registration_granularity_bytes"]
                        for call in alloc.call_args_list
                    ],
                    [
                        pool.page_size * pool._k_layout_dim(),
                        pool.page_size * pool._v_layout_dim(),
                    ],
                )

    def test_unregister_releases_every_registered_chunk_once(self):
        gib = 1024**3
        base = 0x10000000
        buffer = _FakeBuffer(base, 2 * gib + 17)
        cudart = _FakeCudart()

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            _cuda_host_register(buffer, registration_granularity_bytes=gib)
            _cuda_host_unregister(buffer)
            _cuda_host_unregister(buffer)

        self.assertEqual(
            cudart.unregistrations,
            [base + 2 * gib, base + gib, base],
        )

    def test_registration_failure_rolls_back_prior_chunks(self):
        gib = 1024**3
        base = 0x10000000
        buffer = _FakeBuffer(base, 2 * gib + 17)
        cudart = _FakeCudart(fail_on_registration=2)

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
            self.assertRaisesRegex(RuntimeError, "offset=1073741824"),
        ):
            _cuda_host_register(buffer, registration_granularity_bytes=gib)

        self.assertEqual(
            cudart.registrations,
            [(base, gib, 0), (base + gib, gib, 0)],
        )
        self.assertEqual(cudart.unregistrations, [base])

    def test_missing_copy_granularity_preserves_single_registration(self):
        gib = 1024**3
        base = 0x10000000
        total = 2 * gib + 17
        buffer = _FakeBuffer(base, total)
        cudart = _FakeCudart()

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            _cuda_host_register(buffer)

        self.assertEqual(cudart.registrations, [(base, total, 0)])

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
