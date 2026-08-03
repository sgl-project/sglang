import unittest

import psutil
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    HICACHE_HOST_MEMORY_RESERVE_BYTES,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, stage="stage-b", runner_config="1-gpu-small-amd")


def _cuda_major() -> int:
    cuda = getattr(torch.version, "cuda", None)
    try:
        return int(cuda.split(".")[0]) if cuda else 0
    except ValueError:
        return 0


# direct+page_first_direct routes to transfer_kv_all_layer_direct_lf_pf, which on
# CUDA 13 throws (cudaErrorInvalidValue) instead of falling back. M3 uses kernel+layer_first.
_DIRECT_PF_BATCHCOPY_BROKEN_CUDA13 = _cuda_major() >= 13


class _FakeLayerTransferCounter:
    def __init__(self):
        self.waited_layers = []

    def wait_until(self, layer_id: int):
        self.waited_layers.append(layer_id)


def _make_cpu_minimax_sparse_pool(start_layer: int = 4) -> MiniMaxSparseKVPool:
    end_layer = start_layer + 4
    return MiniMaxSparseKVPool(
        size=8,
        page_size=4,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        idx_head_dim=3,
        dense_layer_ids=[start_layer, start_layer + 2],
        sparse_layer_ids=[start_layer + 1, start_layer + 3],
        disable_value_sparse_layer_ids=[start_layer + 1, start_layer + 3],
        device="cpu",
        start_layer=start_layer,
        end_layer=end_layer,
    )


class TestMiniMaxSparseHiCacheIntegration(unittest.TestCase):
    def test_hiradix_extra_pools_include_minimax_indexer(self):
        pool = _make_cpu_minimax_sparse_pool()
        cache = object.__new__(HiRadixCache)
        cache.cache_controller = object.__new__(HybridCacheController)
        cache.kv_cache = pool

        extra = HiRadixCache._get_extra_pools(cache)

        transfers = extra["extra_pools"]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].name, PoolName.INDEXER)
        self.assertEqual(transfers[0].indices_from_pool, PoolName.KV)
        self.assertEqual(transfers[0].hit_policy, PoolHitPolicy.ALL_PAGES)

    def test_index_k_waits_for_full_local_layer(self):
        pool = _make_cpu_minimax_sparse_pool()
        counter = _FakeLayerTransferCounter()
        pool.register_layer_transfer_counter(counter)

        pool.get_index_k_buffer(7)

        self.assertEqual(counter.waited_layers, [3])
        self.assertIsNone(pool.main_pool.layer_transfer_counter)
        self.assertIsNone(pool.index_k_pool.layer_transfer_counter)

    def test_main_kv_waits_on_minimax_wrapper(self):
        pool = _make_cpu_minimax_sparse_pool()
        counter = _FakeLayerTransferCounter()
        pool.register_layer_transfer_counter(counter)

        pool.get_kv_buffer(6)

        self.assertEqual(counter.waited_layers, [2])
        self.assertIsNone(pool.main_pool.layer_transfer_counter)
        self.assertIsNone(pool.index_k_pool.layer_transfer_counter)

    def test_k_only_host_pool_layout_contracts(self):
        if psutil.virtual_memory().available <= HICACHE_HOST_MEMORY_RESERVE_BYTES:
            self.skipTest("Not enough spare host memory for HiCache host pool tests.")

        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = _make_cpu_minimax_sparse_pool(start_layer=0)
                kv_host = MHATokenToKVPoolHost(
                    device_pool=pool.main_pool,
                    host_to_device_ratio=2.0,
                    host_size=0,
                    page_size=pool.page_size,
                    layout=layout,
                    pin_memory=False,
                    device="cpu",
                    allocator_type="default",
                )
                index_host = MHATokenToKOnlyPoolHost(
                    pool.index_k_pool,
                    kv_host,
                    layout=layout,
                    pin_memory=False,
                    device="cpu",
                    allocator_type="default",
                )

                if layout == "layer_first":
                    self.assertEqual(
                        index_host.k_buffer.shape,
                        (
                            index_host.layer_num,
                            index_host.size,
                            index_host.head_num,
                            index_host.head_dim,
                        ),
                    )
                elif layout == "page_first":
                    self.assertEqual(
                        index_host.k_buffer.shape,
                        (
                            index_host.size,
                            index_host.layer_num,
                            index_host.head_num,
                            index_host.head_dim,
                        ),
                    )
                else:
                    self.assertEqual(
                        index_host.k_buffer.shape,
                        (
                            index_host.page_num,
                            index_host.layer_num,
                            index_host.page_size,
                            index_host.head_num,
                            index_host.head_dim,
                        ),
                    )

                page_start = pool.page_size
                flat_page = torch.arange(
                    index_host.layer_num
                    * index_host.page_size
                    * index_host.head_num
                    * index_host.head_dim,
                    dtype=index_host.dtype,
                )
                index_host.set_from_flat_data_page(page_start, flat_page)
                self.assertTrue(
                    torch.equal(index_host.get_data_page(page_start), flat_page)
                )
                self.assertEqual(
                    index_host.get_dummy_flat_data_page().numel(), flat_page.numel()
                )
                self.assertIs(
                    index_host.get_hybrid_pool_buffer()[0], index_host.k_buffer
                )

                indices = torch.arange(
                    page_start,
                    page_start + pool.page_size,
                    dtype=torch.int64,
                )
                if layout == "layer_first":
                    with self.assertRaisesRegex(ValueError, "layer_first"):
                        index_host.get_page_buffer_meta(indices)
                    continue

                ptrs, sizes = index_host.get_page_buffer_meta(indices)
                self.assertEqual(len(ptrs), 1)
                expected_size = (
                    index_host.layer_num
                    * index_host.page_size
                    * index_host.head_num
                    * index_host.head_dim
                    * index_host.dtype.itemsize
                )
                self.assertEqual(sizes, [expected_size] * len(ptrs))


class TestMiniMaxSparseHiCacheTransfer(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for MiniMax sparse host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("MiniMax sparse host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    @staticmethod
    def _token_indices_for_pages(pages: torch.Tensor, page_size: int, device: str):
        parts = [
            torch.arange(
                int(page_id) * page_size,
                (int(page_id) + 1) * page_size,
                device=device,
                dtype=torch.int64,
            )
            for page_id in pages.tolist()
        ]
        return torch.cat(parts, dim=0)

    @staticmethod
    def _host_k_page(host_pool, layer_id: int, page_id: int, page_size: int):
        start = page_id * page_size
        if host_pool.layout == "layer_first":
            return host_pool.k_buffer[layer_id][start : start + page_size]
        if host_pool.layout == "page_first":
            return host_pool.k_buffer[start : start + page_size, layer_id]
        if host_pool.layout == "page_first_direct":
            return host_pool.k_buffer[page_id, layer_id]
        raise ValueError(f"Unsupported layout: {host_pool.layout}")

    @staticmethod
    def _host_v_page(host_pool, layer_id: int, page_id: int, page_size: int):
        start = page_id * page_size
        if host_pool.layout == "layer_first":
            return host_pool.v_buffer[layer_id][start : start + page_size]
        if host_pool.layout == "page_first":
            return host_pool.v_buffer[start : start + page_size, layer_id]
        if host_pool.layout == "page_first_direct":
            return host_pool.v_buffer[page_id, layer_id]
        raise ValueError(f"Unsupported layout: {host_pool.layout}")

    def _run_device_to_host_copy(self, io_backend: str, layout: str):
        page_size = 64
        layer_num = 4
        size = page_size * 4
        dense_layer_ids = [0, 1]
        sparse_layer_ids = [2, 3]

        device_pool = MiniMaxSparseKVPool(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=4,
            head_dim=64,
            idx_head_dim=128,
            dense_layer_ids=dense_layer_ids,
            sparse_layer_ids=sparse_layer_ids,
            disable_value_sparse_layer_ids=sparse_layer_ids,
            device="cuda",
            start_layer=0,
            end_layer=layer_num,
        )
        assert device_pool.index_kv_pool is None
        assert device_pool.index_k_pool is not None

        pin_memory = io_backend == "kernel"
        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        if pin_memory:
            ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            kv_host = MHATokenToKVPoolHost(
                device_pool=device_pool.main_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout=layout,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
            )
            index_host = MHATokenToKOnlyPoolHost(
                device_pool.index_k_pool,
                kv_host,
                layout=layout,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        for layer_id in range(layer_num):
            k_main, v_main = device_pool.get_kv_buffer(layer_id)
            k_main.copy_(torch.randn_like(k_main) + float(layer_id))
            v_main.copy_(torch.randn_like(v_main) + float(layer_id))

        for local_id, global_id in enumerate(sparse_layer_ids):
            idx_k = device_pool.index_k_pool.k_buffer[local_id]
            idx_k.copy_(torch.randn_like(idx_k) + float(global_id) + 100.0)

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor(
            [0, 1, 2],
            device="cuda" if io_backend == "kernel" else "cpu",
            dtype=torch.int64,
        )
        device_indices = self._token_indices_for_pages(
            device_pages, page_size, device="cuda"
        )
        host_indices = self._token_indices_for_pages(
            host_pages,
            page_size,
            device="cuda" if io_backend == "kernel" else "cpu",
        )
        # page_first main-KV backup (staged_write_back.cuh) needs CPU dst_indices,
        # index-k backup (hicache.cuh) needs CUDA indices — feed a CPU copy to main only.
        kv_host_indices = (
            host_indices.cpu()
            if (io_backend, layout) == ("kernel", "page_first")
            else host_indices
        )

        kv_host.backup_from_device_all_layer(
            device_pool.main_pool, kv_host_indices, device_indices, io_backend
        )
        index_host.backup_from_device_all_layer(
            device_pool.index_k_pool, host_indices, device_indices, io_backend
        )

        for layer_id in range(layer_num):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                device_start = device_page * page_size
                got_k = self._host_k_page(kv_host, layer_id, host_page, page_size).cpu()
                expected_k = device_pool.main_pool.k_buffer[layer_id][
                    device_start : device_start + page_size
                ].cpu()
                self.assertTrue(torch.equal(got_k, expected_k))
                got_v = self._host_v_page(kv_host, layer_id, host_page, page_size).cpu()
                expected_v = device_pool.main_pool.v_buffer[layer_id][
                    device_start : device_start + page_size
                ].cpu()
                self.assertTrue(torch.equal(got_v, expected_v))

        for local_id, global_id in enumerate(sparse_layer_ids):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                got = self._host_k_page(
                    index_host, local_id, host_page, page_size
                ).cpu()
                expected = device_pool.index_k_pool.k_buffer[local_id][
                    device_page * page_size : (device_page + 1) * page_size
                ].cpu()
                self.assertTrue(torch.equal(got, expected))

        # Round-trip H2D for one sparse index layer.
        reload_pages = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        host_device = "cuda" if io_backend == "kernel" else "cpu"
        reload_host_pages = torch.tensor([3, 0], device=host_device, dtype=torch.int64)
        reload_device_indices = self._token_indices_for_pages(
            reload_pages, page_size, device="cuda"
        )
        reload_host_indices = self._token_indices_for_pages(
            reload_host_pages, page_size, device=host_device
        )
        device_pool.index_k_pool.k_buffer[0].zero_()
        index_host.load_to_device_per_layer(
            device_pool.index_k_pool,
            reload_host_indices,
            reload_device_indices,
            0,
            io_backend,
        )
        for host_page, device_page in zip(
            reload_host_pages.tolist(), reload_pages.tolist()
        ):
            got = device_pool.index_k_pool.k_buffer[0][
                device_page * page_size : (device_page + 1) * page_size
            ].cpu()
            expected = self._host_k_page(index_host, 0, host_page, page_size).cpu()
            self.assertTrue(torch.equal(got, expected))

    def test_device_to_host_kernel_layer_first(self):
        self._run_device_to_host_copy(io_backend="kernel", layout="layer_first")

    @unittest.skipIf(
        is_hip(),
        "ROCm sgl-kernel transfer_kv_all_layer_lf_pf requires a CUDA dst-indices "
        "tensor for the kernel+page_first combo, while CUDA accepts the CPU "
        "dst-indices this combo feeds. The production io_backend=kernel + "
        "layer_first path is covered by test_device_to_host_kernel_layer_first.",
    )
    def test_device_to_host_kernel_page_first(self):
        self._run_device_to_host_copy(io_backend="kernel", layout="page_first")

    def test_device_to_host_direct_layer_first(self):
        self._run_device_to_host_copy(io_backend="direct", layout="layer_first")

    @unittest.skipIf(
        _DIRECT_PF_BATCHCOPY_BROKEN_CUDA13,
        "direct+page_first_direct host transfer hits cudaMemcpyBatchAsync "
        "cudaErrorInvalidValue on CUDA 13 (sgl-kernel transfer_kv_all_layer_direct_lf_pf "
        "throws instead of falling back to per-page copy); M3 production uses "
        "io_backend=kernel + layer_first, not this combo.",
    )
    def test_device_to_host_direct_page_first_direct(self):
        self._run_device_to_host_copy(io_backend="direct", layout="page_first_direct")


if __name__ == "__main__":
    unittest.main()
