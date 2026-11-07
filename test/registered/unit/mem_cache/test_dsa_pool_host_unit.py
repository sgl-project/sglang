import inspect
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=41, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")


def _make_indexer_host(pool, anchor, *, packed_draft_device_pools=(), **kwargs):
    from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
        prepare_host_pool_config,
    )

    config = prepare_host_pool_config(
        root=pool,
        full_layer_mapping={i: i for i in range(pool.layer_num)},
        transfer_layer_id_max=pool.layer_num,
        transfer_page_size=anchor.page_size,
        packed_draft_device_pools=packed_draft_device_pools,
    ).sidecars[0]
    return DSAIndexerPoolHost(
        config.buffer_info,
        layer_mapping=config.device_to_host_layer,
        target_device_layer_num=pool.layer_num,
        num_host_pages=anchor.page_num,
        transfer_page_size=anchor.page_size,
        layout=anchor.layout,
        buffer_device=pool.device,
        dtype=pool.store_dtype,
        layer_sharded=pool.layer_shard_enabled,
        **kwargs,
    )


class TestDSAOffloadSignatures(unittest.TestCase):
    def test_cpu_copy_methods_accept_mamba_indices(self):
        for method_name in ("get_cpu_copy", "load_cpu_copy"):
            with self.subTest(method_name=method_name):
                signature = inspect.signature(getattr(DSATokenToKVPool, method_name))
                self.assertIn("mamba_indices", signature.parameters)


class TestDSAHiCacheTransfer(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for DSA host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("DSA host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

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

    def _run_device_to_host_indexer_copy(
        self, io_backend: str, use_fp4_indexer: bool = False
    ):
        page_size = 1 if is_hip() else 64
        layer_num = 2
        size = page_size * 4

        device_pool = DSATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
            use_fp4_indexer=use_fp4_indexer,
        )
        pin_memory = io_backend == "kernel"
        mla_host = MLATokenToKVPoolHost(
            buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=page_size,
            layout="layer_first",
            pin_memory=pin_memory,
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=device_pool.kv_cache_dim,
            device_capacity_tokens=(
                device_pool.size
                if device_pool.host_capacity_tokens is None
                else device_pool.host_capacity_tokens
            ),
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            target_layer_num=(
                (device_pool.layer_num + device_pool.layer_shard_size - 1)
                // device_pool.layer_shard_size
                if device_pool.layer_shard_enabled
                else device_pool.layer_num
            ),
            layer_shard_enabled=device_pool.layer_shard_enabled,
        )
        indexer_host = _make_indexer_host(
            device_pool,
            mla_host,
            pin_memory=pin_memory,
            device="cpu",
            allocator_type="default",
        )

        self.addCleanup(mla_host.destroy)
        self.addCleanup(indexer_host.destroy)
        for layer_id in range(layer_num):
            buf = device_pool.index_k_with_scale_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=torch.int32
            ).view_as(buf)
            buf.copy_((data + layer_id).remainder(251).to(torch.uint8))
            kv_buf = device_pool.kv_buffer[layer_id]
            kv_data = torch.arange(
                kv_buf.numel(), device=kv_buf.device, dtype=kv_buf.dtype
            ).view_as(kv_buf)
            kv_buf.copy_(kv_data + layer_id)

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

        mla_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        indexer_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )

        for layer_id in range(layer_num):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                got = indexer_host.index_k_with_scale_buffer[layer_id][host_page].cpu()
                expected = device_pool.index_k_with_scale_buffer[layer_id][
                    device_page
                ].cpu()
                self.assertTrue(torch.equal(got, expected))
                host_start = host_page * page_size
                device_start = device_page * page_size
                got_kv = mla_host.kv_buffer[layer_id][
                    host_start : host_start + page_size
                ].cpu()
                expected_kv = device_pool.kv_buffer[layer_id][
                    device_start : device_start + page_size
                ].cpu()
                self.assertTrue(torch.equal(got_kv, expected_kv))

        expected_index_pages = [
            buffer[device_pages].clone()
            for buffer in device_pool.index_k_with_scale_buffer
        ]
        for buffer in device_pool.index_k_with_scale_buffer:
            buffer.zero_()
        for layer_id in range(layer_num):
            indexer_host.load_to_device_per_layer(
                device_pool, host_indices, device_indices, layer_id, io_backend
            )
        torch.cuda.synchronize()
        for buffer, expected in zip(
            device_pool.index_k_with_scale_buffer, expected_index_pages, strict=True
        ):
            self.assertTrue(torch.equal(buffer[device_pages], expected))
            self.assertEqual(torch.count_nonzero(buffer[0]).item(), 0)

    @unittest.skipIf(
        is_hip(),
        '`io_backend="kernel"` path in MLATokenToKVPoolHost.backup_from_device_all_layer '
        "raises ValueError on AMD (only the `direct` IO backend is wired for ROCm). "
        "The other 62 tests in this file pass on AMD.",
    )
    def test_device_to_host_indexer_kernel(self):
        for fp4 in (False, True):
            with self.subTest(fp4=fp4):
                self._run_device_to_host_indexer_copy("kernel", fp4)

    @unittest.skipIf(
        is_hip(),
        "DSATokenToKVPool with page_size=1 (used on HIP) trips the ROCm 7.2.0 "
        "HIP-preshuffle assert `page_size % 16 == 0` during pool construction "
        "(seen on pr-test-amd-rocm720). The rest of this file passes on AMD.",
    )
    def test_device_to_host_indexer_direct(self):
        for fp4 in (False, True):
            with self.subTest(fp4=fp4):
                self._run_device_to_host_indexer_copy("direct", fp4)


class TestDSAHiCacheDirectKernel(unittest.TestCase):
    """`direct_kernel` must leave both host pools and device pools byte-identical to `direct`."""

    PAGE_SIZE = 64
    LAYER_NUM = 3

    def setUp(self):
        if not torch.cuda.is_available() or not is_cuda():
            self.skipTest("direct_kernel is CUDA-only.")
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def _device_pool(self, layer_num: int) -> DSATokenToKVPool:
        pool = DSATokenToKVPool(
            size=self.PAGE_SIZE * 12,
            page_size=self.PAGE_SIZE,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
        )
        gen = torch.Generator(device="cuda").manual_seed(layer_num)
        for buf in [*pool.kv_buffer, *pool.index_k_with_scale_buffer]:
            buf.view(torch.uint8).copy_(
                torch.randint(
                    0, 256, buf.shape, dtype=torch.uint8, device="cuda", generator=gen
                )
            )
        return pool

    def _host_pools(self, device_pool, draft_pool, layout="page_first_direct"):
        mla_host = MLATokenToKVPoolHost(
            buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.PAGE_SIZE,
            layout=layout,
            pin_memory=True,
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=device_pool.kv_cache_dim,
            draft_buffer_infos=(draft_pool.get_device_pool_infos()[0].buffer_info,),
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            target_layer_num=device_pool.layer_num,
        )
        indexer_host = _make_indexer_host(
            device_pool,
            mla_host,
            packed_draft_device_pools=(draft_pool,),
            pin_memory=True,
            device="cpu",
            allocator_type="default",
        )
        self.addCleanup(mla_host.destroy)
        self.addCleanup(_cuda_host_unregister, indexer_host.index_k_with_scale_buffer)
        for host in (mla_host.kv_buffer, indexer_host.index_k_with_scale_buffer):
            host.view(torch.uint8).copy_(
                torch.randint(
                    0,
                    256,
                    host.shape,
                    dtype=torch.uint8,
                    generator=torch.Generator().manual_seed(7),
                )
            )
        return mla_host, indexer_host

    def _indices(self, pages, device):
        return TestDSAHiCacheTransfer._token_indices_for_pages(
            torch.tensor(pages), self.PAGE_SIZE, device
        )

    def _device_state(self, *pools):
        return [
            buf.clone()
            for p in pools
            for buf in [*p.kv_buffer, *p.index_k_with_scale_buffer]
        ]

    def _load_all(
        self,
        host_pools,
        device_pool,
        draft_pool,
        host_indices,
        device_indices,
        io_backend,
    ):
        for host_pool in host_pools:
            for layer_id in range(self.LAYER_NUM):
                host_pool.load_to_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            host_pool.load_to_device_per_layer(
                draft_pool,
                host_indices,
                device_indices,
                self.LAYER_NUM,
                io_backend,
                is_draft=True,
            )

    def test_matches_direct_with_packed_draft(self):
        # Like L2TransferEngine, transfer on a side stream: cudaMemcpyBatchAsync
        # rejects the legacy default stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._run_matches_direct_with_packed_draft()

    def _run_matches_direct_with_packed_draft(self):
        device_pool = self._device_pool(self.LAYER_NUM)
        draft_pool = self._device_pool(1)
        reference = self._host_pools(device_pool, draft_pool)
        candidate = self._host_pools(device_pool, draft_pool)
        # Prepared through the group fan-out, as HiCacheController does at startup.
        HostPoolGroup(
            [
                PoolEntry(
                    name=name,
                    host_pool=host_pool,
                    device_pool=device_pool,
                    layer_mapper=lambda layer: layer,
                    is_primary_index_anchor=name == PoolName.KV,
                )
                for name, host_pool in zip((PoolName.KV, PoolName.INDEXER), candidate)
            ]
        ).prepare_io_backend("direct_kernel")
        self.assertEqual(candidate[0].layer_num, self.LAYER_NUM + 1)
        self.assertEqual(candidate[1].layer_num, self.LAYER_NUM + 1)

        device_pages, host_pages = [5, 1, 9], [3, 0, 7]
        for host_pool in reference:
            host_pool.backup_from_device_all_layer(
                device_pool,
                self._indices(host_pages, "cpu"),
                self._indices(device_pages, "cpu"),
                "direct",
            )
        for host_pool in candidate:
            host_pool.backup_from_device_all_layer(
                device_pool,
                self._indices(host_pages, "cuda"),
                self._indices(device_pages, "cuda"),
                "direct_kernel",
            )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(reference[0].kv_buffer, candidate[0].kv_buffer))
        self.assertTrue(
            torch.equal(
                reference[1].index_k_with_scale_buffer,
                candidate[1].index_k_with_scale_buffer,
            )
        )

        load_device_pages, load_host_pages = [2, 11], [7, 4]
        initial = self._device_state(device_pool, draft_pool)
        self._load_all(
            reference,
            device_pool,
            draft_pool,
            self._indices(load_host_pages, "cpu"),
            self._indices(load_device_pages, "cpu"),
            "direct",
        )
        torch.cuda.synchronize()
        expected = self._device_state(device_pool, draft_pool)
        for buf, init in zip(
            [
                b
                for p in (device_pool, draft_pool)
                for b in [*p.kv_buffer, *p.index_k_with_scale_buffer]
            ],
            initial,
        ):
            buf.copy_(init)
        self._load_all(
            candidate,
            device_pool,
            draft_pool,
            self._indices(load_host_pages, "cuda"),
            self._indices(load_device_pages, "cuda"),
            "direct_kernel",
        )
        torch.cuda.synchronize()
        for got, want in zip(self._device_state(device_pool, draft_pool), expected):
            self.assertTrue(torch.equal(got, want))

    def test_rejects_non_direct_layout(self):
        device_pool = self._device_pool(self.LAYER_NUM)
        draft_pool = self._device_pool(1)
        for host_pool in self._host_pools(
            device_pool, draft_pool, layout="layer_first"
        ):
            with self.assertRaisesRegex(ValueError, "page_first_direct"):
                host_pool.prepare_io_backend("direct_kernel")


if __name__ == "__main__":
    unittest.main()
