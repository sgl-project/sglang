import inspect
import os
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")


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

    @staticmethod
    def _host_pools(device_pool, layout, pin_memory, allocator_type="default"):
        """The MLA KV host pool over ``device_pool`` and its DSA indexer pool."""
        mla_host = MLATokenToKVPoolHost(
            buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=device_pool.page_size,
            layout=layout,
            pin_memory=pin_memory,
            device="cpu",
            allocator_type=allocator_type,
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
        indexer_host = DSAIndexerPoolHost(
            decl=make_dsa_indexer_pool_decl(device_pool),
            anchor_host=mla_host,
            pin_memory=pin_memory,
            device="cpu",
            allocator_type=allocator_type,
        )
        return mla_host, indexer_host

    def _run_device_to_host_indexer_copy(self, io_backend: str):
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
        )
        pin_memory = io_backend == "kernel"
        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        if pin_memory:
            ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            mla_host, indexer_host = self._host_pools(
                device_pool, layout="layer_first", pin_memory=pin_memory
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        for layer_id in range(layer_num):
            buf = device_pool.index_k_with_scale_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=torch.uint8
            ).view_as(buf)
            buf.copy_((data + layer_id) % 256)
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

    @unittest.skipIf(
        is_hip(),
        '`io_backend="kernel"` path in MLATokenToKVPoolHost.backup_from_device_all_layer '
        "raises ValueError on AMD (only the `direct` IO backend is wired for ROCm). "
        "The other 62 tests in this file pass on AMD.",
    )
    def test_device_to_host_indexer_kernel(self):
        self._run_device_to_host_indexer_copy(io_backend="kernel")

    @unittest.skipIf(
        is_hip(),
        "DSATokenToKVPool with page_size=1 (used on HIP) trips the ROCm 7.2.0 "
        "HIP-preshuffle assert `page_size % 16 == 0` during pool construction "
        "(seen on pr-test-amd-rocm720). The rest of this file passes on AMD.",
    )
    def test_device_to_host_indexer_direct(self):
        self._run_device_to_host_indexer_copy(io_backend="direct")

    @unittest.skipIf(
        is_hip(),
        "DSATokenToKVPool with page_size=64 trips the ROCm 7.2.0 HIP-preshuffle "
        "assert during pool construction.",
    )
    def test_direct_kernels_follow_padded_host_pages(self):
        """The page_first_direct kernels address host pages through the host
        tensor's strides, so pages padded for NIXL O_DIRECT round-trip
        GPU -> host -> GPU and the padding is never written."""
        page_size, layer_num = 64, 2
        # Pages of 2 layers x 64 tokens: KV rows of 136 bf16 (34,816 B) and
        # 132-byte index keys (16,896 B), padded to 9 and 5 blocks of 4 KiB.
        device_pool = DSATokenToKVPool(
            size=page_size * 8,
            page_size=page_size,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=8,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=136,
            index_head_dim=128,
        )
        with (
            mock.patch.dict(os.environ, {"SGLANG_HICACHE_NIXL_USE_DIRECT_IO": "1"}),
            get_context().override_server_args(),
        ):
            mla_host, indexer_host = self._host_pools(
                device_pool,
                layout="page_first_direct",
                pin_memory=True,
                allocator_type="nixl",
            )
        self.addCleanup(indexer_host.destroy)
        self.addCleanup(mla_host.destroy)
        self.assertEqual(mla_host.page_stride_bytes, 9 * 4096)
        self.assertEqual(indexer_host.page_stride_bytes, 5 * 4096)

        device_layers = [
            buffer.view(torch.uint8)
            for buffers in (
                device_pool.kv_buffer,
                device_pool.index_k_with_scale_buffer,
            )
            for buffer in buffers
        ]
        for raw in device_layers:
            raw.copy_(
                torch.randint(0, 256, raw.shape, device="cuda", dtype=torch.uint8)
            )
        expected = [raw.clone() for raw in device_layers]

        src_pages = torch.tensor([1, 2, 3])
        dst_pages = torch.tensor([4, 5, 6])
        src = self._token_indices_for_pages(src_pages, page_size, device="cuda")
        host = self._token_indices_for_pages(torch.tensor([0, 2, 4]), page_size, "cpu")
        dst = self._token_indices_for_pages(dst_pages, page_size, device="cuda")
        # cudaMemcpyBatchAsync, which these kernels use, refuses the legacy
        # default stream; the HiCache controller runs them on its own streams.
        with torch.cuda.stream(torch.cuda.Stream()):
            for host_pool in (mla_host, indexer_host):
                host_pool.backup_from_device_all_layer(device_pool, host, src, "direct")
                for layer_id in range(layer_num):
                    host_pool.load_to_device_per_layer(
                        device_pool, host, dst, layer_id, "direct"
                    )
        torch.cuda.synchronize()

        # KV rows are per token, index keys per page.
        rows_per_page = [page_size] * layer_num + [1] * layer_num
        for raw, want, rows in zip(device_layers, expected, rows_per_page):
            for s, d in zip(src_pages.tolist(), dst_pages.tolist()):
                self.assertTrue(
                    torch.equal(
                        raw[d * rows : (d + 1) * rows], want[s * rows : (s + 1) * rows]
                    )
                )
        for buffer, stride in (
            (mla_host.kv_buffer, mla_host.page_stride_bytes),
            (indexer_host.index_k_with_scale_buffer, indexer_host.page_stride_bytes),
        ):
            page_bytes = buffer[0].numel() * buffer.element_size()
            step = stride // buffer.element_size()
            raw = torch.as_strided(buffer, (buffer.shape[0], step), (step, 1))
            padding = raw.view(torch.uint8)[:, page_bytes:]
            self.assertEqual(padding.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
