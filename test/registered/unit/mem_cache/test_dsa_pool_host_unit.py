import inspect
import unittest

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
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
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
            indexer_host = DSAIndexerPoolHost(
                decl=make_dsa_indexer_pool_decl(device_pool),
                anchor_host=mla_host,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
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

    def _round_trip_with_shared_topk_layers(self, layout: str, io_backend: str):
        page_size = 64
        skip = [False, True, False, True]
        device_pool = DSATokenToKVPool(
            size=page_size * 4,
            page_size=page_size,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=len(skip),
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
            skip_topk_layers=skip,
        )
        pin_memory = io_backend == "kernel"
        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        if pin_memory:
            ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            mla_host = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout=layout,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
                override_kv_cache_dim=device_pool.kv_cache_dim,
            )
            indexer_host = DSAIndexerPoolHost(
                decl=make_dsa_indexer_pool_decl(device_pool),
                anchor_host=mla_host,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        owned = [layer for layer, shared in enumerate(skip) if not shared]
        self.assertEqual(indexer_host.layer_num, len(owned))
        buffers = device_pool.index_k_with_scale_buffer
        for layer_id in owned:
            data = torch.arange(buffers[layer_id].numel(), device="cuda")
            buffers[layer_id].copy_(
                ((data + 7 * layer_id) % 256).view_as(buffers[layer_id])
            )

        # Index placement follows HiCacheController.move_indices / _move_write_operation.
        index_device = "cuda" if io_backend == "kernel" else "cpu"
        backup_host_device = (
            "cpu"
            if layout == "page_first" and indexer_host.can_use_write_back_jit
            else index_device
        )
        device_pages = torch.tensor([1, 2, 3], dtype=torch.int64)
        host_pages = torch.tensor([0, 1, 2], dtype=torch.int64)
        device_indices = self._token_indices_for_pages(
            device_pages, page_size, index_device
        )
        host_indices = self._token_indices_for_pages(
            host_pages, page_size, index_device
        )

        expected = {layer: buffers[layer][device_pages].clone() for layer in owned}
        indexer_host.backup_from_device_all_layer(
            device_pool,
            host_indices.to(backup_host_device),
            device_indices,
            io_backend,
        )
        for layer_id in owned:
            buffers[layer_id].zero_()
        for layer_id in range(len(skip)):
            indexer_host.load_to_device_per_layer(
                device_pool, host_indices, device_indices, layer_id, io_backend
            )
        torch.cuda.synchronize()

        for layer_id in owned:
            self.assertTrue(
                torch.equal(buffers[layer_id][device_pages], expected[layer_id])
            )
        for layer_id, shared in enumerate(skip):
            if shared:
                self.assertEqual(buffers[layer_id].shape[0], 0)

    @unittest.skipIf(is_hip(), "The HIP pool tests above run with page_size=1.")
    def test_shared_topk_layers_round_trip_through_host(self):
        """Bug regression (upstream #30531 review): shared-topk layers' zero-row
        index-K buffers must never enter the HiCache transfer kernels."""
        for layout, io_backend in (
            ("layer_first", "kernel"),
            ("page_first", "kernel"),
            ("layer_first", "direct"),
        ):
            with self.subTest(layout=layout, io_backend=io_backend):
                self._round_trip_with_shared_topk_layers(layout, io_backend)

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


if __name__ == "__main__":
    unittest.main()
