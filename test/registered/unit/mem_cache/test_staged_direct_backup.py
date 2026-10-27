"""Byte-exact pageable D2H packing, including fragmented host allocations."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.mem_cache.pool_host.staged_direct import (
    StagedDirectBackup,
    contiguous_page_runs,
)


class TestStagedDirectBackup(unittest.TestCase):
    def test_runs_preserve_source_order(self):
        self.assertEqual(list(contiguous_page_runs([])), [])
        self.assertEqual(
            list(contiguous_page_runs([4, 5, 1, 8, 9])),
            [(0, 2, 4), (2, 3, 1), (3, 5, 8)],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fragmented_pages_and_reused_staging(self):
        for dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                page, layers, dim = 64, 3, 576
                device = torch.device("cuda:0")
                src = [
                    torch.randint(0, 16, (512, 1, dim), device=device).to(dtype)
                    for _ in range(layers)
                ]
                ptrs = torch.tensor(
                    [s.data_ptr() for s in src], dtype=torch.uint64, device=device
                )
                host = torch.zeros((10, layers, page, 1, dim), dtype=dtype)
                reference = host.clone()
                backup = StagedDirectBackup(
                    host,
                    page,
                    device,
                    max_bytes=2 * layers * page * dim * host.element_size(),
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                for dest in ([4, 5, 1, 8, 9], [0, 2, 3, 6, 7]):
                    indices = torch.randperm(512)[: len(dest) * page]
                    host_indices = (
                        torch.tensor(dest)[:, None] * page + torch.arange(page)
                    ).flatten()
                    for layer, buf in enumerate(src):
                        values = (
                            buf.cpu()
                            .view(torch.uint8)[indices]
                            .reshape(len(dest), page, 1, dim * buf.element_size())
                        )
                        # Compare bytes: float8 CPU advanced indexing is not
                        # supported on every supported PyTorch version.
                        reference.view(torch.uint8)[dest, layer] = values.view(
                            torch.uint8
                        )
                    with torch.cuda.stream(stream):
                        backup.backup(ptrs, host_indices, indices)
                    stream.synchronize()
                    self.assertTrue(
                        torch.equal(host.view(torch.uint8), reference.view(torch.uint8))
                    )
                before = host.clone()
                invalid = torch.arange(page)
                invalid[-1] += 1
                with self.assertRaises(ValueError):
                    backup.backup(ptrs, invalid, torch.arange(page))
                self.assertTrue(
                    torch.equal(host.view(torch.uint8), before.view(torch.uint8))
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_mla_host_pool_dispatch_roundtrip(self):
        import os
        from unittest.mock import patch

        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        with patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}):
            device = MLATokenToKVPool(
                size=256,
                page_size=64,
                dtype=torch.bfloat16,
                kv_lora_rank=16,
                qk_rope_head_dim=8,
                layer_num=2,
                device="cuda",
                enable_memory_saver=False,
            )
            host = MLATokenToKVPoolHost(
                device.get_device_pool_infos()[0].buffer_info,
                target_layer_num=2,
                device_capacity_tokens=256,
                start_layer=0,
                end_layer=2,
                host_to_device_ratio=2,
                host_size=0,
                page_size=64,
                layout="page_first_direct",
                pin_memory=True,
            )
            try:
                original = []
                for layer in device.kv_buffer:
                    layer.copy_(torch.randint(0, 16, layer.shape, device="cuda"))
                    original.append(layer[:128].clone())
                hi = torch.arange(128, 256)
                di = torch.arange(128)
                host.backup_from_device_all_layer(device, hi, di, "direct")
                torch.cuda.synchronize()
                for layer in device.kv_buffer:
                    layer.zero_()
                for i in range(2):
                    host.load_to_device_per_layer(device, hi, di, i, "direct")
                torch.cuda.synchronize()
                for before, after in zip(original, device.kv_buffer):
                    self.assertTrue(torch.equal(before, after[:128]))
                self.assertFalse(host.pin_memory)
                self.assertIsNotNone(host._staged_direct_backup)
                self.assertIsNotNone(host._staged_direct_load)
            finally:
                host.destroy()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_dsa_indexer_host_pool_dispatch_roundtrip(self):
        import os
        from unittest.mock import patch

        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        from sglang.srt.mem_cache.pool_host.dsa import (
            DSAIndexerPoolHost,
            make_dsa_indexer_pool_decl,
        )
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
        from sglang.srt.runtime_context import get_parallel

        with (
            patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}),
            get_parallel().override(attn_dcp_size=1),
        ):
            device = DSATokenToKVPool(
                size=256,
                page_size=64,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                kv_cache_dim=576,
                layer_num=2,
                device="cuda",
                index_head_dim=128,
                enable_memory_saver=False,
            )
            anchor = MLATokenToKVPoolHost(
                device.get_device_pool_infos()[0].buffer_info,
                target_layer_num=2,
                device_capacity_tokens=256,
                start_layer=0,
                end_layer=2,
                host_to_device_ratio=2,
                host_size=0,
                page_size=64,
                layout="page_first_direct",
                pin_memory=True,
            )
            host = DSAIndexerPoolHost(make_dsa_indexer_pool_decl(device), anchor)
            try:
                original = []
                for layer in device.index_k_with_scale_buffer:
                    layer.random_(0, 256)
                    original.append(layer[:2].clone())
                hi, di = torch.arange(128, 256), torch.arange(128)
                host.backup_from_device_all_layer(device, hi, di, "direct")
                torch.cuda.synchronize()
                for layer in device.index_k_with_scale_buffer:
                    layer.zero_()
                for i in range(2):
                    host.load_to_device_per_layer(device, hi, di, i, "direct")
                torch.cuda.synchronize()
                for before, after in zip(original, device.index_k_with_scale_buffer):
                    self.assertTrue(torch.equal(before, after[:2]))
                self.assertIsNotNone(host._staged_direct_backup)
                self.assertIsNotNone(host._staged_direct_load)
            finally:
                host.destroy()
                anchor.destroy()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_packed_indexer_rows_with_gpu_indices(self):
        # Scales and index keys share opaque rows; preserve all byte patterns.
        rows, layers, row_bytes = 32, 3, 8448
        src = [
            torch.randint(0, 256, (rows, row_bytes), dtype=torch.uint8, device="cuda")
            for _ in range(layers)
        ]
        ptrs = torch.tensor(
            [b.data_ptr() for b in src], dtype=torch.uint64, device="cuda"
        )
        host = torch.zeros((rows, layers, 1, row_bytes), dtype=torch.uint8)
        backup = StagedDirectBackup(
            host, 1, src[0].device, max_bytes=2 * layers * row_bytes
        )
        cpu_indices = torch.tensor([17, 3, 9, 3, 21])
        indices = cpu_indices.cuda()
        dest = torch.tensor([8, 9, 1, 20, 21])
        backup.backup(ptrs, dest, indices)
        torch.cuda.synchronize()
        reference = torch.zeros_like(host)
        for layer, buf in enumerate(src):
            reference[dest, layer, 0] = buf.cpu()[cpu_indices]
        self.assertTrue(torch.equal(host, reference))


if __name__ == "__main__":
    unittest.main()
