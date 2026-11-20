import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8
from sglang.srt.mem_cache.pool_host.mha_mxfp8 import MHATokenToKVPoolMXFP8Host
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


class TestMXFP8HostAllocation(CustomTestCase):
    def test_constructor_counts_payload_and_scale_pages(self):
        pool = MHATokenToKVPoolMXFP8(
            size=256,
            page_size=128,
            dtype=torch.float8_e4m3fn,
            head_num=1,
            head_dim=128,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        host = MHATokenToKVPoolMXFP8Host(
            pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2,
            host_size=0,
            page_size=pool.page_size,
            layout="page_first",
            pin_memory=False,
            device_capacity_tokens=pool.size
            if pool.host_capacity_tokens is None
            else pool.host_capacity_tokens,
            start_layer=pool.start_layer,
            end_layer=pool.end_layer,
            write_back_staging=pool.hicache_write_back_staging,
        )
        self.addCleanup(host.destroy)
        expected_token_bytes = (
            2
            * pool.layer_num
            * pool.head_num
            * (pool.head_dim + pool.head_dim // pool.MXFP8_SCALE_BLOCK_SIZE)
        )
        self.assertEqual(host.size_per_token, expected_token_bytes)
        self.assertEqual(host.page_size, pool.page_size)
        self.assertTrue(host.can_use_write_back_jit)
        self.assertEqual(
            host.kv_buffer.nbytes + host.k_scale_host.nbytes + host.v_scale_host.nbytes,
            host.size * expected_token_bytes,
        )

    def test_host_reuses_prebudget_staging(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.pool_host.mha import (
            MHATokenToKVPoolHost,
            prepare_mha_write_back_staging,
        )

        pool = MHATokenToKVPool(
            size=256,
            page_size=16,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        info = pool.get_device_pool_infos()[0].buffer_info
        reserved = prepare_mha_write_back_staging(
            info,
            layer_num=2,
            page_size=16,
            page_capacity=64,
        )
        self.assertIs(
            prepare_mha_write_back_staging(
                info,
                layer_num=2,
                page_size=16,
                page_capacity=33,
                staging=reserved,
            ),
            reserved,
        )
        host = MHATokenToKVPoolHost(
            info,
            2,
            0,
            16,
            "page_first",
            pin_memory=False,
            device_capacity_tokens=256,
            start_layer=0,
            end_layer=2,
            write_back_staging=reserved,
        )
        self.addCleanup(host.destroy)
        self.assertTrue(host.can_use_write_back_jit)
        for actual, original in zip(
            (host.staging_k_buffer, host.staging_v_buffer), reserved, strict=True
        ):
            self.assertEqual(actual.data_ptr(), original.data_ptr())
            self.assertEqual(
                actual.untyped_storage().data_ptr(),
                original.untyped_storage().data_ptr(),
            )
            self.assertEqual(actual.shape[0], host.staging_token_capacity)
        self.assertIsNone(pool.hicache_write_back_staging)

    def test_payload_and_scales_restore_without_device_owner(self):
        """Captured scale buffers must restore without the removed owner helpers."""
        pool = MHATokenToKVPoolMXFP8(
            size=256,
            page_size=128,
            dtype=torch.float8_e4m3fn,
            head_num=1,
            head_dim=128,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        info = pool.get_device_pool_infos()[0].buffer_info
        host = MHATokenToKVPoolMXFP8Host(
            info,
            2,
            0,
            128,
            "page_first",
            device_capacity_tokens=256,
            start_layer=0,
            end_layer=2,
        )
        self.addCleanup(host.destroy)
        del pool
        payload = (*info.buffers.k, *info.buffers.v)
        scales = (*info.scales.k, *info.scales.v)
        for value, buffer in enumerate((*payload, *scales), 1):
            buffer.view(torch.uint8).fill_(value)
        expected = [b[128:256].view(torch.uint8).clone() for b in payload]
        expected += [b[1:2].view(torch.uint8).clone() for b in scales]
        device_indices = torch.arange(128, 256, device="cuda")
        host_indices = torch.arange(128)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            host.backup_from_device_all_layer(
                object(), host_indices, device_indices, "kernel"
            )
        stream.synchronize()
        for buffer in (*payload, *scales):
            buffer.view(torch.uint8).zero_()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for layer in range(2):
                host.load_to_device_per_layer(
                    object(), host_indices.cuda(), device_indices, layer, "kernel"
                )
        stream.synchronize()
        actual = [b[128:256].view(torch.uint8) for b in payload]
        actual += [b[1:2].view(torch.uint8) for b in scales]
        for before, after in zip(expected, actual, strict=True):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
