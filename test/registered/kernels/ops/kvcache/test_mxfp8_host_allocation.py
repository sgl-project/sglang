import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8
from sglang.srt.mem_cache.pool_host.mha_mxfp8 import MHATokenToKVPoolMXFP8Host
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


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
            pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=pool.page_size,
            layout="page_first",
            pin_memory=False,
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


if __name__ == "__main__":
    unittest.main()
